"""Trening CSRNet na GWHD 2021 (mapy gęstości).

Użycie:
    python train.py --config configs/smoke.yaml
    python train.py --config configs/smoke.yaml --resume outputs/smoke/last.pt

Wyniki w ``<output_dir>/<run_name>/``:
    config.yaml               - użyta konfiguracja
    subsets.json              - numery obrazów użytych z train/validation
    history.csv               - strata i metryki po każdej epoce
    last.pt, best.pt          - checkpoint bieżący (co epokę) i najlepszy (wg val MAE)
    predictions_val_best.csv  - predykcje per obraz walidacyjny dla najlepszego modelu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.dataset import GWHDDensityDataset, load_split
from src.model import CSRNet
from src.utils import (
    count_metrics,
    get_rng_state,
    load_checkpoint,
    save_checkpoint,
    seed_worker,
    set_rng_state,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
HISTORY_FIELDS = ["epoch", "train_loss", "train_mae", "val_mae", "val_rmse", "epoch_time_s"]


def resolve(path: str | Path) -> Path:
    """Ścieżka względna -> względem katalogu repozytorium."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def pick_indices(total: int, n: int | None, rng: np.random.Generator) -> list[int]:
    """Losowy (powtarzalny) podzbiór numerów obrazów; ``None`` = wszystkie."""
    if n is None or n >= total:
        return list(range(total))
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def train_one_epoch(model, loader, optimizer, device, scale: float) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["density"].to(device) * scale
        preds = model(images)
        assert preds.shape == targets.shape, f"wyjście {preds.shape} != cel {targets.shape}"
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, dataset, device, scale: float,
             progress_every: int | None = None) -> tuple[dict[str, float], list[dict]]:
    """Liczenie na całych obrazach (po jednym). Zwraca metryki i predykcje per obraz.

    ``progress_every``: co ile obrazów wypisać postęp i szacowany czas do końca (None - nie wypisuje).
    """
    model.eval()
    rows = []
    start = time.time()
    for i in range(len(dataset)):
        if progress_every and i > 0 and i % progress_every == 0:
            left = (time.time() - start) / i * (len(dataset) - i)
            print(f"{time.strftime('%H:%M')}  {i}/{len(dataset)} obrazów, zostało ok. {left / 60:.0f} min",
                  flush=True)
        s = dataset[i]
        pred = model(s["image"].unsqueeze(0).to(device)).sum().item() / scale
        rows.append({"index": s["index"], "domain": s["domain"], "true_count": s["count"].item(),
                     "pred_count": pred, "abs_error": abs(pred - s["count"].item())})
    metrics = count_metrics([r["pred_count"] for r in rows], [r["true_count"] for r in rows])
    return metrics, rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="plik YAML z konfiguracją")
    parser.add_argument("--resume", help="checkpoint (np. outputs/<run>/last.pt), od którego wznowić")
    args = parser.parse_args()

    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve(cfg["output_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = float(cfg["density_scale"])

    # --- dane ---
    data_zip = resolve(cfg["data_zip"])
    rng = np.random.default_rng(cfg["seed"])
    train_idx = pick_indices(len(load_split(data_zip, "train")), cfg["n_train"], rng)
    val_idx = pick_indices(len(load_split(data_zip, "validation")), cfg["n_val"], rng)
    (out_dir / "subsets.json").write_text(json.dumps({"train": train_idx, "validation": val_idx}))

    common = {"zip_path": data_zip, "sigma": cfg["sigma"], "downsample": cfg["downsample"]}
    # brak klucza w starszych konfiguracjach = dotychczasowe zachowanie (sam losowy wycinek)
    train_ds = GWHDDensityDataset(split="train", crop_size=cfg["crop_size"], indices=train_idx,
                                  augmentation=cfg.get("augmentation", "crop_only"), **common)
    val_ds = GWHDDensityDataset(split="validation", crop_size=None, indices=val_idx, **common)
    train_eval_ds = (GWHDDensityDataset(split="train", crop_size=None, indices=train_idx, **common)
                     if cfg.get("eval_train") else None)

    loader_gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        worker_init_fn=seed_worker, generator=loader_gen)

    # --- model ---
    model = CSRNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    start_epoch, best_mae = 1, math.inf
    history = []
    if args.resume:
        ckpt = load_checkpoint(resolve(args.resume), map_location=device)
        if ckpt["config"] != cfg:
            print("UWAGA: konfiguracja różni się od zapisanej w checkpoincie")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_mae = ckpt["epoch"] + 1, ckpt["best_val_mae"]
        set_rng_state(ckpt["rng"])
        loader_gen.set_state(ckpt["loader_generator"].cpu())
        # wiersze historii z epok, które zostaną powtórzone, są usuwane
        history = [r for r in read_history(out_dir / "history.csv") if int(r["epoch"]) < start_epoch]
        print(f"Wznowienie od epoki {start_epoch} (najlepsze val MAE: {best_mae:.3f})")
    else:
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"Urządzenie: {device} | train: {len(train_ds)} obrazów | val: {len(val_ds)} obrazów")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, device, scale)
        row = {"epoch": epoch, "train_loss": train_loss, "train_mae": "", "val_mae": "", "val_rmse": ""}

        val_metrics = None
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            val_metrics, val_rows = evaluate(model, val_ds, device, scale)
            row.update(val_mae=val_metrics["mae"], val_rmse=val_metrics["rmse"])
            if train_eval_ds is not None:
                row["train_mae"] = evaluate(model, train_eval_ds, device, scale)[0]["mae"]

        row["epoch_time_s"] = round(time.time() - t0, 1)
        history.append(row)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_mae,
            "val_metrics": val_metrics,
            "config": cfg,
            "rng": get_rng_state(),
            "loader_generator": loader_gen.get_state(),
        }
        if val_metrics is not None and val_metrics["mae"] < best_mae:
            best_mae = state["best_val_mae"] = val_metrics["mae"]
            save_checkpoint(out_dir / "best.pt", state)
            write_csv(out_dir / "predictions_val_best.csv", val_rows, list(val_rows[0]))
        save_checkpoint(out_dir / "last.pt", state)
        write_csv(out_dir / "history.csv", history, HISTORY_FIELDS)

        msg = f"epoka {epoch:3d} | loss {train_loss:.5f} | {row['epoch_time_s']}s"
        if val_metrics is not None:
            msg += f" | val MAE {val_metrics['mae']:.2f} RMSE {val_metrics['rmse']:.2f}"
            if row["train_mae"] != "":
                msg += f" | train MAE {row['train_mae']:.2f}"
        print(msg, flush=True)

    print(f"Koniec. Najlepsze val MAE: {best_mae:.3f} | wyniki: {out_dir}")


if __name__ == "__main__":
    main()
