"""Ewaluacja wytrenowanego CSRNet: liczba kłosów per obraz, MAE i RMSE łącznie, per domena
i per grupa liczby kłosów.

Użycie:
    python evaluate.py --checkpoint outputs/<run>/best.pt                    # cały zbiór walidacyjny
    python evaluate.py --checkpoint outputs/smoke/best.pt --subset outputs/smoke/subsets.json
    python evaluate.py --checkpoint outputs/<run>/best.pt --split test --allow-test

Zbiór testowy wymaga flagi ``--allow-test``: używamy go jednokrotnie, do końcowej oceny
wybranego modelu, a nie do wybierania modelu ani hiperparametrów.

Grupy liczby kłosów: progi to kwantyle rozkładu liczby kłosów w CAŁYM zbiorze uczącym
(3657 obrazów), bez zaglądania do walidacji i testu. Obraz trafia do grupy ``g``, gdy
``próg[g-1] < liczba <= próg[g]``.

Wyniki w ``<katalog checkpointu>/eval_<split>/`` (albo ``--out``):
    predictions_<split>.csv  - per obraz: index, image_name, domain, true_count, pred_count,
                               error (pred - true), abs_error, group
    metrics_<split>.csv      - scope (all / domain / group), name, n_images, mae, rmse,
                               mean_error (> 0: zawyżanie, < 0: zaniżanie), mean_true_count
    count_groups.json        - progi grup, sposób ich wyznaczenia i checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset import GWHDDensityDataset, load_split
from src.model import CSRNet
from src.utils import count_metrics, load_checkpoint
from train import evaluate, resolve, write_csv

LOCAL_DATA_ZIP = "data/raw/zenodo/gwhd_2021.zip"
METRIC_FIELDS = ["scope", "name", "n_images", "mae", "rmse", "mean_error", "mean_true_count"]


def count_group_edges(data_zip: Path, n_groups: int) -> list[float]:
    """Progi grup = kwantyle liczby kłosów w zbiorze uczącym (n_groups - 1 progów)."""
    counts = load_split(data_zip, "train")["count"].to_numpy()
    return np.quantile(counts, np.linspace(0, 1, n_groups + 1)[1:-1]).tolist()


def group_labels(edges: list[float]) -> list[str]:
    """Czytelne etykiety przedziałów, np. ``['<=35', '35-52', '>52']`` (lewy koniec otwarty)."""
    fmt = [f"{e:g}" for e in edges]
    middle = [f"{a}-{b}" for a, b in zip(fmt, fmt[1:])]
    return [f"<={fmt[0]}", *middle, f">{fmt[-1]}"]


def summarize(rows: list[dict], scope: str, key: str | None, order: list[str] | None = None) -> list[dict]:
    """Metryki dla wszystkich obrazów (``key=None``) albo osobno dla każdej wartości ``key``.

    ``order`` - kolejność wierszy (np. grupy od najmniejszej liczby kłosów); domyślnie alfabetyczna.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault("all" if key is None else str(r[key]), []).append(r)
    out = []
    for name in [n for n in order if n in groups] if order else sorted(groups):
        g = groups[name]
        m = count_metrics([r["pred_count"] for r in g], [r["true_count"] for r in g])
        out.append({"scope": scope, "name": name, "n_images": len(g), "mae": m["mae"], "rmse": m["rmse"],
                    "mean_error": float(np.mean([r["error"] for r in g])),
                    "mean_true_count": float(np.mean([r["true_count"] for r in g]))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="np. outputs/<run>/best.pt")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true", help="wymagane dla --split test")
    parser.add_argument("--subset", help="subsets.json z treningu: tylko obrazy użyte w walidacji")
    parser.add_argument("--n-groups", type=int, default=3, help="liczba grup liczby kłosów (domyślnie tercyle)")
    parser.add_argument("--data-zip", help="archiwum GWHD 2021 (domyślnie data_zip z konfiguracji w checkpoincie)")
    parser.add_argument("--out", help="katalog wyników (domyślnie <katalog checkpointu>/eval_<split>)")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        parser.error("zbiór testowy tylko do końcowej oceny - dodaj --allow-test, jeśli to ten moment")
    if args.subset and args.split != "validation":
        parser.error("--subset dotyczy tylko zbioru walidacyjnego")

    ckpt_path = resolve(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    data_zip = resolve(args.data_zip or cfg["data_zip"])
    if not args.data_zip and not data_zip.exists():  # checkpoint z Colaba ma ścieżkę /content/...
        data_zip = resolve(LOCAL_DATA_ZIP)
        print("data_zip z checkpointu niedostępny - używam", data_zip)
    scale = float(cfg["density_scale"])
    out_dir = resolve(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = None
    if args.subset:
        indices = json.loads(resolve(args.subset).read_text())["validation"]
    dataset = GWHDDensityDataset(data_zip, args.split, sigma=cfg["sigma"], crop_size=None,
                                 downsample=cfg["downsample"], indices=indices, kernel=cfg.get("kernel", "fixed"),
                                 box_sigma_factor=cfg.get("box_sigma_factor", 0.25))

    model = CSRNet(pretrained=False).to(device)  # wagi i tak z checkpointu - bez pobierania VGG16
    model.load_state_dict(ckpt["model"])

    edges = count_group_edges(data_zip, args.n_groups)
    labels = group_labels(edges)
    print(f"Urządzenie: {device} | {args.split}: {len(dataset)} obrazów | checkpoint: epoka {ckpt['epoch']}")
    print("Progi grup (kwantyle train):", ", ".join(f"{e:g}" for e in edges), "->", labels)

    _, rows = evaluate(model, dataset, device, scale, progress_every=50)
    names = dict(zip(dataset.source_indices, dataset.image_names))
    for r in rows:
        r["image_name"] = names[r["index"]]
        r["error"] = r["pred_count"] - r["true_count"]
        r["group"] = labels[int(np.searchsorted(edges, r["true_count"], side="left"))]

    pred_fields = ["index", "image_name", "domain", "true_count", "pred_count", "error", "abs_error", "group"]
    metrics = summarize(rows, "all", None) + summarize(rows, "domain", "domain") + summarize(rows, "group", "group", labels)
    write_csv(out_dir / f"predictions_{args.split}.csv", rows, pred_fields)
    write_csv(out_dir / f"metrics_{args.split}.csv", metrics, METRIC_FIELDS)
    (out_dir / "count_groups.json").write_text(json.dumps({
        "edges": edges, "labels": labels, "rule": "prog[g-1] < true_count <= prog[g]",
        "source": "kwantyle liczby kłosów w całym zbiorze uczącym", "n_groups": args.n_groups,
        "checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "split": args.split, "subset": args.subset,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for m in metrics:
        print(f"{m['scope']:>6} | {m['name']:<14} | n={m['n_images']:5d} | MAE {m['mae']:7.2f} | "
              f"RMSE {m['rmse']:7.2f} | śr. błąd {m['mean_error']:+7.2f}")
    print("Wyniki:", out_dir)


if __name__ == "__main__":
    main()
