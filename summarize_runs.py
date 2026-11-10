"""Zestawienie treningów: najlepsze val MAE oraz średnia z ostatnich ewaluacji.

Najlepsze val MAE to minimum z ok. 30 zaszumionych pomiarów, więc częściowo zależy od
szczęścia. Średnia (i odchylenie standardowe) z ``--last`` ostatnich ewaluacji opisuje
poziom, na którym model się ustabilizował - porównujemy warianty obiema miarami.

Użycie:
    python summarize_runs.py                                    # wszystkie outputs/*/ poza smoke*
    python summarize_runs.py --runs outputs/csrnet_crop_only outputs/csrnet_thesis_aug

Wynik: tabela w konsoli i ``outputs/runs_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from train import resolve

CONFIG_FIELDS = ["augmentation", "kernel", "crop_size", "lr_schedule", "sigma", "epochs", "seed"]
# brak klucza w konfiguracji = zachowanie sprzed jego dodania
CONFIG_DEFAULTS = {"augmentation": "crop_only", "kernel": "fixed", "lr_schedule": "constant"}


def summarize_run(run_dir: Path, last: int) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    h = pd.read_csv(run_dir / "history.csv").dropna(subset=["val_mae"])
    best = h.loc[h["val_mae"].idxmin()]
    tail = h.tail(last)
    row = {"run": run_dir.name}
    row.update({k: cfg.get(k, CONFIG_DEFAULTS.get(k)) for k in CONFIG_FIELDS})
    row.update({
        "epochs_done": int(h["epoch"].max()),
        "n_evals": len(h),
        "best_epoch": int(best["epoch"]),
        "best_val_mae": best["val_mae"],
        "best_val_rmse": best["val_rmse"],
        f"last{last}_val_mae_mean": tail["val_mae"].mean(),
        f"last{last}_val_mae_std": tail["val_mae"].std(),
        f"last{last}_val_rmse_mean": tail["val_rmse"].mean(),
        "last_epochs": f"{int(tail['epoch'].min())}-{int(tail['epoch'].max())}",
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", help="katalogi treningów (domyślnie wszystkie w outputs/)")
    parser.add_argument("--last", type=int, default=5, help="liczba ostatnich ewaluacji do średniej")
    parser.add_argument("--out", default="outputs/runs_summary.csv")
    args = parser.parse_args()

    if args.runs:
        run_dirs = [resolve(r) for r in args.runs]
    else:
        # smoke* to testy działania kodu - ich wyników nie raportujemy
        run_dirs = sorted(p.parent for p in resolve("outputs").glob("*/history.csv")
                          if not p.parent.name.startswith("smoke"))
    run_dirs = [d for d in run_dirs if (d / "config.yaml").exists()]
    if not run_dirs:
        raise SystemExit("brak treningów (katalogów z history.csv i config.yaml)")

    df = pd.DataFrame([summarize_run(d, args.last) for d in run_dirs])
    out = resolve(args.out)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 2):
        print(df.drop(columns=["sigma", "seed"]).to_string(index=False))
    print("\nZapisano:", out)


if __name__ == "__main__":
    main()
