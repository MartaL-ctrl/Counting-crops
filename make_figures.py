"""Rysunki do pracy (PDF do LaTeX-a + PNG do podglądu) w ``outputs/figures/``.

Użycie:
    python make_figures.py data                      # rysunki o danych (rozdz. 3) - bez modelu
    python make_figures.py run --run outputs/<run>   # rysunki wyników jednego lub kilku treningów
    python make_figures.py run --run outputs/csrnet_crop_only --run outputs/csrnet_thesis_aug

Tryb ``data`` czyta adnotacje (CSV z archiwum) i kilka obrazów; nie potrzebuje GPU ani checkpointu.
Tryb ``run`` czyta ``history.csv`` i wyniki ``evaluate.py`` (``eval_<split>/``) z katalogów treningów.
Rysunki z treningu testowego (smoke) służą tylko sprawdzeniu skryptu - nie wstawiamy ich do pracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from evaluate import count_group_edges  # noqa: E402
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, GWHDDensityDataset, load_split  # noqa: E402
from src.density import clip_boxes, density_map, downsample_density  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = ROOT / "data" / "raw" / "zenodo" / "gwhd_2021.zip"
OUT = ROOT / "outputs" / "figures"

SPLITS = ["train", "validation", "test"]
SPLIT_PL = {"train": "uczący", "validation": "walidacyjny", "test": "testowy"}
# stała kolejność kolorów kategorii (podzbiory, potem kolejne treningi)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SPLIT_COLOR = dict(zip(SPLITS, SERIES))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
POINT_COLOR = "#ff3b30"  # punkty/ramki na zdjęciach - kontrast z zielenią łanu
SIGMA, DOWNSAMPLE = 10.0, 8

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})
WIDTH = 6.3  # [cal] szerokość kolumny tekstu A4 z marginesami 2,5 cm


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("zapisano", OUT / f"{name}.pdf")


def denormalize(image) -> np.ndarray:
    """Tensor (3, H, W) po normalizacji ImageNet -> tablica (H, W, 3) w [0, 1]."""
    img = image.numpy().transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return img.clip(0, 1)


def pl(x: float, digits: int = 1) -> str:
    """Liczba z przecinkiem dziesiętnym (polska notacja w pracy)."""
    return f"{x:.{digits}f}".replace(".", ",")


def image_axes(ax) -> None:
    """Zdjęcie/mapa bez osi, siatki i ramki."""
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def overlay(ax, density: np.ndarray, extent: tuple, vmax: float | None = None) -> None:
    """Mapa gęstości na zdjęciu; piksele bliskie zeru przezroczyste (nie barwią całego zdjęcia)."""
    masked = np.ma.masked_less(density, density.max() * 0.05 if density.max() > 0 else 1)
    ax.imshow(masked, cmap="Reds", alpha=0.65, extent=extent, vmin=0, vmax=vmax, interpolation="nearest")


# --------------------------------------------------------------------------- dane (rozdz. 3)

def fig_count_hist(dfs: dict[str, pd.DataFrame], edges: list[float]) -> None:
    """Rozkład liczby kłosów na obraz w podzbiorach + progi grup (tercyle zbioru uczącego)."""
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.6), sharex=True)
    bins = np.arange(0, max(df["count"].max() for df in dfs.values()) + 6, 5)
    for ax, split in zip(axes, SPLITS):
        c = dfs[split]["count"]
        ax.hist(c, bins=bins, color=SPLIT_COLOR[split], edgecolor="white", linewidth=0.8)
        for e in edges:
            ax.axvline(e + 0.5, color=INK, linestyle="--", linewidth=1)
        ax.set_ylabel("obrazy")
        ax.set_title(f"zbiór {SPLIT_PL[split]} ({len(c)} obrazów, mediana {np.median(c):g}, "
                     f"średnio {pl(c.mean())})", loc="left")
    top = axes[0].get_ylim()[1]
    for e in edges:
        axes[0].text(e + 2, top * 0.92, f"próg {e:g}", fontsize=8, color=INK)
    axes[-1].set_xlabel("liczba kłosów na obrazie")
    fig.tight_layout()
    save(fig, "dane_rozklad_liczby_klosow")


def nice_group_labels(edges: list[float]) -> list[str]:
    """Etykiety grup do rysunków: liczby kłosów są całkowite, więc (34, 54] = 35–54."""
    return [f"≤ {edges[0]:g}", f"{edges[0] + 1:g}–{edges[1]:g}", f"> {edges[1]:g}"]


def fig_count_groups(dfs: dict[str, pd.DataFrame], edges: list[float]) -> pd.DataFrame:
    """Udział grup liczby kłosów w podzbiorach (słupki skumulowane do 100%) + tabela CSV."""
    labels = nice_group_labels(edges)
    rows = []
    for split in SPLITS:
        g = np.searchsorted(edges, dfs[split]["count"].to_numpy(), side="left")
        for k, lab in enumerate(labels):
            rows.append({"podzbior": SPLIT_PL[split], "grupa": lab, "obrazy": int((g == k).sum()),
                         "procent": 100 * float((g == k).mean())})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "dane_grupy_liczby_klosow.csv", index=False)

    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    shades = ["#9ec5f4", "#3987e5", "#1c5cab"]  # jedna barwa, jaśniej = mniej kłosów
    for i, split in enumerate(SPLITS):
        left = 0.0
        for k, lab in enumerate(labels):
            p = table[(table.podzbior == SPLIT_PL[split]) & (table.grupa == lab)]["procent"].item()
            ax.barh(i, p, left=left, color=shades[k], edgecolor="white", linewidth=2,
                    label=lab if i == 0 else None)
            if p >= 6:
                ax.text(left + p / 2, i, f"{p:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if k else INK)
            left += p
    ax.set_yticks(range(3), [SPLIT_PL[s] for s in SPLITS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("udział obrazów [%]")
    ax.grid(axis="y", visible=False)
    ax.legend(title="liczba kłosów", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "dane_grupy_liczby_klosow")
    return table


def fig_domains(dfs: dict[str, pd.DataFrame]) -> None:
    """Liczba obrazów w każdej domenie, kolor = podzbiór."""
    rows = []
    for split in SPLITS:
        vc = dfs[split]["domain"].value_counts()
        rows += [(split, d, n) for d, n in vc.items()]
    fig, ax = plt.subplots(figsize=(WIDTH, 7.6))
    y = 0
    for split in SPLITS:
        part = [r for r in rows if r[0] == split]
        for _, d, n in part:
            ax.barh(y, n, color=SPLIT_COLOR[split], height=0.75)
            ax.text(n + 4, y, str(n), va="center", fontsize=7, color=MUTED)
            y += 1
        y += 0.8
    names = [d for split in SPLITS for (s, d, _) in rows if s == split]
    pos = []
    y = 0
    for split in SPLITS:
        k = sum(1 for r in rows if r[0] == split)
        pos += list(np.arange(y, y + k))
        y += k + 0.8
    ax.set_yticks(pos, names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("liczba obrazów")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=SPLIT_COLOR[s]) for s in SPLITS],
              labels=[f"{SPLIT_PL[s]} ({dfs[s]['domain'].nunique()} domen)" for s in SPLITS], loc="lower right")
    save(fig, "dane_domeny")


def fig_box_sizes(dfs: dict[str, pd.DataFrame]) -> None:
    """Rozkład rozmiaru kłosa: pierwiastek z pola ramki (po przycięciu), osobno dla podzbiorów."""
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 4.2), sharex=True)
    bins = np.arange(0, 250, 5)
    for ax, split in zip(axes, SPLITS):
        b = np.concatenate([clip_boxes(x, 1024, 1024) for x in dfs[split]["boxes"]])
        size = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
        ax.hist(size, bins=bins, color=SPLIT_COLOR[split], edgecolor="white", linewidth=0.8)
        ax.axvline(np.median(size), color=INK, linestyle="--", linewidth=1)
        ax.set_title(f"zbiór {SPLIT_PL[split]}: mediana {np.median(size):.0f} px "
                     f"({len(size)} ramek; {pl(100 * (size >= 250).mean())}% ≥ 250 px poza wykresem)", loc="left")
        ax.set_ylabel("ramki")
    axes[-1].set_xlabel("rozmiar kłosa = √(szerokość × wysokość ramki) [px]")
    fig.tight_layout()
    save(fig, "dane_rozmiar_klosow")


def fig_density_example(zip_path: Path, dfs: dict[str, pd.DataFrame]) -> None:
    """Etapy przygotowania etykiety: ramki -> punkty -> mapa gęstości -> mapa zmniejszona 8x."""
    df = dfs["train"]
    idx = int((df["count"] - df["count"].median()).abs().idxmin())  # obraz o typowej liczbie kłosów
    ds = GWHDDensityDataset(zip_path, "train", sigma=SIGMA, indices=[idx])
    img = ds.read_image(0)
    boxes = clip_boxes(df.loc[idx, "boxes"], 1024, 1024)
    pts = df.loc[idx, "points"]
    dens = density_map(pts, 1024, 1024, SIGMA)
    small = downsample_density(dens, DOWNSAMPLE)

    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.9))
    axes[0].imshow(img)
    for x1, y1, x2, y2 in boxes:
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=POINT_COLOR, linewidth=0.6))
    axes[0].set_title(f"(a) ramki ({len(boxes)})")
    axes[1].imshow(img)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=4, color=POINT_COLOR, linewidths=0)
    axes[1].set_title("(b) środki ramek")
    axes[2].imshow(dens, cmap="Blues")
    axes[2].set_title(f"(c) mapa 1024×1024\nsuma = {pl(dens.sum(), 2)}")
    axes[3].imshow(small, cmap="Blues", interpolation="nearest")
    axes[3].set_title(f"(d) mapa 128×128\nsuma = {pl(small.sum(), 2)}")
    for ax in axes:
        image_axes(ax)
    fig.tight_layout()
    save(fig, "dane_przyklad_mapy_gestosci")
    print(f"  obraz: {df.loc[idx, 'image_name']} ({df.loc[idx, 'domain']}), {len(pts)} kłosów, sigma={SIGMA}")


def fig_augmentation(zip_path: Path, dfs: dict[str, pd.DataFrame]) -> None:
    """Ten sam losowy wycinek bez i z profilem thesis_aug (odbicie, jasność, kontrast) + mapy."""
    df = dfs["train"]
    idx = [int(i) for i in df.index[(df["count"] >= 40) & (df["count"] <= 60)][:3]]
    plain = GWHDDensityDataset(zip_path, "train", sigma=SIGMA, crop_size=256, indices=idx, augmentation="crop_only")
    aug = GWHDDensityDataset(zip_path, "train", sigma=SIGMA, crop_size=256, indices=idx, augmentation="thesis_aug")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.4))
    for k in range(3):
        # dla przejrzystości rysunku wybieramy ziarno, przy którym thesis_aug odbija wycinek
        for seed in range(1000 + 50 * k, 1050 + 50 * k):
            np.random.seed(seed)
            a = plain[k]
            np.random.seed(seed)
            b = aug[k]
            if a["count"] > 0 and not np.array_equal(a["density"].numpy(), b["density"].numpy()):
                break
        for row, (s, name) in enumerate([(a, "crop_only"), (b, "thesis_aug")]):
            ax = axes[row, k]
            ax.imshow(denormalize(s["image"]))
            overlay(ax, s["density"][0].numpy(), extent=(0, 256, 256, 0))
            ax.set_title(f"{name}\nliczba kłosów {s['count']:.0f}, suma mapy {pl(s['density'].sum(), 2)}",
                         fontsize=7.5)
            image_axes(ax)
    fig.tight_layout()
    save(fig, "dane_augmentacja")


def run_data(zip_path: Path) -> None:
    dfs = {s: load_split(zip_path, s) for s in SPLITS}
    edges = count_group_edges(zip_path, 3)
    fig_count_hist(dfs, edges)
    print(fig_count_groups(dfs, edges).to_string(index=False))
    fig_domains(dfs)
    fig_box_sizes(dfs)
    fig_density_example(zip_path, dfs)
    fig_augmentation(zip_path, dfs)


# --------------------------------------------------------------------------- wyniki (rozdz. 5)

def load_run(run_dir: Path, split: str) -> dict:
    cfg_path = run_dir / "config.yaml"
    name = run_dir.name
    if cfg_path.exists():
        import yaml
        name = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("run_name", name)
    ev = run_dir / f"eval_{split}"
    return {
        "name": name,
        "history": pd.read_csv(run_dir / "history.csv"),
        "pred": pd.read_csv(ev / f"predictions_{split}.csv") if ev.exists() else None,
        "metrics": pd.read_csv(ev / f"metrics_{split}.csv") if ev.exists() else None,
        "groups": json.loads((ev / "count_groups.json").read_text(encoding="utf-8")) if ev.exists() else None,
    }


def fig_learning_curves(runs: list[dict], tag: str) -> None:
    """Strata treningowa i val MAE w funkcji epoki (dwa osobne wykresy, nie dwie osie y)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    for r, color in zip(runs, SERIES):
        h = r["history"]
        axes[0].plot(h["epoch"], h["train_loss"], color=color, linewidth=1.5, label=r["name"])
        ev = h.dropna(subset=["val_mae"])
        axes[1].plot(ev["epoch"], ev["val_mae"], "o-", color=color, linewidth=1.5, markersize=4, label=r["name"])
        best = ev.loc[ev["val_mae"].idxmin()]
        axes[1].annotate(pl(best["val_mae"], 2), (best["epoch"], best["val_mae"]), textcoords="offset points",
                         xytext=(0, -12), ha="center", fontsize=7, color=INK)
    axes[0].set(title="strata treningowa (MSE)", xlabel="epoka", yscale="log")
    axes[1].set(title="MAE na zbiorze walidacyjnym", xlabel="epoka", ylabel="MAE [kłosy]")
    if len(runs) > 1:
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"wyniki_krzywe_uczenia{tag}")


def fig_scatter(runs: list[dict], split: str, tag: str) -> None:
    """Przewidziana vs prawdziwa liczba kłosów per obraz (po jednym panelu na trening)."""
    runs = [r for r in runs if r["pred"] is not None]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(min(WIDTH, 3.1 * len(runs)), 3.1), squeeze=False)
    lim = max(max(r["pred"]["true_count"].max(), r["pred"]["pred_count"].max()) for r in runs) * 1.05
    for ax, r, color in zip(axes[0], runs, SERIES):
        p = r["pred"]
        m = r["metrics"].query("scope == 'all'").iloc[0]
        ax.plot([0, lim], [0, lim], color=MUTED, linestyle="--", linewidth=1)
        ax.scatter(p["true_count"], p["pred_count"], s=9, color=color, alpha=0.6, linewidths=0)
        for e in r["groups"]["edges"]:
            ax.axvline(e + 0.5, color=GRID, linewidth=1, zorder=0)
        ax.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel="prawdziwa liczba kłosów",
               ylabel="przewidziana liczba kłosów")
        ax.set_title(f"{r['name']} (zbiór {SPLIT_PL[split]}, n={len(p)})\nMAE {pl(m['mae'], 2)}, RMSE {pl(m['rmse'], 2)}", fontsize=8)
    fig.tight_layout()
    save(fig, f"wyniki_rozrzut_{split}{tag}")


def fig_group_mae(runs: list[dict], split: str, tag: str) -> None:
    """MAE w grupach liczby kłosów; słupki obok siebie dla kolejnych treningów (np. CSRNet vs YOLO)."""
    runs = [r for r in runs if r["metrics"] is not None]
    if not runs:
        return
    labels = runs[0]["groups"]["labels"]
    x = np.arange(len(labels))
    width = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for j, (r, color) in enumerate(zip(runs, SERIES)):
        g = r["metrics"].query("scope == 'group'").set_index("name").reindex(labels)
        bars = ax.bar(x + (j - (len(runs) - 1) / 2) * width, g["mae"], width * 0.92, color=color, label=r["name"])
        for b, n in zip(bars, g["n_images"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n:.0f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    edges = runs[0]["groups"]["edges"]
    ax.set_xticks(x, [f"{lab} kłosów" for lab in nice_group_labels(edges)] if len(edges) == 2 else labels)
    ax.set_ylabel(f"MAE [kłosy], zbiór {SPLIT_PL[split]}")
    ax.grid(axis="x", visible=False)
    if len(runs) > 1:
        ax.legend(fontsize=7)
    save(fig, f"wyniki_mae_grupy_{split}{tag}")


def run_runs(run_dirs: list[Path], split: str, tag: str) -> None:
    runs = [load_run(d, split) for d in run_dirs]
    fig_learning_curves(runs, tag)
    fig_scatter(runs, split, tag)
    fig_group_mae(runs, split, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("data", help="rysunki o danych (rozdz. 3)")
    d.add_argument("--data-zip", default=str(DEFAULT_ZIP))
    r = sub.add_parser("run", help="rysunki wyników treningów (rozdz. 5)")
    r.add_argument("--run", action="append", required=True, help="katalog treningu (można podać kilka razy)")
    r.add_argument("--split", choices=["validation", "test"], default="validation")
    r.add_argument("--tag", default="", help="dopisek do nazw plików, np. _smoke")
    args = parser.parse_args()

    if args.mode == "data":
        run_data(Path(args.data_zip))
    else:
        run_runs([Path(p) if Path(p).is_absolute() else ROOT / p for p in args.run], args.split, args.tag)


if __name__ == "__main__":
    main()
