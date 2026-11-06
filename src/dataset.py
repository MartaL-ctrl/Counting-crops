"""Dataset GWHD 2021 (pliki parquet z Hugging Face) dla modeli map gęstości."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.density import boxes_to_points, density_map, downsample_density

# Statystyki ImageNet - wymagane, bo CSRNet używa wag VGG16 wytrenowanych na ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_PATTERNS = {"train": "train-*.parquet", "validation": "validation-*.parquet", "test": "test-*.parquet"}


def split_files(data_dir: str | Path, split: str) -> list[Path]:
    """Zwraca posortowaną listę plików parquet danego podziału (train/validation/test)."""
    files = sorted(Path(data_dir).glob(SPLIT_PATTERNS[split]))
    if not files:
        raise FileNotFoundError(f"brak plików {SPLIT_PATTERNS[split]} w {data_dir}")
    return files


class GWHDDensityDataset(Dataset):
    """Próbki: obraz (znormalizowany tensor), mapa gęstości i liczba kłosów.

    Tryb treningowy (``crop_size`` podany): losowy wycinek ``crop_size x crop_size``.
    Tryb ewaluacji (``crop_size=None``): cały obraz 1024x1024.

    Mapa gęstości jest liczona z punktów leżących w wycinku, więc jej suma jest
    dokładnie równa liczbie kłosów w wycinku (``count``). Mapa ma rozdzielczość
    zmniejszoną ``downsample`` razy (CSRNet: 8) i NIE jest skalowana - ewentualne
    mnożenie przez stałą odbywa się w kodzie treningowym.

    Cały podział jest trzymany w RAM jako skompresowane bajty PNG (train: ok. 0,7 GB).
    Na Windows przy ``num_workers > 0`` każdy proces roboczy dostaje kopię, więc
    lokalnie zalecane jest ``num_workers=0``.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        sigma: float,
        crop_size: int | None = None,
        downsample: int = 8,
        indices: Sequence[int] | None = None,
    ) -> None:
        if crop_size is not None and crop_size % downsample:
            raise ValueError(f"crop_size={crop_size} musi być podzielny przez downsample={downsample}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        # numery obrazów w całym podziale (stabilne identyfikatory do zapisu predykcji)
        self.source_indices: list[int] = list(indices) if indices is not None else list(range(len(df)))
        df = df.iloc[self.source_indices].reset_index(drop=True)

        self.image_bytes: list[bytes] = [img["bytes"] for img in df["image"]]
        self.points: list[np.ndarray] = [boxes_to_points(list(o["boxes"])) for o in df["objects"]]
        self.domains: list[str] = df["domain"].tolist()
        self.sigma = sigma
        self.crop_size = crop_size
        self.downsample = downsample
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_bytes)

    def __getitem__(self, idx: int) -> dict:
        img = np.asarray(Image.open(io.BytesIO(self.image_bytes[idx])).convert("RGB"))
        h, w = img.shape[:2]
        points = self.points[idx]

        if self.crop_size is not None:
            cs = self.crop_size
            # np.random (a nie torch) - ziarno w procesach roboczych ustawia seed_worker
            top = np.random.randint(0, h - cs + 1)
            left = np.random.randint(0, w - cs + 1)
            img = img[top:top + cs, left:left + cs]
            inside = (
                (points[:, 0] >= left) & (points[:, 0] < left + cs)
                & (points[:, 1] >= top) & (points[:, 1] < top + cs)
            )
            points = points[inside] - np.array([left, top], dtype=np.float32)
            h = w = cs

        if h % self.downsample or w % self.downsample:
            raise ValueError(f"rozmiar obrazu {(h, w)} niepodzielny przez {self.downsample}")

        density = downsample_density(density_map(points, h, w, self.sigma), self.downsample)

        image = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W), 0..1
        image = (image - self.mean) / self.std
        return {
            "image": image,
            "density": torch.from_numpy(density).unsqueeze(0),  # (1, H/ds, W/ds)
            "count": torch.tensor(len(points), dtype=torch.float32),
            "domain": self.domains[idx],
            "index": self.source_indices[idx],
        }
