"""Dataset GWHD 2021 (oryginalne wydanie z Zenodo: archiwum ZIP z obrazami i plikami CSV)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.density import boxes_to_points, clip_boxes, density_map, downsample_density

# Statystyki ImageNet - wymagane, bo CSRNet używa wag VGG16 wytrenowanych na ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ZIP_ROOT = "gwhd_2021/"
SPLIT_CSV = {"train": "competition_train.csv", "validation": "competition_val.csv", "test": "competition_test.csv"}
IMAGE_SIZE = 1024  # wszystkie obrazy GWHD 2021 mają 1024x1024 (sprawdzone dla 6515 plików)

# Profile augmentacji stosowane PO losowym wycinku (wycinek ustawia osobno crop_size).
# hflip_p - prawdopodobieństwo odbicia w poziomie; brightness/contrast - maksymalna względna
# zmiana (0.15 = mnożnik losowany jednostajnie z [0.85, 1.15]).
AUGMENTATION_PROFILES = {
    "crop_only": {"hflip_p": 0.0, "brightness": 0.0, "contrast": 0.0},
    "thesis_aug": {"hflip_p": 0.5, "brightness": 0.15, "contrast": 0.15},
}

# W CSV trzy nazwy plików występują dwukrotnie (w dwóch domenach, z różnymi ramkami), a w ZIP
# są trzy pliki, do których nie odwołuje się żaden wiersz. Przypisanie ustalono wzrokowo
# (ramki pasują do kłosów tylko na jednym z obrazów) - szczegóły w docs/dane_gwhd.md.
NAME_FIXES = {
    ("d88963636d49127bda0597ef73f1703e92d6f111caefc44902d5932b8cd3fa94.png", "Arvalis_10"):
        "8cc5870f73527da07937acc806002e1272e6200095c656ca326a680a90fab507.png",
    ("1961bcf453d5b2206c428c1c14fe55d1f26f3c655db0a2b6a83094476e8edb5b.png", "Arvalis_10"):
        "b588e1b55fbf8c4c6af08886013c0c36b70dd617fb1a5070829295a5c3ab31a8.png",
    ("da9846512ff19b8cd7278c8c973f75d36de8c4eb4e593b8285f6821aae1f4203.png", "CIMMYT_1"):
        "094dcc9098204e6f751504515f3b0a7b5f5ad500a8bd2ec10124ed3d4fdbb6ed.png",
}


def parse_boxes(text: str) -> np.ndarray:
    """``"x1 y1 x2 y2;x1 y1 x2 y2;..."`` -> tablica ``(N, 4)``; ``"no_box"`` -> ``(0, 4)``."""
    text = str(text).strip()
    if text == "no_box":
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([[float(v) for v in b.split()] for b in text.split(";") if b.strip()],
                    dtype=np.float32).reshape(-1, 4)


def load_split(zip_path: str | Path, split: str) -> pd.DataFrame:
    """Adnotacje podziału (train/validation/test) z pliku CSV w archiwum.

    Kolumny: ``image_name`` (plik w ``gwhd_2021/images/``, po poprawkach z ``NAME_FIXES``),
    ``domain``, ``boxes`` (surowe ramki xyxy), ``n_boxes_raw``, ``points`` (środki ramek
    przyciętych do obrazu) i ``count`` (liczba kłosów = liczba punktów).
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open(ZIP_ROOT + SPLIT_CSV[split]) as f:
        df = pd.read_csv(f)
    df["image_name"] = [NAME_FIXES.get((n, d), n) for n, d in zip(df["image_name"], df["domain"])]
    if df["image_name"].duplicated().any():
        raise ValueError(f"powtórzone nazwy obrazów w {split}: {df.loc[df.image_name.duplicated(), 'image_name'].tolist()}")
    df["boxes"] = df["BoxesString"].map(parse_boxes)
    df["n_boxes_raw"] = df["boxes"].map(len)
    df["points"] = [boxes_to_points(clip_boxes(b, IMAGE_SIZE, IMAGE_SIZE)) for b in df["boxes"]]
    df["count"] = df["points"].map(len)
    return df.drop(columns="BoxesString")


class GWHDDensityDataset(Dataset):
    """Próbki: obraz (znormalizowany tensor), mapa gęstości i liczba kłosów.

    Tryb treningowy (``crop_size`` podany): losowy wycinek ``crop_size x crop_size``,
    a potem przekształcenia z profilu ``augmentation`` (``AUGMENTATION_PROFILES``).
    Tryb ewaluacji (``crop_size=None``): cały obraz 1024x1024, bez augmentacji.

    Odbicie w poziomie jest wykonywane na obrazie i na gotowej mapie gęstości. Jest to
    równoważne liczeniu mapy z odbitych punktów (jądro Gaussa jest symetryczne), a nie
    wymaga przeliczania współrzędnych na granicach pikseli. Jasność i kontrast zmieniają
    tylko obraz.

    Mapa gęstości jest liczona z punktów leżących w wycinku, więc jej suma jest
    dokładnie równa liczbie kłosów w wycinku (``count``). Mapa ma rozdzielczość
    zmniejszoną ``downsample`` razy (CSRNet: 8) i NIE jest skalowana - ewentualne
    mnożenie przez stałą odbywa się w kodzie treningowym.

    Obrazy są czytane z archiwum ZIP przy każdym wywołaniu (bez rozpakowywania, bez
    trzymania 10 GB w RAM). Każdy proces roboczy DataLoadera otwiera własny uchwyt do ZIP.
    """

    def __init__(
        self,
        zip_path: str | Path,
        split: str,
        sigma: float,
        crop_size: int | None = None,
        downsample: int = 8,
        indices: Sequence[int] | None = None,
        augmentation: str = "crop_only",
    ) -> None:
        if crop_size is not None and crop_size % downsample:
            raise ValueError(f"crop_size={crop_size} musi być podzielny przez downsample={downsample}")
        if augmentation not in AUGMENTATION_PROFILES:
            raise ValueError(f"nieznany profil augmentacji {augmentation!r}; dostępne: {list(AUGMENTATION_PROFILES)}")
        if crop_size is None and augmentation != "crop_only":
            raise ValueError("augmentacja tylko w trybie treningowym (crop_size podany); ewaluacja: crop_only")
        df = load_split(zip_path, split)
        # numery obrazów w całym podziale (stabilne identyfikatory do zapisu predykcji)
        self.source_indices: list[int] = list(indices) if indices is not None else list(range(len(df)))
        df = df.iloc[self.source_indices].reset_index(drop=True)

        self.zip_path = str(zip_path)
        self.image_names: list[str] = df["image_name"].tolist()
        self.points: list[np.ndarray] = df["points"].tolist()
        self.domains: list[str] = df["domain"].tolist()
        self.sigma = sigma
        self.crop_size = crop_size
        self.downsample = downsample
        self.augmentation = augmentation
        self.aug = AUGMENTATION_PROFILES[augmentation]
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._zip: zipfile.ZipFile | None = None

    def __getstate__(self) -> dict:
        # uchwyt do ZIP nie może być kopiowany do procesów roboczych (Windows: pickle)
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def __len__(self) -> int:
        return len(self.image_names)

    def read_image(self, idx: int) -> np.ndarray:
        """Obraz ``idx`` jako tablica ``(H, W, 3)`` uint8."""
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        data = self._zip.read(ZIP_ROOT + "images/" + self.image_names[idx])
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    def __getitem__(self, idx: int) -> dict:
        img = self.read_image(idx)
        h, w = img.shape[:2]
        if (h, w) != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{self.image_names[idx]}: rozmiar {(h, w)}, oczekiwano {IMAGE_SIZE}x{IMAGE_SIZE}")
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

        # Losowania tylko dla włączonych przekształceń: crop_only zużywa liczby losowe tak
        # samo jak kod sprzed dodania profili, więc daje identyczne wycinki.
        if self.aug["hflip_p"] > 0 and np.random.rand() < self.aug["hflip_p"]:
            img = img[:, ::-1]
            density = density[:, ::-1]  # mapa (H/ds, W/ds): odbicie po szerokości

        image = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W), 0..1
        if self.aug["brightness"] > 0:
            b = self.aug["brightness"]
            image = (image * np.random.uniform(1 - b, 1 + b)).clamp(0, 1)
        if self.aug["contrast"] > 0:
            c = self.aug["contrast"]
            # kontrast względem średniej jasności obrazu (jak w torchvision: szarość ITU-R 601)
            gray_mean = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).mean()
            image = ((image - gray_mean) * np.random.uniform(1 - c, 1 + c) + gray_mean).clamp(0, 1)
        image = (image - self.mean) / self.std
        return {
            "image": image,
            "density": torch.from_numpy(density.copy()).unsqueeze(0),  # (1, H/ds, W/ds)
            "count": torch.tensor(len(points), dtype=torch.float32),
            "domain": self.domains[idx],
            "index": self.source_indices[idx],
        }
