"""Punkty obiektów i mapy gęstości (jądro Gaussa o stałej sigmie).

Konwencje:
- ramki w formacie GWHD z Hugging Face: ``[x, y, w, h]`` w pikselach (lewy górny róg + rozmiar),
- punkty: tablica ``(N, 2)`` z kolumnami ``[x, y]`` (x = kolumna, y = wiersz),
- mapy: tablice ``(H, W)`` typu float32; każdy punkt wnosi do sumy mapy dokładnie 1.
"""

from __future__ import annotations

import numpy as np


def boxes_to_points(boxes: np.ndarray) -> np.ndarray:
    """Zamienia ramki ``(N, 4)`` ``[x, y, w, h]`` na środki ``(N, 2)`` ``[cx, cy]``.

    Założenie: środek ramki reprezentuje położenie kłosa.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)


def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Kwadratowe jądro Gaussa o promieniu ``ceil(truncate * sigma)``, znormalizowane do sumy 1."""
    if sigma <= 0:
        raise ValueError(f"sigma musi być dodatnia, podano {sigma}")
    radius = int(np.ceil(truncate * sigma))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return (kernel / kernel.sum()).astype(np.float32)


def density_map(points: np.ndarray, height: int, width: int, sigma: float) -> np.ndarray:
    """Tworzy mapę gęstości ``(height, width)``, w której suma = liczba punktów.

    - Punkty poza obrazem są pomijane (nie wnoszą nic do mapy).
    - Jądro przycięte krawędzią obrazu jest ponownie normalizowane do sumy 1,
      więc kłos przy krawędzi liczy się jako cały (decyzja do opisania w pracy).
    """
    density = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return density

    kernel = gaussian_kernel(sigma)
    r = kernel.shape[0] // 2
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            continue
        col, row = int(x), int(y)
        top, bottom = max(row - r, 0), min(row + r + 1, height)
        left, right = max(col - r, 0), min(col + r + 1, width)
        piece = kernel[top - row + r:bottom - row + r, left - col + r:right - col + r]
        density[top:bottom, left:right] += piece / piece.sum()
    return density


def downsample_density(density: np.ndarray, factor: int) -> np.ndarray:
    """Zmniejsza mapę ``factor`` razy przez sumowanie bloków ``factor x factor``.

    Sumowanie (a nie uśrednianie czy interpolacja) zachowuje dokładnie sumę mapy,
    czyli liczbę obiektów. Wymiary mapy muszą być podzielne przez ``factor``.
    """
    h, w = density.shape
    if h % factor or w % factor:
        raise ValueError(f"wymiary mapy {density.shape} niepodzielne przez {factor}")
    return density.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
