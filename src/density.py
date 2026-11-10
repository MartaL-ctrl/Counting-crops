"""Punkty obiektów i mapy gęstości (jądro Gaussa o stałej sigmie albo dopasowane do ramki).

Konwencje:
- ramki w formacie oryginalnego GWHD 2021 (Zenodo): ``[x_min, y_min, x_max, y_max]`` w pikselach,
- punkty: tablica ``(N, 2)`` z kolumnami ``[x, y]`` (x = kolumna, y = wiersz),
- mapy: tablice ``(H, W)`` typu float32; każdy punkt wnosi do sumy mapy dokładnie 1.
"""

from __future__ import annotations

import numpy as np


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Przycina ramki ``(N, 4)`` xyxy do obrazu i usuwa te, z których nic nie zostało.

    W GWHD 2021 część ramek (domena CIMMYT_3) wychodzi poza obraz 1024x1024;
    ramka leżąca całkowicie poza obrazem ma po przycięciu zerowe pole i jest odrzucana.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep]


def boxes_to_points(boxes: np.ndarray) -> np.ndarray:
    """Zamienia ramki ``(N, 4)`` xyxy na środki ``(N, 2)`` ``[cx, cy]``.

    Założenie: środek ramki reprezentuje położenie kłosa. Dla ramek przyciętych przez
    ``clip_boxes`` środek leży zawsze wewnątrz obrazu.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Kwadratowe jądro Gaussa o promieniu ``ceil(truncate * sigma)``, znormalizowane do sumy 1."""
    if sigma <= 0:
        raise ValueError(f"sigma musi być dodatnia, podano {sigma}")
    radius = int(np.ceil(truncate * sigma))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return (kernel / kernel.sum()).astype(np.float32)


def anisotropic_gaussian_kernel(sigma_x: float, sigma_y: float, truncate: float = 3.0) -> np.ndarray:
    """Jądro Gaussa o osobnych sigmach w poziomie i pionie (osie jądra = osie obrazu).

    Wymiary ``(2*ry + 1, 2*rx + 1)``, gdzie ``r = ceil(truncate * sigma)``; suma = 1.
    """
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError(f"sigmy muszą być dodatnie, podano {sigma_x}, {sigma_y}")
    rx, ry = int(np.ceil(truncate * sigma_x)), int(np.ceil(truncate * sigma_y))
    gx = np.exp(-(np.arange(-rx, rx + 1, dtype=np.float64) ** 2) / (2 * sigma_x**2))
    gy = np.exp(-(np.arange(-ry, ry + 1, dtype=np.float64) ** 2) / (2 * sigma_y**2))
    kernel = np.outer(gy, gx)
    return (kernel / kernel.sum()).astype(np.float32)


def box_sigmas(sizes: np.ndarray, factor: float, min_sigma: float = 2.0) -> np.ndarray:
    """Sigmy ``(N, 2)`` ``[sigma_x, sigma_y]`` z wymiarów ramek ``(N, 2)`` ``[szerokość, wysokość]``.

    ``sigma = factor * bok`` (dolne ograniczenie ``min_sigma``). Dla ``factor = 0.25`` ramka
    sięga ±2 sigma od środka, czyli zawiera ok. 91% masy jądra (0,954 w każdej osi).
    Ramki są równoległe do osi obrazu, więc kłos leżący poziomo (szeroka ramka) dostaje
    plamkę wydłużoną w poziomie; ukośnego ułożenia ramka nie opisuje.
    """
    sizes = np.asarray(sizes, dtype=np.float32).reshape(-1, 2)
    return np.maximum(sizes * factor, min_sigma)


def density_map(points: np.ndarray, height: int, width: int, sigma: float,
                sigmas: np.ndarray | None = None) -> np.ndarray:
    """Tworzy mapę gęstości ``(height, width)``, w której suma = liczba punktów.

    - ``sigmas`` (``(N, 2)``, z ``box_sigmas``): osobne jądro dla każdego punktu, ``sigma``
      jest wtedy ignorowana. Bez ``sigmas``: jedno jądro o stałej ``sigma`` dla wszystkich.
    - Punkty poza obrazem są pomijane (nie wnoszą nic do mapy).
    - Jądro przycięte krawędzią obrazu jest ponownie normalizowane do sumy 1,
      więc kłos przy krawędzi liczy się jako cały (decyzja do opisania w pracy).
    """
    density = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return density
    if sigmas is not None and len(sigmas) != len(points):
        raise ValueError(f"{len(sigmas)} par sigm dla {len(points)} punktów")

    fixed = gaussian_kernel(sigma) if sigmas is None else None
    for i, (x, y) in enumerate(points):
        if not (0 <= x < width and 0 <= y < height):
            continue
        kernel = fixed if sigmas is None else anisotropic_gaussian_kernel(*sigmas[i])
        ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
        col, row = int(x), int(y)
        top, bottom = max(row - ry, 0), min(row + ry + 1, height)
        left, right = max(col - rx, 0), min(col + rx + 1, width)
        piece = kernel[top - row + ry:bottom - row + ry, left - col + rx:right - col + rx]
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
