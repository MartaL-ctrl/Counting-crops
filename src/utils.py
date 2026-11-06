"""Narzędzia ogólne: powtarzalność (seed), metryki liczenia, checkpointy."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Ustawia ziarna random, NumPy i PyTorch (CPU i wszystkie GPU).

    Przy ``deterministic=True`` włącza deterministyczne warianty operacji cuDNN/PyTorch.
    Trening jest wtedy wolniejszy, a operacje bez deterministycznej implementacji
    zgłaszają ostrzeżenie (``warn_only=True``), zamiast przerywać działanie.

    Uwaga: identyczne wyniki bit po bicie są gwarantowane tylko na tej samej
    wersji bibliotek, sterownika i modelu GPU. Między np. lokalną kartą a Colabem
    wyniki będą zbliżone, ale nie identyczne.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # obejmuje też wszystkie urządzenia CUDA

    if deterministic:
        # wymagane przez cuBLAS do deterministycznych mnożeń macierzy na GPU;
        # musi być ustawione, zanim CUDA zostanie zainicjalizowana
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Ustawia ziarna random/NumPy w procesie roboczym DataLoadera.

    Użycie: ``DataLoader(..., worker_init_fn=seed_worker, generator=g)``,
    gdzie ``g = torch.Generator().manual_seed(seed)``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state() -> dict[str, Any]:
    """Stany generatorów losowych (random, NumPy, PyTorch CPU/GPU) - do checkpointu."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Przywraca stany zapisane przez ``get_rng_state``.

    Stany PyTorch muszą być tensorami na CPU - ``load_checkpoint(map_location="cuda")``
    przenosi je na GPU, więc tu wracają na CPU.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def count_metrics(pred: Sequence[float], true: Sequence[float]) -> dict[str, float]:
    """MAE i RMSE liczby obiektów na obraz."""
    err = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean()))}


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Zapisuje checkpoint atomowo: najpierw do pliku tymczasowego, potem podmiana.

    Przerwanie zapisu (np. rozłączenie Colaba / Dysku Google) nie niszczy
    poprzedniego, kompletnego checkpointu.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Wczytuje checkpoint zapisany przez ``save_checkpoint``.

    ``weights_only=False``, bo checkpoint zawiera też konfigurację i stany RNG -
    wczytuj tylko własne, zaufane pliki.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
