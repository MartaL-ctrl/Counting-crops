# Counting-crops

Praca inżynierska: estymacja liczby roślin uprawnych na podstawie map gęstości,
z porównaniem do metody detekcji. Kandydat na zbiór danych: GWHD 2021
(Global Wheat Head Detection).

## Struktura

```
Counting-crops/
├── src/
│   ├── utils.py      # seed, metryki MAE/RMSE, checkpointy
│   ├── density.py    # ramki -> punkty -> mapy gęstości
│   ├── dataset.py    # Dataset PyTorch dla plików parquet GWHD
│   └── model.py      # CSRNet
├── configs/          # parametry eksperymentów (YAML)
├── train.py          # trening z checkpointami i wznawianiem
├── notebooks/        # nauka, eksploracja, testy
├── data/             # dane lokalne - NIE trafiają do Gita
│   └── raw/gwhd_hf/  # pliki *.parquet
├── outputs/          # wyniki treningów - NIE trafiają do Gita
├── requirements.txt
└── README.md
```

## Środowisko (Windows, PowerShell)

Jednorazowo - utworzenie środowiska z Pythona 3.12:

```powershell
py -3.12 -m venv .venv
```

Za każdym razem przed pracą - aktywacja:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jeśli PowerShell zgłasza błąd o „running scripts is disabled”, jednorazowo:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Instalacja bibliotek (w aktywnym środowisku):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch instaluje się osobno, bo wariant zależy od karty GPU.
Lokalnie (GTX 1050 Ti, architektura Pascal, sterownik 552.22) używamy wariantu
CUDA 12.6 - nowsze warianty (CUDA 13.x) nie obsługują już kart Pascal:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

Sprawdzenie (powinno wypisać `True` i listę zawierającą `sm_61`):

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

W Google Colab PyTorch jest już zainstalowany - nie instaluj go ponownie.

## Dane

Danych nie ma w repozytorium. Źródło: GWHD 2021 w formacie parquet z Hugging Face,
https://huggingface.co/datasets/GlobalWheat/GWHD_2021 (wersja 1.0, ta sama co na Zenodo 5092309).
Cztery pliki z folderu `data/` (razem ok. 1,26 GB) umieść w:

```
data/raw/gwhd_hf/
```

Ważne: ramki w tych plikach mają format `[x, y, szerokość, wysokość]`, wbrew opisowi
na Hugging Face (`[x_min, y_min, x_max, y_max]`). Sprawdzone na całym zbiorze walidacyjnym.

Inną lokalizację danych ustawia się polem `data_dir` w pliku konfiguracyjnym.

## Uruchomienie notebooka

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks/nauka_01_mapa_gestosci.ipynb
```

Albo otwórz plik `.ipynb` w VS Code i wybierz kernel `.venv (Python 3.12)`.

## Trening

```powershell
.\.venv\Scripts\Activate.ps1
python train.py --config configs/smoke.yaml                                 # nowy trening
python train.py --config configs/smoke.yaml --resume outputs/smoke/last.pt  # wznowienie
```

Parametry eksperymentu są w plikach `configs/*.yaml`. Wyniki trafiają do `outputs/<run_name>/`
(folder ignorowany przez Git): `history.csv`, `predictions_val_best.csv`, `last.pt` (co epokę),
`best.pt` (najlepsze val MAE), `config.yaml`, `subsets.json`.

`configs/smoke.yaml` to tylko test działania kodu (32 obrazy) - jego wyników nie raportujemy.
