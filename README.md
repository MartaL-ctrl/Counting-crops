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
│   ├── dataset.py    # Dataset PyTorch: CSV i obrazy z archiwum ZIP GWHD 2021
│   └── model.py      # CSRNet
├── configs/          # parametry eksperymentów (YAML)
├── train.py          # trening z checkpointami i wznawianiem
├── evaluate.py       # MAE/RMSE łącznie, per domena i per grupa liczby kłosów
├── make_figures.py   # rysunki do pracy (PDF + PNG) w outputs/figures/
├── notebooks/        # nauka, eksploracja, testy
├── data/             # dane lokalne - NIE trafiają do Gita
│   └── raw/zenodo/gwhd_2021.zip
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

Danych nie ma w repozytorium. Źródło: oryginalne wydanie GWHD 2021 z Zenodo,
https://zenodo.org/records/5092309 (DOI 10.5281/zenodo.5092309, wersja 1.0, CC BY 4.0).
Plik `gwhd_2021.zip` (10,2 GB, MD5 `22b4b542c9ae7e056d7fcdeae9ecaed5`) umieść **bez rozpakowywania** w:

```
data/raw/zenodo/gwhd_2021.zip
```

Kod czyta obrazy i pliki CSV bezpośrednio z archiwum. Inną lokalizację ustawia się polem `data_zip`
w pliku konfiguracyjnym. Opis danych, wykryte błędy wydania i sposób ich obsługi: `docs/dane_gwhd.md`.

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

Augmentację zbioru uczącego wybiera pole `augmentation`: `crop_only` (sam losowy wycinek, domyślnie)
albo `thesis_aug` (wycinek + odbicie poziome p=0,5 + jasność i kontrast ±15%). Walidacja i test są
zawsze bez augmentacji.

## Ewaluacja

```powershell
python evaluate.py --checkpoint outputs/<run>/best.pt                        # cały zbiór walidacyjny
python evaluate.py --checkpoint outputs/<run>/best.pt --split test --allow-test  # tylko końcowa ocena
```

Wyniki w `outputs/<run>/eval_<split>/`: predykcje per obraz, MAE/RMSE (łącznie, per domena,
per grupa liczby kłosów) i progi grup (kwantyle liczby kłosów w zbiorze uczącym).
Zbiór testowy wymaga `--allow-test` - nie wybieramy na nim modelu ani hiperparametrów.

## Trening w Google Colab

Otwórz `notebooks/colab_train.ipynb` w Colabie (link „Open in Colab” na górze notebooka),
ustaw GPU i uruchom wszystko. Notebook pobiera kod z GitHuba, archiwum z Zenodo na dysk Colaba
(w każdej sesji; opcjonalnie trzyma kopię na Dysku Google, ale zajmuje ona 10,2 GB),
a wyniki zapisuje do `MyDrive/counting-crops/outputs/`.
Po rozłączeniu sesji wystarczy uruchomić wszystko ponownie - trening wznowi się od `last.pt`.
Konfiguracje dla Colaba: `configs/colab_*.yaml`.
