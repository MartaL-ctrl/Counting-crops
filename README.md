# Counting-crops

Praca inżynierska: estymacja liczby roślin uprawnych na podstawie map gęstości,
z porównaniem do metody detekcji. Kandydat na zbiór danych: GWHD 2021
(Global Wheat Head Detection).

## Struktura

```
Counting-crops/
├── notebooks/
│   └── 01_inspekcja_danych.ipynb   # krok 1: obejrzenie obrazu i adnotacji
├── data/                           # dane lokalne - NIE trafiają do Gita
│   └── raw/
│       └── gwhd_2021.zip
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

PyTorch instaluje się osobno - komendą wygenerowaną na
https://pytorch.org/get-started/locally/ dla swojego systemu i karty GPU.

## Dane

Danych nie ma w repozytorium. Pobrany ZIP z GWHD 2021 umieść w:

```
data/raw/gwhd_2021.zip
```

Albo wskaż inną lokalizację zmienną środowiskową `GWHD_ZIP`, np.:

```powershell
$env:GWHD_ZIP = "E:\dane\gwhd_2021.zip"
```

W Google Colab (po zamontowaniu Dysku Google):

```python
import os
os.environ["GWHD_ZIP"] = "/content/drive/MyDrive/gwhd_2021.zip"
```

## Uruchomienie notebooka

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks/01_inspekcja_danych.ipynb
```

Albo otwórz plik `.ipynb` w VS Code i wybierz kernel `.venv (Python 3.12)`.
