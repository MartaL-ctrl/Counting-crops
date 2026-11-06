# Dane: GWHD 2021 – opis wykorzystanej kopii

Dokument opisuje **dokładnie tę kopię zbioru, która jest używana w projekcie**. Wszystkie liczby
w sekcjach 2–6 policzono skryptem z lokalnych plików (stan na 2026-11-06), a nie przepisano
z publikacji. Liczby z publikacji podano osobno, do porównania.

---

## 1. Nazwa, źródło i wersja

| | |
|---|---|
| Zbiór | Global Wheat Head Dataset 2021 (GWHD 2021) |
| Wersja | 1.0 (wg karty zbioru: „Published: July 12, 2021, Version 1.0”) |
| Wykorzystana kopia | Hugging Face, organizacja **GlobalWheat**: https://huggingface.co/datasets/GlobalWheat/GWHD_2021 |
| Format kopii | 4 pliki Apache Parquet (`data/train-00000-of-00002`, `train-00001-of-00002`, `validation-00000-of-00001`, `test-00000-of-00001`), razem 1,26 GB |
| Data pobrania | 2026-11-02; sumy SHA256 wszystkich plików zgodne z deklarowanymi przez Hugging Face |
| Wydanie oryginalne | Zenodo, rekord 5092309 (DOI 10.5281/zenodo.5092309), plik `gwhd_2021.zip` (10,2 GB) – **nieużywane** |
| Licencja | CC BY 4.0 |
| Lokalnie | `data/raw/gwhd_hf/` (poza repozytorium Git) |

Kopia z Hugging Face **nie jest identyczna z wydaniem na Zenodo** – różni się formatem obrazów
(sekcja 3) i formatem adnotacji (sekcja 4). W pracy należy podać, że używana jest kopia z Hugging Face.

---

## 2. Liczby obrazów i adnotacji (policzone)

| Podział | Obrazy | Ramki (kłosy) | Obrazy bez kłosów | Domeny | Kraje |
|---|---:|---:|---:|---:|---|
| train | 3 655 | 163 651 | 50 | 18 | Francja, Szwajcaria, Wielka Brytania, Norwegia, Belgia |
| validation | 1 476 | 44 347 | 28 | 11 | Japonia, Kanada, Australia, Chiny |
| test | 1 381 | 67 373 | 47 | 18 | USA, Australia, Meksyk, Chiny, Japonia, Sudan |
| **razem** | **6 512** | **275 371** | **125** | **47** | **12** |

- **Domena** (sub-dataset) = jedna sesja pomiarowa: ta sama instytucja, miejsce, data, platforma i czujnik.
- Żadna domena nie występuje w więcej niż jednym podziale.
- Wszystkie ramki należą do jednej kategorii (`categories = 0`, kłos pszenicy).
- Rozkład per domena: tabela w dodatku A.

### Porównanie z publikacjami

| Źródło | Obrazy | Kłosy | Uwagi |
|---|---:|---:|---|
| **Ta kopia** | **6 512** | **275 371** | policzone |
| Preprint arXiv 2105.07660 v2, tabela 1 | 6 515 | 275 187 | „Total 6515 275187” |
| Preprint, tekst | ok. 6 500 | ok. 275 000 | zaokrąglone |
| Zenodo 5092309 | > 6 000 | 300k+ | zaokrąglone |

| Podział | Ta kopia | Preprint (GWC 2021) |
|---|---:|---:|
| train | 3 655 / 18 domen | 3 657 / 18 sub-datasetów |
| validation | 1 476 / 11 domen | 1 476 / 11 sub-datasetów |
| test | 1 381 / 18 domen | 1 373 / 18 sub-datasetów |

Przyczyny różnic nie da się ustalić z dostępnych źródeł. Preprint powstał przed końcem konkursu
i ostatecznym wydaniem; jego tabela per domena też różni się od tej kopii (np. UQ_10: 53 obrazy
w preprincie, 106 tutaj, przy tej samej liczbie kłosów 8 629; ARC_1: 888 kłosów w preprincie, 1 169 tutaj).
W UQ_10 i UQ_11 sprawdzono, że nie ma zduplikowanych obrazów.
**W pracy podajemy liczby z tej kopii** i zaznaczamy, że różnią się nieznacznie od preprintu.

---

## 3. Obrazy

| | |
|---|---|
| Rozdzielczość | 1024 × 1024 px, wszystkie 6 512 obrazów |
| Format w tej kopii | **JPEG**, RGB, średnio ok. 190 kB na obraz |
| Kompresja | tablica kwantyzacji luminancji `[8, 6, 5, 8, 12, 20, 26, 31, …]` = standardowa tablica IJG dla jakości **75**, identyczna we wszystkich obrazach |
| Format w oryginale (Zenodo) | PNG (bezstratny) – wg opisu rekordu: „All images have a .png extension” |
| Nazwy plików | 64-znakowy skrót szesnastkowy + `.jpg`, wszystkie unikalne (pole `image.path`) |
| Skala | wg preprintu rozdzielczość przestrzenna (GSD) 0,2–0,4 mm/px; obrazy to wycinki 1024 × 1024 z większych zdjęć |

**Wniosek:** przy konwersji do formatu parquet obrazy zostały **stratnie przekompresowane**
(PNG → JPEG, jakość ok. 75). Kompresja może zacierać drobne szczegóły, istotne zwłaszcza dla małych
kłosów. Trzeba to opisać w pracy (zob. sekcja 7).

---

## 4. Adnotacje

| | Ta kopia (Hugging Face, parquet) | Oryginał (Zenodo, CSV) |
|---|---|---|
| Zapis ramek | kolumna `objects.boxes`: lista list liczb całkowitych | kolumna `BoxesString`: tekst, liczby rozdzielone spacją, ramki średnikiem |
| Format współrzędnych | **`[x, y, w, h]`**: lewy górny róg + szerokość i wysokość, w pikselach | `[x_min, y_min, x_max, y_max]` |
| Obraz bez kłosów | pusta lista `boxes: []` | tekst `no_box` |
| Identyfikator obrazu | `image.path` (nazwa pliku) | `image_name` |

**Uwaga:** karta zbioru na Hugging Face podaje format `[x_min, y_min, x_max, y_max]` – to opis
skopiowany z oryginału i **nie zgadza się z danymi w plikach parquet**. Format `[x, y, w, h]`
zweryfikowano na 44 347 ramkach walidacyjnych:

- warunek `x + w ≤ 1024` i `y + h ≤ 1024` spełnia 100% ramek,
- warunek `x_max > x_min` i `y_max > y_min` (interpretacja xyxy) spełnia tylko 2,7% ramek.

Nie ma ramek o zerowej szerokości ani wysokości. Mediana rozmiaru ramki: train 73 × 67 px,
validation 88 × 87 px, test 48 × 47 px.

**Obsługa obrazów bez kłosów:** pusta lista ramek daje 0 punktów i mapę gęstości złożoną z samych zer
(liczba = 0). Takie obrazy pozostają w zbiorze (`src/dataset.py`, `src/density.py`).

---

## 5. Podział danych – wybór

Autorzy GWHD 2021 i społeczność używają dwóch podziałów:

### a) Global Wheat Challenge 2021 (GWC 2021)

Podział z konkursu na AIcrowd, przeznaczony do porównywania metod lokalizacji i liczenia kłosów
w zastosowaniach fenotypowych. Według preprintu: train z Europy (3 657 obrazów, 18 sub-datasetów),
walidacja publiczna (1 476 obrazów, 11 sub-datasetów) i test prywatny (1 373 obrazy, 18 sub-datasetów)
z pozostałych regionów. Na Zenodo odpowiadają mu pliki `competition_train.csv`, `competition_val.csv`
i `competition_test.csv`. Usask_1 (Kanada) przeniesiono między edycjami 2020 i 2021 z treningu
do walidacji i testu.

### b) GlobalWheat-WILDS

Podział z benchmarku WILDS (Koh i in., 2021), przeznaczony do badania **zmiany domeny**. Treningowe
sesje pochodzą z Europy, a walidacja i test OOD (spoza rozkładu treningowego) z rozłącznych
kontynentów. Dodatkowo są zbiory ID (*in-distribution*) z tych samych sesji co trening. Według
streszczenia artykułu WILDS (dodatek E.5, **liczby niezweryfikowane dosłownie**): train 2 943,
val OOD 1 424, test OOD 1 434, val ID 357, test ID 357. Do użycia wymaga pakietu `wilds`
i oryginalnych obrazów.

### Wybór: **GWC 2021**

Uzasadnienie:
1. **Zgodność z posiadanymi danymi.** Kopia z Hugging Face ma dokładnie strukturę GWC 2021:
   18 / 11 / 18 domen, a walidacja ma tyle samo obrazów (1 476) co w preprincie. Podział WILDS
   wymagałby innych plików i ponownego przypisania obrazów.
2. **Cel pracy.** Praca porównuje metody liczenia (mapa gęstości vs detekcja) – to zadanie
   fenotypowe, do którego autorzy przeznaczyli GWC 2021. Badanie zmiany domeny nie jest głównym celem.
3. **Brak przecieku danych.** Domeny są rozłączne między podziałami (sprawdzone), więc obrazy
   z jednej sesji nie trafiają jednocześnie do treningu i testu.
4. **Porównywalność.** Podział jest publiczny i stały, więc inni mogą powtórzyć eksperyment.

Konsekwencje do opisania w pracy:
- Walidacja i test pochodzą z innych kontynentów niż trening. Wyniki mierzą więc także
  **uogólnianie na nowe lokalizacje**, a błędy będą zapewne większe niż przy losowym podziale.
- Do wyboru najlepszej epoki i hiperparametrów używany jest wyłącznie zbiór validation.
  Test jest uruchamiany dopiero na końcu.
- Metryką konkursu była WDA (*weighted domain accuracy*) dla detekcji. W tej pracy podstawowe
  metryki liczenia to **MAE i RMSE** liczby kłosów na obraz, więc wyniki nie są wprost porównywalne
  z rankingiem konkursu.

Planowane rozszerzenie: dodatkowa analiza wyników w grupach według liczby kłosów na obraz.
Progi grup zostaną ustalone na zbiorze train, bez użycia testu.

---

## 6. Metadane i filtrowanie

| Kolumna | Użycie w projekcie |
|---|---|
| `image` (`bytes`, `path`) | obraz; `path` jako identyfikator |
| `objects.boxes` | ramki → środki → mapy gęstości (`src/density.py`) |
| `objects.categories` | nieużywane (jedna klasa) |
| `domain` | zapis w predykcjach per obraz; analiza błędów per domena |
| `country`, `location`, `development_stage` | tylko opis zbioru i ewentualna analiza wyników; nie są wejściem modelu |

**Filtrowanie: brak.** Używane są wszystkie 6 512 obrazów, łącznie z 125 obrazami bez kłosów.

Wykryte anomalie (pozostawione bez zmian):
- **Duplikat w teście:** dwa obrazy z domeny Terraref_2 (test, indeksy 98 i 1142) mają bajtowo
  identyczną zawartość, ale różne nazwy plików i różne adnotacje: 0 i 1 ramka. Oba obrazy są w tym
  samym podziale, więc nie powodują przecieku. Wpływ na MAE jest pomijalny (1 obraz z 1 381).
- `ETHZ_1` (Szwajcaria) ma w kolumnie `location` wartość „Usask”. To prawdopodobnie błąd metadanych,
  bo Usask to Saskatchewan w Kanadzie.
- `Utokyo_3` ma `development_stage = "multiple"`, a `Ukyoto_1` – „Post-Flowering” z wielką literą
  (gdzie indziej „Post-flowering”).

---

## 7. Otwarte kwestie

1. **JPEG zamiast PNG.** Do decyzji: (a) zostać przy tej kopii i opisać kompresję jako ograniczenie
   albo (b) do końcowych eksperymentów użyć oryginalnych PNG z Zenodo (10,2 GB). Nazwy plików
   wyglądają jak w oryginale (skrót szesnastkowy), więc zapewne da się je dopasować do
   `competition_*.csv` po nazwie bez rozszerzenia – **niesprawdzone**.
2. **Różnice 3 obrazów i 184 kłosów względem preprintu** – źródło nieznane. Można je wyjaśnić,
   porównując z CSV na Zenodo.
3. **Podział WILDS** – liczby w sekcji 5b pochodzą ze streszczenia i wymagają weryfikacji w artykule
   WILDS. Przy cytowaniu sprawdzić też, czy opis obu podziałów jest w wersji czasopiśmienniczej
   artykułu o GWHD 2021 (Plant Phenomics, 2021). Preprint arXiv v2 nie wspomina o WILDS.

---

## 8. Źródła

- E. David i in., *Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat
  head localization methods*, arXiv:2105.07660 (v2, 3.06.2021) – https://arxiv.org/abs/2105.07660
- Wersja czasopiśmiennicza: E. David i in., *Global Wheat Head Detection 2021: An Improved Dataset
  for Benchmarking Wheat Head Detection Methods*, Plant Phenomics, 2021 – **dane bibliograficzne do sprawdzenia**
- Zenodo, *Global Wheat Head Dataset 2021*, wersja 1.0, DOI 10.5281/zenodo.5092309 – https://zenodo.org/records/5092309
- Hugging Face, GlobalWheat/GWHD_2021 – https://huggingface.co/datasets/GlobalWheat/GWHD_2021
- P. W. Koh i in., *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML 2021 – https://arxiv.org/abs/2012.07421

---

## Dodatek A. Domeny w poszczególnych podziałach (policzone)

### train (18 domen, 3 655 obrazów, 163 651 ramek)

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Ramki |
|---|---|---|---|---:|---:|
| Arvalis_1 | Francja | Gréoux | Post-flowering | 66 | 2 935 |
| Arvalis_2 | Francja | Gréoux | Filling | 401 | 21 003 |
| Arvalis_3 | Francja | Gréoux | Filling - Ripening | 588 | 21 893 |
| Arvalis_4 | Francja | Gréoux | Filling | 204 | 4 270 |
| Arvalis_5 | Francja | VLB | Filling | 448 | 8 180 |
| Arvalis_6 | Francja | VSC | Filling - Ripening | 160 | 8 698 |
| Arvalis_7 | Francja | VLB | Filling - Ripening | 24 | 1 247 |
| Arvalis_8 | Francja | VLB | Filling - Ripening | 20 | 1 062 |
| Arvalis_9 | Francja | VLB | Ripening | 32 | 1 894 |
| Arvalis_10 | Francja | Mons | Filling | 58 | 1 524 |
| Arvalis_11 | Francja | VLB | Filling | 60 | 2 818 |
| Arvalis_12 | Francja | Gréoux | Filling | 29 | 1 277 |
| ETHZ_1 | Szwajcaria | „Usask” (sic) | Filling | 747 | 49 603 |
| Inrae_1 | Francja | Toulouse | Filling - Ripening | 176 | 3 634 |
| NMBU_1 | Norwegia | NMBU | Filling | 82 | 7 345 |
| NMBU_2 | Norwegia | NMBU | Ripening | 98 | 5 211 |
| Rres_1 | Wielka Brytania | Rothamsted | Filling - Ripening | 432 | 19 210 |
| ULiège-GxABT_1 | Belgia | Gembloux | Ripening | 30 | 1 847 |

### validation (11 domen, 1 476 obrazów, 44 347 ramek)

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Ramki |
|---|---|---|---|---:|---:|
| NAU_1 | Chiny | Baima | Post-flowering | 20 | 1 240 |
| UQ_1 | Australia | Gatton | Post-flowering | 22 | 640 |
| UQ_2 | Australia | Gatton | Post-flowering | 16 | 39 |
| UQ_3 | Australia | Gatton | Filling | 14 | 297 |
| UQ_4 | Australia | Gatton | Filling | 30 | 1 039 |
| UQ_5 | Australia | Gatton | Filling - Ripening | 30 | 3 680 |
| UQ_6 | Australia | Gatton | Filling - Ripening | 30 | 1 147 |
| Usask_1 | Kanada | Saskatchewan | Filling - Ripening | 200 | 5 985 |
| Utokyo_1 | Japonia | NARO-Tsukuba | Ripening | 538 | 14 185 |
| Utokyo_2 | Japonia | NARO-Tsukuba | Ripening | 456 | 13 010 |
| Utokyo_3 | Japonia | NARO-Hokkaido | multiple | 120 | 3 085 |

### test (18 domen, 1 381 obrazów, 67 373 ramki)

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Ramki |
|---|---|---|---|---:|---:|
| ARC_1 | Sudan | Wad Medani | Filling | 30 | 1 169 |
| CIMMYT_1 | Meksyk | Ciudad Obregon | Post-flowering | 68 | 2 785 |
| CIMMYT_2 | Meksyk | Ciudad Obregon | Post-flowering | 77 | 2 771 |
| CIMMYT_3 | Meksyk | Ciudad Obregon | Post-flowering | 60 | 1 561 |
| KSU_1 | USA | KSU | Post-flowering | 100 | 6 435 |
| KSU_2 | USA | KSU | Post-flowering | 100 | 5 302 |
| KSU_3 | USA | KSU | Filling | 95 | 5 217 |
| KSU_4 | USA | KSU | Ripening | 60 | 3 285 |
| NAU_2 | Chiny | Baima | Post-flowering | 100 | 4 918 |
| NAU_3 | Chiny | Baima | Filling | 100 | 4 596 |
| Terraref_1 | USA | Maricopa, AZ | Ripening | 144 | 3 360 |
| Terraref_2 | USA | Maricopa, AZ | Filling | 106 | 1 274 |
| UQ_7 | Australia | Gatton | Ripening | 17 | 1 335 |
| UQ_8 | Australia | McAllister | Ripening | 41 | 4 835 |
| UQ_9 | Australia | Brookstead | Filling - Ripening | 33 | 2 886 |
| UQ_10 | Australia | Gatton | Filling - Ripening | 106 | 8 629 |
| UQ_11 | Australia | Gatton | Post-flowering | 84 | 4 345 |
| Ukyoto_1 | Japonia | Kyoto | Post-Flowering | 60 | 2 670 |
