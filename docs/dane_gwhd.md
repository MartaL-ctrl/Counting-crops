# Dane: GWHD 2021 – opis wykorzystanej wersji

Dokument opisuje **dokładnie te pliki, których używa projekt**. Wszystkie liczby w sekcjach 2–7
policzono skryptem z lokalnej kopii archiwum (stan na 2026-11-08), korzystając z tego samego kodu
wczytującego co trening (`src/dataset.py: load_split`). Liczby z publikacji podano osobno, do porównania.

---

## 1. Nazwa, źródło i wersja

| | |
|---|---|
| Zbiór | Global Wheat Head Dataset 2021 (GWHD 2021) |
| Wersja | 1.0, opublikowana 12.07.2021 |
| Źródło | **Zenodo, rekord 5092309**: https://zenodo.org/records/5092309, DOI 10.5281/zenodo.5092309 |
| Plik | `gwhd_2021.zip`, 10 215 190 132 B (10,2 GB) |
| Suma kontrolna | MD5 `22b4b542c9ae7e056d7fcdeae9ecaed5`, zgodna z wartością podaną na Zenodo |
| Data pobrania | 2026-11-08 |
| Licencja | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Autor rekordu | Etienne David (INRAE) i in. |
| Lokalnie | `data/raw/zenodo/gwhd_2021.zip` (poza repozytorium Git); archiwum **nie jest rozpakowywane**, bo kod czyta obrazy bezpośrednio z ZIP-a |

Zawartość archiwum (6521 wpisów):

| Plik | Opis |
|---|---|
| `gwhd_2021/images/*.png` | 6515 obrazów |
| `gwhd_2021/competition_train.csv` | podział treningowy GWC 2021 (3657 wierszy) |
| `gwhd_2021/competition_val.csv` | podział walidacyjny, „public test” (1476 wierszy) |
| `gwhd_2021/competition_test.csv` | podział testowy, „private test” (1382 wiersze) |
| `gwhd_2021/metadata_dataset.csv` | metadane 47 domen: kraj, lokalizacja, faza rozwoju (separator `;`) |

Opis na Zenodo wymienia plik `Metadata.csv`, ale w archiwum nazywa się on `metadata_dataset.csv`.

**Wcześniej używana kopia** (Hugging Face `GlobalWheat/GWHD_2021`, parquet, 1,26 GB) została zastąpiona
oryginałem. Porównanie obu wersji jest w sekcji 8.

---

## 2. Liczby obrazów i adnotacji

| Podział | Obrazy | Ramki w CSV | Kłosy (po przycięciu) | Średnio / obraz | Mediana | Maks. | Obrazy bez kłosów | Domeny | Kraje |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 3 657 | 163 690 | 163 690 | 44,76 | 45 | 129 | 50 | 18 | Francja, Szwajcaria, Wielka Brytania, Norwegia, Belgia |
| validation | 1 476 | 44 347 | 44 347 | 30,05 | 28 | 179 | 28 | 11 | Japonia, Kanada, Australia, Chiny |
| test | 1 382 | 67 431 | 67 424 | 48,79 | 46 | 190 | 47 | 18 | USA, Australia, Meksyk, Chiny, Japonia, Sudan |
| **razem** | **6 515** | **275 468** | **275 461** | 42,28 | 39 | 190 | **125** | **47** | **12** |

- „Kłosy” to liczba ramek po przycięciu do obrazu (sekcja 4). Ta liczba jest wartością referencyjną
  w treningu i ocenie. Różnica 7 ramek dotyczy ramek leżących całkowicie poza obrazem.
- **Domena** to jedna sesja pomiarowa (sub-dataset): ta sama instytucja, miejsce, data, platforma i czujnik.
- Żadna domena ani żaden obraz nie występuje w więcej niż jednym podziale (sprawdzone).
- Wszystkie ramki należą do jednej klasy (kłos pszenicy).
- Rozkład per domena: dodatek A.

### Porównanie z publikacją

| | Ta wersja | Preprint arXiv 2105.07660 v2 |
|---|---:|---:|
| Obrazy razem | 6 515 | 6 515 (tabela 1) |
| Kłosy razem | 275 461 (275 468 ramek w CSV) | 275 187 (tabela 1) |
| train | 3 657 / 18 domen | 3 657 / 18 sub-datasetów |
| validation | 1 476 / 11 domen | 1 476 / 11 sub-datasetów |
| test | 1 382 / 18 domen | 1 373 / 18 sub-datasetów |

Liczba obrazów w train i walidacji zgadza się z preprintem. Test ma 9 obrazów więcej, a łączna liczba
kłosów jest większa o 274. Preprint powstał przed zakończeniem konkursu i jego tabela per domena
różni się od wydanych plików (np. UQ_10: 53 obrazy w preprincie, 106 w CSV). W pracy podajemy
liczby z wydanych plików.

---

## 3. Obrazy

| | |
|---|---|
| Liczba plików | 6 515, wszystkie z rozszerzeniem `.png`, łącznie 10,40 GB |
| Rozdzielczość | 1024 × 1024 px – wszystkie |
| Tryb kolorów | RGB (8 bitów na kanał), bez profilu ICC |
| **Faktyczny format** | **4 978 plików PNG (bezstratnie)** i **1 537 plików JPEG zapisanych z rozszerzeniem `.png`** |
| Jakość JPEG | 1 535 plików: tablica kwantyzacji `[2, 1, 1, 2, 2, 4, 5, 6, …]`, czyli jakość ok. 95 (IJG); 2 pliki: `[8, 6, 5, 8, 12, 20, 26, 31, …]`, czyli jakość ok. 75 |
| Skala | wg preprintu rozdzielczość przestrzenna (GSD) 0,2–0,4 mm/px; obrazy to wycinki 1024 × 1024 z większych zdjęć |

Pliki JPEG to dokładnie domeny **dodane w wersji 2021** (we wszystkich obrazach domeny):
- train (405 obrazów): Arvalis_7, Arvalis_8, Arvalis_9, Arvalis_10, Arvalis_11, Arvalis_12, NMBU_1, NMBU_2;
- test (1 132 z 1 382 obrazów): wszystkie domeny poza Terraref_1 i Terraref_2;
- walidacja: brak (same PNG).

Dwa pliki w jakości ok. 75 to obrazy Arvalis_12 `d8896363…` i `1961bcf4…`, czyli te, których nazwy
występują w CSV dwukrotnie (sekcja 6).

**Wniosek do pracy:** oryginał nie jest w całości bezstratny. Domeny z GWHD 2020 są w PNG, a domeny
dodane w 2021 – w JPEG wysokiej jakości. Zbiór testowy składa się w 82% z obrazów JPEG, a treningowy
w 11%. Biblioteki (PIL) rozpoznają format po zawartości, a nie po rozszerzeniu, więc wczytywanie
działa poprawnie.

---

## 4. Adnotacje

| | |
|---|---|
| Plik | `competition_{train,val,test}.csv`, kolumny `image_name,BoxesString,domain` |
| `image_name` | nazwa pliku **z rozszerzeniem** `.png` (opis na Zenodo podaje: „without the suffix” – niezgodne z plikami) |
| Format ramki | `x_min y_min x_max y_max` w pikselach, liczby całkowite; ramki rozdzielone `;` |
| Obraz bez kłosów | `BoxesString = no_box` (125 obrazów); wszystkie obrazy bez kłosów są oznaczone w ten sposób |

Format zweryfikowano na wszystkich 275 468 ramkach: warunek `x_max > x_min` i `y_max > y_min`
spełnia 100% ramek. Interpretacja `[x, y, w, h]` (warunek `x + w ≤ 1024`) dałaby tylko 24–25% poprawnych
ramek. Nie ma ramek o zerowej szerokości lub wysokości ani współrzędnych ujemnych.

Wymiary ramek (po przycięciu): mediana 73 × 67 px w train, 88 × 87 px w walidacji i 48 × 47 px w teście.
Minimum to 1 × 1 px (test), maksimum 705 × 491 px (test).

**Ramki wychodzące poza obraz:** 19 ramek w 5 obrazach testowych domeny CIMMYT_3 ma `x_max` lub
`y_max` do 1070 px przy szerokości obrazu 1024 px. Obsługa (`src/density.py: clip_boxes`):
- współrzędne przycinane do zakresu [0, 1024];
- ramka, która po przycięciu ma zerowe pole (leżała całkowicie poza obrazem), jest odrzucana – dotyczy
  to 7 ramek;
- środek ramki liczony jest z ramki przyciętej, więc zawsze leży wewnątrz obrazu.

| Obraz (początek nazwy) | Ramki w CSV | Przycięte | Pozostałe kłosy |
|---|---:|---:|---:|
| `7228cb212d54` | 40 | 1 | 40 |
| `0ca12ceda4a0` | 21 | 1 | 21 |
| `0af5c1bc7536` | 55 | 10 | 51 |
| `4c9c82eeefaa` | 33 | 4 | 32 |
| `6be51c1a5132` | 33 | 3 | 31 |

---

## 5. Podział danych – wybór

Społeczność używa dwóch podziałów GWHD 2021:

### a) Global Wheat Challenge 2021 (GWC 2021)
Podział z konkursu na AIcrowd, przeznaczony do porównywania metod lokalizacji i liczenia kłosów.
Zawierają go pliki `competition_*.csv` z archiwum. Według preprintu train pochodzi z Europy, a walidacja
(„public test”) i test („private test”) z pozostałych regionów. Usask_1 (Kanada) przeniesiono między
edycjami 2020 i 2021 z treningu do walidacji. Opis na Zenodo („training dataset will be the images
acquired in Europe and Canada”) jest pod tym względem nieaktualny.

### b) GlobalWheat-WILDS
Podział z benchmarku WILDS (Koh i in., 2021), przeznaczony do badania zmiany domeny. Zawiera zbiory OOD
z rozłącznych kontynentów oraz zbiory ID z sesji treningowych. Według streszczenia artykułu WILDS
(dodatek E.5, **liczby niezweryfikowane dosłownie**): train 2 943, val OOD 1 424, test OOD 1 434,
val ID 357, test ID 357.

### Wybór: GWC 2021
1. **Gotowy, oficjalny podział** w wydanych plikach CSV, bez dodatkowego przypisywania obrazów.
2. **Cel pracy:** porównanie metod liczenia (mapa gęstości i detekcja) to zadanie, do którego przeznaczono
   GWC 2021; badanie zmiany domeny nie jest głównym celem.
3. **Brak przecieku danych:** domeny są rozłączne między podziałami (sprawdzone).
4. **Powtarzalność:** podział jest publiczny i stały.

Konsekwencje:
- walidacja i test pochodzą z innych kontynentów niż trening, więc wyniki mierzą także uogólnianie na nowe
  lokalizacje;
- do wyboru epoki i hiperparametrów używany jest tylko zbiór walidacyjny, a test jest używany raz,
  na końcu;
- metryką konkursu była WDA (*weighted domain accuracy*) dla detekcji, a tu podstawowe metryki to
  MAE i RMSE liczby kłosów na obraz, więc wyniki nie są wprost porównywalne z rankingiem.

Planowane: analiza wyników w grupach według liczby kłosów na obraz, z progami wyznaczonymi na train.

---

## 6. Wykryte błędy w wydaniu i ich obsługa

### 6.1 Trzy nazwy plików przypisane do dwóch obrazów
W CSV trzy nazwy występują dwukrotnie: w dwóch domenach, z różnymi ramkami. Jednocześnie w archiwum
są trzy pliki PNG, do których nie odwołuje się żaden wiersz CSV (wszystkie trzy to JPEG, tak jak
pozostałe obrazy tych domen). Właściwe przypisanie ustalono wzrokowo: ramki z każdego wiersza naniesiono
na wszystkich kandydatów. W każdym przypadku ramki pokrywają się z kłosami tylko na jednym obrazie,
a na pozostałych trafiają w tło.

| Wiersz CSV: nazwa (początek), domena, podział | Ramek | Właściwy plik |
|---|---:|---|
| `d8896363…`, Arvalis_10, train | 25 | `8cc5870f73527da07937acc806002e1272e6200095c656ca326a680a90fab507.png` (nieodwołany w CSV) |
| `1961bcf4…`, Arvalis_10, train | 14 | `b588e1b55fbf8c4c6af08886013c0c36b70dd617fb1a5070829295a5c3ab31a8.png` (nieodwołany w CSV) |
| `da984651…`, CIMMYT_1, test | 58 | `094dcc9098204e6f751504515f3b0a7b5f5ad500a8bd2ec10124ed3d4fdbb6ed.png` (nieodwołany w CSV) |
| `d8896363…`, Arvalis_12, train | 42 | plik o tej nazwie |
| `1961bcf4…`, Arvalis_12, train | 45 | plik o tej nazwie |
| `da984651…`, CIMMYT_3, test | 24 | plik o tej nazwie |

Poprawka jest zapisana w kodzie (`src/dataset.py: NAME_FIXES`). Po niej każdy z 6515 wierszy wskazuje
inny plik i każdy plik z archiwum jest użyty dokładnie raz.

### 6.2 Ramki poza obrazem
Opisane w sekcji 4: 19 ramek w CIMMYT_3, przycięte; 7 odrzuconych.

### 6.3 Duplikat obrazu w teście
Pliki `12906c1e75f9…png` i `d747d8ff8dda…png` (Terraref_2, test) mają bajtowo identyczną zawartość,
ale różne adnotacje: 0 i 1 ramka. Oba są w tym samym podziale, więc nie powodują przecieku. Pozostawione;
wpływ na MAE jest pomijalny.

### 6.4 Metadane
- domena `ETHZ_1` w CSV ma w `metadata_dataset.csv` nazwę `Ethz_1` (inna wielkość liter); kod łączy
  nazwy bez rozróżniania wielkości liter;
- `ETHZ_1` (Szwajcaria) ma lokalizację „Usask” – prawdopodobnie błąd (Usask to Saskatchewan, Kanada);
- `Terraref_2` ma lokalizację z wiodącą spacją („ Maricopa,AZ”);
- `Utokyo_3` ma fazę rozwoju „multiple”, a `Ukyoto_1` – „Post-Flowering” (gdzie indziej „Post-flowering”).

**Filtrowanie:** brak. Używane są wszystkie 6 515 obrazów, łącznie ze 125 obrazami bez kłosów (ich mapa
gęstości składa się z samych zer).

---

## 7. Metadane w projekcie

| Kolumna / plik | Użycie |
|---|---|
| `image_name` | identyfikator i ścieżka obrazu w ZIP |
| `BoxesString` | ramki → przycięcie → środki → mapy gęstości |
| `domain` | zapis w predykcjach per obraz; analiza błędów per domena |
| `metadata_dataset.csv` (kraj, lokalizacja, faza rozwoju) | tylko opis zbioru i analiza wyników; nie są wejściem modelu |

---

## 8. Porównanie z kopią z Hugging Face (używaną wcześniej)

| | Zenodo (używane) | Hugging Face `GlobalWheat/GWHD_2021` |
|---|---|---|
| Obrazy | 6 515 | 6 512 (brak 3 plików z sekcji 6.1; dla podwójnych nazw zachowano wiersz Arvalis_12 / CIMMYT_3) |
| Format obrazów | PNG + JPEG q≈95 (sekcja 3) | wszystkie przekodowane do JPEG q≈75 |
| Ramki | xyxy, `no_box` | `[x, y, w, h]`, pusta lista (opis karty podaje błędnie xyxy) |
| Adnotacje | – | identyczne z Zenodo (po przycięciu, ±1 px) dla 6 508 obrazów; różne dla 4 obrazów CIMMYT_3 z ramkami poza kadrem |
| Różnica pikseli | – | PSNR ok. 36 dB względem oryginału (średnio, próbka 120 obrazów) |

---

## 9. Źródła

- E. David i in., *Global Wheat Head Dataset 2021: more diversity to improve the benchmarking of wheat
  head localization methods*, arXiv:2105.07660 (v2, 3.06.2021) – https://arxiv.org/abs/2105.07660
- Wersja czasopiśmiennicza: E. David i in., *Global Wheat Head Detection 2021: An Improved Dataset
  for Benchmarking Wheat Head Detection Methods*, Plant Phenomics, 2021 – **dane bibliograficzne do sprawdzenia**
- E. David i in., *Global Wheat Head Dataset 2021*, Zenodo, wersja 1.0, 2021, DOI 10.5281/zenodo.5092309
- P. W. Koh i in., *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML 2021 – https://arxiv.org/abs/2012.07421

---

## Dodatek A. Domeny (policzone; „JPEG” = liczba plików JPEG z rozszerzeniem .png)

### train – 18 domen, 3 657 obrazów, 163 690 kłosów

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Kłosy | Bez kłosów | JPEG |
|---|---|---|---|---:|---:|---:|---:|
| Arvalis_1 | Francja | Gréoux | Post-flowering | 66 | 2 935 | 0 | 0 |
| Arvalis_2 | Francja | Gréoux | Filling | 401 | 21 003 | 0 | 0 |
| Arvalis_3 | Francja | Gréoux | Filling - Ripening | 588 | 21 893 | 0 | 0 |
| Arvalis_4 | Francja | Gréoux | Filling | 204 | 4 270 | 0 | 0 |
| Arvalis_5 | Francja | VLB | Filling | 448 | 8 180 | 46 | 0 |
| Arvalis_6 | Francja | VSC | Filling - Ripening | 160 | 8 698 | 0 | 0 |
| Arvalis_7 | Francja | VLB | Filling - Ripening | 24 | 1 247 | 0 | 24 |
| Arvalis_8 | Francja | VLB | Filling - Ripening | 20 | 1 062 | 0 | 20 |
| Arvalis_9 | Francja | VLB | Ripening | 32 | 1 894 | 0 | 32 |
| Arvalis_10 | Francja | Mons | Filling | 60 | 1 563 | 2 | 60 |
| Arvalis_11 | Francja | VLB | Filling | 60 | 2 818 | 0 | 60 |
| Arvalis_12 | Francja | Gréoux | Filling | 29 | 1 277 | 0 | 29 |
| ETHZ_1 | Szwajcaria | „Usask” (sic) | Filling | 747 | 49 603 | 0 | 0 |
| Inrae_1 | Francja | Toulouse | Filling - Ripening | 176 | 3 634 | 0 | 0 |
| NMBU_1 | Norwegia | NMBU | Filling | 82 | 7 345 | 0 | 82 |
| NMBU_2 | Norwegia | NMBU | Ripening | 98 | 5 211 | 2 | 98 |
| Rres_1 | Wielka Brytania | Rothamsted | Filling - Ripening | 432 | 19 210 | 0 | 0 |
| ULiège-GxABT_1 | Belgia | Gembloux | Ripening | 30 | 1 847 | 0 | 0 |

### validation – 11 domen, 1 476 obrazów, 44 347 kłosów

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Kłosy | Bez kłosów | JPEG |
|---|---|---|---|---:|---:|---:|---:|
| NAU_1 | Chiny | Baima | Post-flowering | 20 | 1 240 | 0 | 0 |
| UQ_1 | Australia | Gatton | Post-flowering | 22 | 640 | 1 | 0 |
| UQ_2 | Australia | Gatton | Post-flowering | 16 | 39 | 13 | 0 |
| UQ_3 | Australia | Gatton | Filling | 14 | 297 | 4 | 0 |
| UQ_4 | Australia | Gatton | Filling | 30 | 1 039 | 1 | 0 |
| UQ_5 | Australia | Gatton | Filling - Ripening | 30 | 3 680 | 1 | 0 |
| UQ_6 | Australia | Gatton | Filling - Ripening | 30 | 1 147 | 2 | 0 |
| Usask_1 | Kanada | Saskatchewan | Filling - Ripening | 200 | 5 985 | 0 | 0 |
| Utokyo_1 | Japonia | NARO-Tsukuba | Ripening | 538 | 14 185 | 0 | 0 |
| Utokyo_2 | Japonia | NARO-Tsukuba | Ripening | 456 | 13 010 | 0 | 0 |
| Utokyo_3 | Japonia | NARO-Hokkaido | multiple | 120 | 3 085 | 6 | 0 |

### test – 18 domen, 1 382 obrazy, 67 424 kłosy

| Domena | Kraj | Lokalizacja | Faza rozwoju | Obrazy | Kłosy | Bez kłosów | JPEG |
|---|---|---|---|---:|---:|---:|---:|
| ARC_1 | Sudan | Wad Medani | Filling | 30 | 1 169 | 0 | 30 |
| CIMMYT_1 | Meksyk | Ciudad Obregon | Post-flowering | 69 | 2 843 | 3 | 69 |
| CIMMYT_2 | Meksyk | Ciudad Obregon | Post-flowering | 77 | 2 771 | 3 | 77 |
| CIMMYT_3 | Meksyk | Ciudad Obregon | Post-flowering | 60 | 1 554 | 8 | 60 |
| KSU_1 | USA | KSU | Post-flowering | 100 | 6 435 | 0 | 100 |
| KSU_2 | USA | KSU | Post-flowering | 100 | 5 302 | 0 | 100 |
| KSU_3 | USA | KSU | Filling | 95 | 5 217 | 0 | 95 |
| KSU_4 | USA | KSU | Ripening | 60 | 3 285 | 0 | 60 |
| NAU_2 | Chiny | Baima | Post-flowering | 100 | 4 918 | 0 | 100 |
| NAU_3 | Chiny | Baima | Filling | 100 | 4 596 | 0 | 100 |
| Terraref_1 | USA | Maricopa, AZ | Ripening | 144 | 3 360 | 11 | 0 |
| Terraref_2 | USA | Maricopa, AZ | Filling | 106 | 1 274 | 20 | 0 |
| UQ_7 | Australia | Gatton | Ripening | 17 | 1 335 | 0 | 17 |
| UQ_8 | Australia | McAllister | Ripening | 41 | 4 835 | 0 | 41 |
| UQ_9 | Australia | Brookstead | Filling - Ripening | 33 | 2 886 | 0 | 33 |
| UQ_10 | Australia | Gatton | Filling - Ripening | 106 | 8 629 | 0 | 106 |
| UQ_11 | Australia | Gatton | Post-flowering | 84 | 4 345 | 0 | 84 |
| Ukyoto_1 | Japonia | Kyoto | Post-Flowering | 60 | 2 670 | 2 | 60 |
