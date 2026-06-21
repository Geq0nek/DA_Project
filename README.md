# Bayesian modeling of Premier League standings

Projekt analityczny do probabilistycznego prognozowania liczby punktow druzyn Premier League na podstawie historycznych danych meczowych. Modele nie przewiduja pojedynczych meczow. Jednostka predykcji to jedna druzyna w jednym sezonie, a wynikiem jest rozklad punktow tej druzyny.

Projekt powstal jako analiza bayesowska w Pythonie, notebookach Jupyter i Stan. Porownuje dwa podejscia:

- Model 1: statyczny model druzynowy z latentnym skillem klubu i cechami procesowymi.
- Model 2: hierarchiczny model druzynowy z dlugookresowa jakoscia klubu, efektem sezonu i statusem beniaminka.

## Cel projektu

Celem jest zbudowanie i porownanie modeli, ktore odpowiadaja na pytania:

- ile punktow moze zdobyc dana druzyna w sezonie Premier League,
- jak duza jest niepewnosc tej prognozy,
- czy cechy procesowe z poprzedniego sezonu poprawiaja predykcje wzgledem czystego modelu hierarchicznego,
- jak modele wypadaja w backtescie sezonu 2025/26,
- jak wyglada prognoza punktow na sezon 2026/27.

Pelna tabela ligowa jest skladana opcjonalnie przez uruchomienie predykcji punktow dla 20 druzyn i posortowanie ich po prognozowanych punktach.

## Dane

Dane znajduja sie w katalogu `Data/datahub.io` i pochodza z [datahub.io - English Premier League](https://datahub.io/football/english-premier-league), ktore agreguje dane z [football-data.co.uk](https://www.football-data.co.uk/).

W repo znajduja sie pliki `season-*.csv` dla sezonow od `0910` do `2526`. Kazdy wiersz oznacza jeden mecz. Najwazniejsze kolumny:

- `Date` - data meczu,
- `HomeTeam`, `AwayTeam` - gospodarze i goscie,
- `FTHG`, `FTAG` - gole po pelnym czasie,
- `FTR` - wynik meczu: `H`, `D`, `A`,
- `HST`, `AST` - strzaly celne gospodarzy i gosci,
- dodatkowe statystyki meczowe: strzaly, faule, rozne, kartki.

Modele sa trenowane nie na poziomie meczu, ale po agregacji do tabel sezonowych: jeden wiersz oznacza pare `(season, team)` i koncowa liczbe punktow `Pts`.

## Struktura projektu

```text
.
├── Data/datahub.io/                  # Dane meczowe Premier League
├── stan/
│   ├── team_static.stan              # Model 1: statyczny skill druzyny + cechy
│   └── team_hierarchical.stan        # Model 2: model hierarchiczny
├── helping_functions.py              # Funkcje do wczytywania danych, cech, predykcji i metryk
├── 00_project_overview_and_priors.ipynb
├── 01_eda_and_tables.ipynb
├── 02_model1_static_team.ipynb
├── 03_model2_hierarchical_team.ipynb
├── 04_forecast_2627_comparison.ipynb
└── 05_backtest_models_comparison.ipynb
```

## Notebooki

| Plik | Opis |
| --- | --- |
| `00_project_overview_and_priors.ipynb` | Opis projektu, zalozenia modeli i uzasadnienie priorow. |
| `01_eda_and_tables.ipynb` | EDA, wczytanie danych i budowa tabel sezonowych. |
| `02_model1_static_team.ipynb` | Dopasowanie Modelu 1, diagnostyka, PPC i backtest punktow. |
| `03_model2_hierarchical_team.ipynb` | Dopasowanie Modelu 2, diagnostyka, PPC, LOO/WAIC i backtest. |
| `04_forecast_2627_comparison.ipynb` | Forecast sezonu 2026/27 i porownanie modeli przez LOO/WAIC. |
| `05_backtest_models_comparison.ipynb` | Bezposredni backtest Model 1 vs Model 2 na sezonie 2025/26. |

## Modele

### Model 1: `team_static.stan`

Model statyczny zaklada jeden latentny skill dla kazdej druzyny w calym okresie treningowym. Predykcja punktow korzysta z:

- indeksu druzyny,
- `sot_diff_pg` - roznicy strzalow celnych na mecz z poprzedniego sezonu,
- `pts_lag1` - punktow z poprzedniego sezonu,
- `ppg_last10` - punktow na mecz w ostatnich 10 kolejkach poprzedniego sezonu,
- `is_promoted` - informacji, czy druzyna jest beniaminkiem albo nie grala w poprzednim sezonie Premier League.

Likelihood opiera sie na rozkladzie Studenta-t z ustalonym `nu = 5`, co zwieksza odpornosc modelu na nietypowe sezony.

### Model 2: `team_hierarchical.stan`

Model hierarchiczny nie uzywa cech procesowych w likelihoodzie. Jest baseline'em opartym na latentnej strukturze:

- `team_skill` - trwala jakosc druzyny,
- `season_effect` - wspolny efekt sezonu,
- `is_promoted` - efekt beniaminka,
- Student-t residual noise.

Dzieki temu model testuje, ile da sie przewidziec sama struktura druzyn i sezonow, bez dodatkowych cech z poprzedniego sezonu.

## Pipeline analizy

1. Wczytanie wszystkich plikow `season-*.csv`.
2. Dodanie kolumny `season` i parsowanie dat.
3. Budowa tabel koncowych przez `compute_table`.
4. Wyliczenie cech druzyna-sezon przez `load_season_tables`.
5. Standaryzacja cech ciaglych na danych treningowych.
6. Dopasowanie modeli Stan przez `cmdstanpy`.
7. Diagnostyka samplera: divergences, R-hat, ESS, E-BFMI.
8. Posterior predictive checks.
9. Backtest sezonu 2025/26.
10. Forecast sezonu 2026/27 i porownanie modeli.

## Instalacja

Projekt wymaga Pythona oraz CmdStan. Przykladowe srodowisko:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn arviz cmdstanpy jupyter
python -m cmdstanpy.install_cmdstan
```

Po instalacji mozna uruchomic Jupyter:

```bash
jupyter lab
```

## Uruchamianie

Zalecana kolejnosc pracy:

1. `00_project_overview_and_priors.ipynb` - przeczytaj zalozenia projektu i priory.
2. `01_eda_and_tables.ipynb` - sprawdz dane i tabele sezonowe.
3. `02_model1_static_team.ipynb` - dopasuj Model 1.
4. `03_model2_hierarchical_team.ipynb` - dopasuj Model 2.
5. `05_backtest_models_comparison.ipynb` - porownaj modele na sezonie 2025/26.
6. `04_forecast_2627_comparison.ipynb` - wykonaj forecast 2026/27 i porownanie LOO/WAIC.

Przykladowe uzycie funkcji pomocniczych:

```python
import helping_functions as hf

matches = hf.load_matches()
tables = hf.load_season_tables(matches, hf.ALL_TRAIN_SEASONS)
table_2526 = hf.compute_table(matches, "2526")
teams_2627 = hf.pl_2627_squad(matches)
```

## Wyniki i interpretacja

Glownym wynikiem modelu jest rozklad punktow dla jednej druzyny:

- srednia i mediana prognozowanych punktow,
- przedzialy niepewnosci, np. kwantyle 5% i 95%,
- blad punktow w backtescie,
- ranking tabeli jako pochodna prognoz punktowych.

W obecnej wersji projektu notebook `04_forecast_2627_comparison.ipynb` wskazuje, ze PSIS-LOO faworyzuje Model 1. Model 2 pozostaje waznym punktem odniesienia, bo jest prostszym, interpretowalnym baseline'em hierarchicznym.

## Najwazniejsze funkcje

- `load_matches` - wczytuje wszystkie sezony do jednej ramki danych.
- `compute_table` - buduje tabele ligowa dla jednego sezonu.
- `load_season_tables` - tworzy dane druzyna-sezon z cechami.
- `prepare_table_stan_static` - przygotowuje dane dla Modelu 1.
- `prepare_table_stan_hierarchical` - przygotowuje dane dla Modelu 2.
- `predict_team_points` - generuje posterior predictive points dla jednej druzyny.
- `build_predicted_table` - sklada predykcje punktow wielu druzyn w ranking.
- `compare_forecast_to_actual` - porownuje forecast z rzeczywista tabela.
- `forecast_season_summary` - liczy metryki backtestu.

## Autorzy

Kacper Ciesla i Tomasz Drag.
