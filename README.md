# Bayesian modeling of Premier League standings

This project builds probabilistic models for forecasting Premier League team points from historical match data. The models do not predict individual matches. The prediction unit is one team in one season, and the output is a full predictive distribution for that team's final points.

The analysis is implemented in Python, Jupyter notebooks, and Stan. It compares two Bayesian team-level models:

- Model 1: a static team model with latent club skill and lagged process features.
- Model 2: a hierarchical team model with long-run club quality, season effects, and promoted-team status.

## Project goal

The goal is to build and compare models that answer the following questions:

- how many points a given Premier League team may score in a season,
- how uncertain that forecast is,
- whether lagged process features improve prediction compared with a cleaner hierarchical baseline,
- how the models perform in a 2025/26 backtest,
- what the points forecast looks like for the 2026/27 season.

A full league table is assembled only as a derived output by running point forecasts for all 20 teams and ranking them by predicted points.

## Data

The data is stored in `Data/datahub.io` and comes from [datahub.io - English Premier League](https://datahub.io/football/english-premier-league), which aggregates data originally collected by [football-data.co.uk](https://www.football-data.co.uk/).

The repository contains `season-*.csv` files for seasons from `0910` to `2526`. Each row represents one match. Key columns include:

- `Date` - match date,
- `HomeTeam`, `AwayTeam` - home and away teams,
- `FTHG`, `FTAG` - full-time goals,
- `FTR` - full-time result: `H`, `D`, or `A`,
- `HST`, `AST` - home and away shots on target,
- additional match statistics: shots, fouls, corners, and cards.

The Stan models are not trained directly on match-level rows. Match data is aggregated into season tables, where each row represents one `(season, team)` pair and the target variable is final points `Pts`.

## Project structure

```text
.
├── Data/datahub.io/                  # Premier League match data
├── stan/
│   ├── team_static.stan              # Model 1: static team skill + covariates
│   └── team_hierarchical.stan        # Model 2: hierarchical team model
├── helping_functions.py              # Data loading, features, prediction, and metrics
├── 00_project_overview_and_priors.ipynb
├── 01_eda_and_tables.ipynb
├── 02_model1_static_team.ipynb
├── 03_model2_hierarchical_team.ipynb
├── 04_forecast_2627_comparison.ipynb
└── 05_backtest_models_comparison.ipynb
```

## Notebooks

| File | Description |
| --- | --- |
| `00_project_overview_and_priors.ipynb` | Project overview, model assumptions, and prior justification. |
| `01_eda_and_tables.ipynb` | Exploratory data analysis, data loading, and season table construction. |
| `02_model1_static_team.ipynb` | Model 1 fitting, diagnostics, PPC, and point backtest. |
| `03_model2_hierarchical_team.ipynb` | Model 2 fitting, diagnostics, PPC, LOO/WAIC, and backtest. |
| `04_forecast_2627_comparison.ipynb` | 2026/27 forecast and model comparison using LOO/WAIC. |
| `05_backtest_models_comparison.ipynb` | Direct Model 1 vs Model 2 backtest on the 2025/26 season. |

## Models

### Model 1: `team_static.stan`

The static model assumes one latent skill value for each team across the full training period. The point prediction uses:

- team index,
- `sot_diff_pg` - shots-on-target difference per match from the previous season,
- `pts_lag1` - points from the previous season,
- `ppg_last10` - points per game over the final 10 matches of the previous season,
- `is_promoted` - whether the team is promoted or absent from the previous Premier League season.

The likelihood uses a Student-t distribution with fixed `nu = 5`, making the model more robust to unusual seasons.

### Model 2: `team_hierarchical.stan`

The hierarchical model intentionally excludes process features from the likelihood. It acts as a baseline based on latent structure:

- `team_skill` - persistent team quality,
- `season_effect` - shared season-level effect,
- `is_promoted` - promoted-team effect,
- Student-t residual noise.

This model tests how much can be forecast from team and season structure alone, without additional lagged process covariates.

## Analysis pipeline

1. Load all `season-*.csv` files.
2. Add the `season` column and parse match dates.
3. Build final season tables with `compute_table`.
4. Generate team-season features with `load_season_tables`.
5. Standardize continuous features on the training data.
6. Fit Stan models through `cmdstanpy`.
7. Check sampler diagnostics: divergences, R-hat, ESS, and E-BFMI.
8. Run posterior predictive checks.
9. Backtest the 2025/26 season.
10. Forecast the 2026/27 season and compare models.

## Installation

The project requires Python and CmdStan. Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn arviz cmdstanpy jupyter
python -m cmdstanpy.install_cmdstan
```

Then start Jupyter:

```bash
jupyter lab
```

## Usage

Recommended notebook order:

1. `00_project_overview_and_priors.ipynb` - read the assumptions and priors.
2. `01_eda_and_tables.ipynb` - inspect the data and generated season tables.
3. `02_model1_static_team.ipynb` - fit Model 1.
4. `03_model2_hierarchical_team.ipynb` - fit Model 2.
5. `05_backtest_models_comparison.ipynb` - compare the models on the 2025/26 season.
6. `04_forecast_2627_comparison.ipynb` - run the 2026/27 forecast and LOO/WAIC comparison.

Example helper-function usage:

```python
import helping_functions as hf

matches = hf.load_matches()
tables = hf.load_season_tables(matches, hf.ALL_TRAIN_SEASONS)
table_2526 = hf.compute_table(matches, "2526")
teams_2627 = hf.pl_2627_squad(matches)
```

## Results and interpretation

The main model output is a predictive distribution for one team's final points:

- mean and median predicted points,
- uncertainty intervals, such as 5% and 95% quantiles,
- point forecast error in backtesting,
- league-table ranking as a derived summary of point forecasts.

In the current version, `04_forecast_2627_comparison.ipynb` reports that PSIS-LOO favors Model 1. Model 2 remains useful as a simpler and more interpretable hierarchical baseline.

## Key helper functions

- `load_matches` - loads all seasons into one DataFrame.
- `compute_table` - builds a league table for one season.
- `load_season_tables` - creates team-season rows with features.
- `prepare_table_stan_static` - prepares Stan data for Model 1.
- `prepare_table_stan_hierarchical` - prepares Stan data for Model 2.
- `predict_team_points` - generates posterior predictive points for one team.
- `build_predicted_table` - turns multiple team point forecasts into a ranking.
- `compare_forecast_to_actual` - compares a forecast with the actual season table.
- `forecast_season_summary` - computes backtest metrics.

## Authors

Kacper Ciesla and Tomasz Drag.
