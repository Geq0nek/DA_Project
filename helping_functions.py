from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "Data" / "datahub.io"


def load_matches(data_path: Path | None = None) -> pd.DataFrame:
    """Load all season-*.csv files into one matches DataFrame."""
    data_path = Path(data_path or DEFAULT_DATA_PATH)
    frames = []
    for file in sorted(data_path.glob("season-*.csv")):
        df = pd.read_csv(file)
        df["season"] = file.stem.replace("season-", "")
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"])
    matches["total_goals"] = matches["FTHG"] + matches["FTAG"]
    return matches


def prepare_stan_data(matches: pd.DataFrame,train_seasons: Iterable[str]) -> tuple[dict, dict[str, int], list[str]]:
    """
    Build Stan data dict (1-based team indices) from training seasons only.

    Returns (stan_data, team_to_idx, team_names).
    """
    train = matches[matches["season"].isin(list(train_seasons))].copy()
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    team_to_idx = {name: i + 1 for i, name in enumerate(teams)}  # Stan is 1-based

    home = train["HomeTeam"].map(team_to_idx).to_numpy()
    away = train["AwayTeam"].map(team_to_idx).to_numpy()

    stan_data = {
        "N": len(train),
        "T": len(teams),
        "home": home.astype(int),
        "away": away.astype(int),
        "goals_h": train["FTHG"].astype(int).to_numpy(),
        "goals_a": train["FTAG"].astype(int).to_numpy(),
    }
    return stan_data, team_to_idx, teams


def prepare_stan_data_hierarchical(
    matches: pd.DataFrame,
    train_seasons: Iterable[str],
) -> tuple[dict, dict[str, int], list[str], dict[str, int]]:
    """
    Stan data with per-match season index (1-based).

    Returns (stan_data, team_to_idx, team_names, season_to_idx).
    """
    ordered_seasons = list(train_seasons)
    season_to_idx = {s: i + 1 for i, s in enumerate(ordered_seasons)}

    train = matches[matches["season"].isin(ordered_seasons)].copy()
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    team_to_idx = {name: i + 1 for i, name in enumerate(teams)}

    stan_data = {
        "N": len(train),
        "S": len(ordered_seasons),
        "T": len(teams),
        "season": train["season"].map(season_to_idx).astype(int).to_numpy(),
        "home": train["HomeTeam"].map(team_to_idx).astype(int).to_numpy(),
        "away": train["AwayTeam"].map(team_to_idx).astype(int).to_numpy(),
        "goals_h": train["FTHG"].astype(int).to_numpy(),
        "goals_a": train["FTAG"].astype(int).to_numpy(),
    }
    return stan_data, team_to_idx, teams, season_to_idx


def simulate_seasons_from_hierarchical_draws(
    fit,
    teams: list[str],
    team_to_idx: dict[str, int],
    last_season_index: int,
    n_table_sims: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Forecast using attack/defense from the last training season (Stan index S).
    """
    rng = np.random.default_rng(seed)
    att_draws = fit.stan_variable("att")  # (draws, S, T)
    def_draws = fit.stan_variable("def")
    home_adv_draws = fit.stan_variable("home_adv")

    s_idx = last_season_index - 1  # 0-based slice for last season
    n_draws = att_draws.shape[0]

    position_records = {t: [] for t in teams}
    points_records = {t: [] for t in teams}

    for _ in range(n_table_sims):
        d = rng.integers(0, n_draws)
        att = {t: 0.0 for t in teams}
        def_ = {t: 0.0 for t in teams}
        for team in teams:
            if team in team_to_idx:
                j = team_to_idx[team] - 1
                att[team] = att_draws[d, s_idx, j]
                def_[team] = def_draws[d, s_idx, j]

        table = simulate_season_table(teams, att, def_, home_adv_draws[d], rng)
        for _, row in table.iterrows():
            position_records[row["team"]].append(row["position"])
            points_records[row["team"]].append(row["Pts"])

    summary = pd.DataFrame({"team": teams})
    summary["pos_median"] = [np.median(position_records[t]) for t in teams]
    summary["pos_mean"] = [np.mean(position_records[t]) for t in teams]
    summary["pts_median"] = [np.median(points_records[t]) for t in teams]
    summary["pts_mean"] = [np.mean(points_records[t]) for t in teams]
    return summary.sort_values("pos_median").reset_index(drop=True)


def season_fixture_pairs(teams: list[str]) -> list[tuple[str, str]]:
    """Round-robin home/away fixtures (each pair plays twice)."""
    pairs = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i != j:
                pairs.append((home, away))
    return pairs


def simulate_match_goals(att_h, def_h, att_a, def_a, home_adv, rng):
    log_lam_h = home_adv + att_h - def_a
    log_lam_a = att_a - def_h
    return rng.poisson(np.exp(log_lam_h)), rng.poisson(np.exp(log_lam_a))


def simulate_season_table(teams: list[str], att: dict[str, float], def_: dict[str, float], home_adv: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate one full double round-robin and return league table."""
    tab = {t: {"Pts": 0, "GF": 0, "GA": 0} for t in teams}
    for home, away in season_fixture_pairs(teams):
        gh, ga = simulate_match_goals(
            att[home], def_[home], att[away], def_[away], home_adv, rng
        )
        tab[home]["GF"] += gh
        tab[home]["GA"] += ga
        tab[away]["GF"] += ga
        tab[away]["GA"] += gh
        if gh > ga:
            tab[home]["Pts"] += 3
        elif gh < ga:
            tab[away]["Pts"] += 3
        else:
            tab[home]["Pts"] += 1
            tab[away]["Pts"] += 1

    rows = [
        {"team": t, "Pts": p["Pts"], "GD": p["GF"] - p["GA"], "GF": p["GF"], "GA": p["GA"]}
        for t, p in tab.items()
    ]
    out = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=False)
    out["position"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def simulate_seasons_from_draws(fit, teams: list[str], team_to_idx: dict[str, int], n_table_sims: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate many season tables using posterior draws from CmdStanPy fit.
    Unknown teams (not in training index) get att=0, def=0.
    """
    rng = np.random.default_rng(seed)
    att_draws = fit.stan_variable("att")  # (draws, T)
    def_draws = fit.stan_variable("def")
    home_adv_draws = fit.stan_variable("home_adv")

    n_draws = att_draws.shape[0]
    idx_to_team = {idx: name for name, idx in team_to_idx.items()}

    position_records = {t: [] for t in teams}
    points_records = {t: [] for t in teams}

    for _ in range(n_table_sims):
        d = rng.integers(0, n_draws)
        att = {t: 0.0 for t in teams}
        def_ = {t: 0.0 for t in teams}
        for team in teams:
            if team in team_to_idx:
                j = team_to_idx[team] - 1
                att[team] = att_draws[d, j]
                def_[team] = def_draws[d, j]

        table = simulate_season_table(teams, att, def_, home_adv_draws[d], rng)
        for _, row in table.iterrows():
            position_records[row["team"]].append(row["position"])
            points_records[row["team"]].append(row["Pts"])

    summary = pd.DataFrame({"team": teams})
    summary["pos_median"] = [np.median(position_records[t]) for t in teams]
    summary["pos_mean"] = [np.mean(position_records[t]) for t in teams]
    summary["pts_median"] = [np.median(points_records[t]) for t in teams]
    summary["pts_mean"] = [np.mean(points_records[t]) for t in teams]
    return summary.sort_values("pos_median").reset_index(drop=True)


def compute_table(df, season):
    """Table of the league from the matches of one season."""
    s = df[df["season"] == season].copy()
    teams = sorted(set(s["HomeTeam"]) | set(s["AwayTeam"]))
    tab = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0} for t in teams}
    for _, r in s.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        gh, ga = int(r["FTHG"]), int(r["FTAG"])
        tab[h]["GF"] += gh
        tab[h]["GA"] += ga
        tab[a]["GF"] += ga
        tab[a]["GA"] += gh
        if r["FTR"] == "H":
            tab[h]["W"] += 1
            tab[a]["L"] += 1
        elif r["FTR"] == "A":
            tab[a]["W"] += 1
            tab[h]["L"] += 1
        else:
            tab[h]["D"] += 1
            tab[a]["D"] += 1
    rows = []
    for t, p in tab.items():
        pts = 3 * p["W"] + p["D"]
        rows.append({
            "team": t, "Pts": pts, "GD": p["GF"] - p["GA"],
            "GF": p["GF"], "GA": p["GA"], "W": p["W"], "D": p["D"], "L": p["L"],
        })
    out = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=False)
    out["position"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)

def teams_in_season(df, season):
    s = df[df["season"] == season]
    return set(s["HomeTeam"]) | set(s["AwayTeam"])


# 2026/27 squad (user-confirmed promotion / relegation)
RELEGATED_FROM_2526 = {"Burnley", "West Ham", "Wolves"}
PROMOTED_TO_2627 = {"Coventry", "Ipswich", "Hull"}


def pl_2627_squad(matches: pd.DataFrame) -> list[str]:
    """20 teams for forecast season 2026/27."""
    return sorted(
        (teams_in_season(matches, "2526") - RELEGATED_FROM_2526) | PROMOTED_TO_2627
    )


ALL_TRAIN_SEASONS = [
    "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617",
    "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526",
]

BACKTEST_TRAIN_SEASONS = ALL_TRAIN_SEASONS[:-1]  # through 2425
BACKTEST_TEST_SEASON = "2526"
FORECAST_TRAIN_SEASONS = ALL_TRAIN_SEASONS  # through 2526
FORECAST_SEASON_LABEL = "2627"