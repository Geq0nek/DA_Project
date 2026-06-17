from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "Data" / "datahub.io"
STUDENT_T_NU = 5.0  # fixed df in table Stan models (robust regression default)


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


TABLE_FEATURE_COLS = ("sot_diff_pg", "pts_lag1", "ppg_last10")


def compute_team_season_features(
    matches: pd.DataFrame,
    season: str,
    ordered_seasons: Iterable[str],
    pts_by_season_team: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """
    Process features per team for one season (from match-level CSV).

    - sot_diff_pg: (shots on target for − against) per match
    - pts_lag1: points in the previous season (0 if promoted / first season)
    - ppg_last10: points per game over the last 10 league matches
    """
    ordered = list(ordered_seasons)
    s = matches[matches["season"] == season].sort_values("Date")
    teams = sorted(set(s["HomeTeam"]) | set(s["AwayTeam"]))

    sot_for: dict[str, float] = {t: 0.0 for t in teams}
    sot_against: dict[str, float] = {t: 0.0 for t in teams}
    match_pts: dict[str, list[float]] = {t: [] for t in teams}
    n_games: dict[str, int] = {t: 0 for t in teams}

    for _, r in s.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        sot_for[h] += float(r["HST"])
        sot_against[h] += float(r["AST"])
        sot_for[a] += float(r["AST"])
        sot_against[a] += float(r["HST"])

        if r["FTR"] == "H":
            match_pts[h].append(3.0)
            match_pts[a].append(0.0)
        elif r["FTR"] == "A":
            match_pts[h].append(0.0)
            match_pts[a].append(3.0)
        else:
            match_pts[h].append(1.0)
            match_pts[a].append(1.0)
        n_games[h] += 1
        n_games[a] += 1

    prev_season = None
    if season in ordered:
        idx = ordered.index(season)
        if idx > 0:
            prev_season = ordered[idx - 1]

    if pts_by_season_team is None:
        pts_by_season_team = {}
        if prev_season is not None:
            prev_tab = compute_table(matches, prev_season)
            for _, row in prev_tab.iterrows():
                pts_by_season_team[(prev_season, row["team"])] = float(row["Pts"])

    rows = []
    for team in teams:
        ng = max(n_games[team], 1)
        sot_diff_pg = (sot_for[team] - sot_against[team]) / ng
        last10 = match_pts[team][-10:]
        ppg_last10 = float(np.mean(last10)) if last10 else 0.0
        if prev_season is None:
            pts_lag1 = 0.0
        else:
            pts_lag1 = pts_by_season_team.get((prev_season, team), 0.0)

        rows.append({
            "season": season,
            "team": team,
            "sot_diff_pg": sot_diff_pg,
            "pts_lag1": pts_lag1,
            "ppg_last10": ppg_last10,
        })
    return pd.DataFrame(rows)


def _standardize_features(
    df: pd.DataFrame,
    cols: Iterable[str] = TABLE_FEATURE_COLS,
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score feature columns; return updated frame and (mean, std) per column."""
    out = df.copy()
    if stats is None:
        stats = {}
        for col in cols:
            mu = float(out[col].mean())
            sd = float(out[col].std())
            if sd <= 0:
                sd = 1.0
            stats[col] = (mu, sd)
    for col in cols:
        mu, sd = stats[col]
        out[col] = (out[col] - mu) / sd
    return out, stats


def load_season_tables(
    matches: pd.DataFrame,
    seasons: Iterable[str],
    *,
    with_features: bool = True,
    lag_features: bool = True,
) -> pd.DataFrame:
    """Long-format league tables: one row per (season, team), optional process features.

    With lag_features=True, features for target season s are taken from season s-1.
    That avoids leaking same-season shots/form into a pre-season table forecast.
    """
    ordered = sorted(seasons)
    frames = []
    pts_lookup: dict[tuple[str, str], float] = {}
    for season in ordered:
        tab = compute_table(matches, season)
        for _, row in tab.iterrows():
            pts_lookup[(season, row["team"])] = float(row["Pts"])
        tab = tab.assign(season=season)
        frames.append(tab)
    tables = pd.concat(frames, ignore_index=True)

    if not with_features:
        return tables

    raw_feat_frames = [
        compute_team_season_features(matches, season, ordered, pts_lookup)
        for season in ordered
    ]
    raw_features = pd.concat(raw_feat_frames, ignore_index=True)

    if not lag_features:
        return tables.merge(raw_features, on=["season", "team"], how="left")

    raw_by_season_team = raw_features.set_index(["season", "team"])
    rows = []
    for _, row in tables[["season", "team"]].iterrows():
        season = row["season"]
        team = row["team"]
        season_idx = ordered.index(season)
        if season_idx == 0:
            feature_row = {
                "season": season,
                "team": team,
                "sot_diff_pg": 0.0,
                "pts_lag1": 0.0,
                "ppg_last10": 0.0,
            }
        else:
            prev_season = ordered[season_idx - 1]
            if (prev_season, team) in raw_by_season_team.index:
                prev = raw_by_season_team.loc[(prev_season, team)]
                feature_row = {
                    "season": season,
                    "team": team,
                    "sot_diff_pg": float(prev["sot_diff_pg"]),
                    "pts_lag1": float(pts_lookup.get((prev_season, team), 0.0)),
                    "ppg_last10": float(prev["ppg_last10"]),
                }
            else:
                feature_row = {
                    "season": season,
                    "team": team,
                    "sot_diff_pg": 0.0,
                    "pts_lag1": 0.0,
                    "ppg_last10": 0.0,
                }
        rows.append(feature_row)
    features = pd.DataFrame(rows)
    return tables.merge(features, on=["season", "team"], how="left")


def build_forecast_features(
    matches: pd.DataFrame,
    feature_season: str,
    teams: list[str],
    ordered_seasons: Iterable[str],
    feature_stats: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """
    Covariates for forecast teams from one reference season (z-scored).

    Teams absent in feature_season get 0 on the z-scale (training mean).
    """
    ordered = list(ordered_seasons)
    feat = compute_team_season_features(matches, feature_season, ordered)
    feat = feat.set_index("team")
    pts_last = compute_table(matches, feature_season).set_index("team")["Pts"]
    raw_rows = []
    for team in teams:
        if team in feat.index:
            row = feat.loc[team]
            raw_rows.append({
                "team": team,
                "sot_diff_pg": float(row["sot_diff_pg"]),
                "pts_lag1": float(pts_last.get(team, 0.0)),
                "ppg_last10": float(row["ppg_last10"]),
            })
        else:
            raw_rows.append({
                "team": team,
                "sot_diff_pg": feature_stats["sot_diff_pg"][0],
                "pts_lag1": feature_stats["pts_lag1"][0],
                "ppg_last10": feature_stats["ppg_last10"][0],
            })
    raw_df = pd.DataFrame(raw_rows)
    z_df, _ = _standardize_features(raw_df, stats=feature_stats)
    return {
        row["team"]: {
            "sot_diff_pg": float(row["sot_diff_pg"]),
            "pts_lag1": float(row["pts_lag1"]),
            "ppg_last10": float(row["ppg_last10"]),
        }
        for _, row in z_df.iterrows()
    }


def prepare_table_stan_static(
    tables: pd.DataFrame,
    train_seasons: Iterable[str],
) -> tuple[dict, dict[str, int], list[str], dict[str, tuple[float, float]]]:
    """Stan data for table_static.stan from historical tables + features."""
    train = tables[tables["season"].isin(list(train_seasons))].copy()
    teams = sorted(train["team"].unique())
    team_to_idx = {name: i + 1 for i, name in enumerate(teams)}

    train, feature_stats = _standardize_features(train)

    stan_data = {
        "N": len(train),
        "T": len(teams),
        "nu": STUDENT_T_NU,
        "team": train["team"].map(team_to_idx).astype(int).to_numpy(),
        "pts": train["Pts"].astype(float).to_numpy(),
        "sot_diff_pg": train["sot_diff_pg"].astype(float).to_numpy(),
        "pts_lag1": train["pts_lag1"].astype(float).to_numpy(),
        "ppg_last10": train["ppg_last10"].astype(float).to_numpy(),
    }
    return stan_data, team_to_idx, teams, feature_stats


def prepare_table_stan_hierarchical(
    tables: pd.DataFrame,
    train_seasons: Iterable[str],
) -> tuple[dict, dict[str, int], list[str], dict[str, int], dict[str, tuple[float, float]]]:
    """Stan data for table_hierarchical.stan from historical tables + features."""
    ordered_seasons = list(train_seasons)
    season_to_idx = {s: i + 1 for i, s in enumerate(ordered_seasons)}

    train = tables[tables["season"].isin(ordered_seasons)].copy()
    teams = sorted(train["team"].unique())
    team_to_idx = {name: i + 1 for i, name in enumerate(teams)}

    train, feature_stats = _standardize_features(train)
    train["obs_pos"] = train.groupby("season").cumcount() + 1
    season_counts = train.groupby("season").size()
    if season_counts.nunique() != 1:
        raise ValueError("table_hierarchical.stan expects equal teams per season")

    stan_data = {
        "N": len(train),
        "S": len(ordered_seasons),
        "T": len(teams),
        "K": int(season_counts.iloc[0]),
        "nu": STUDENT_T_NU,
        "season": train["season"].map(season_to_idx).astype(int).to_numpy(),
        "obs_pos": train["obs_pos"].astype(int).to_numpy(),
        "team": train["team"].map(team_to_idx).astype(int).to_numpy(),
        "pts": train["Pts"].astype(float).to_numpy(),
        "sot_diff_pg": train["sot_diff_pg"].astype(float).to_numpy(),
        "pts_lag1": train["pts_lag1"].astype(float).to_numpy(),
        "ppg_last10": train["ppg_last10"].astype(float).to_numpy(),
    }
    return stan_data, team_to_idx, teams, season_to_idx, feature_stats


def ppc_table_replicates(
    fit,
    stan_data: dict,
    *,
    model: str = "static",
    nu: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Posterior predictive replicates of team-season points.

    Returns array with shape (n_draws, N).
    """
    if model not in {"static", "hierarchical"}:
        raise ValueError("model must be 'static' or 'hierarchical'")

    nu = STUDENT_T_NU if nu is None else nu
    rng = np.random.default_rng(seed)

    intercept = fit.stan_variable("intercept")
    beta_sot = fit.stan_variable("beta_sot")
    beta_lag = fit.stan_variable("beta_lag")
    beta_form = fit.stan_variable("beta_form")
    sigma_pts = fit.stan_variable("sigma_pts")
    skill = fit.stan_variable("skill")

    team_idx = np.asarray(stan_data["team"], dtype=int) - 1
    sot = np.asarray(stan_data["sot_diff_pg"], dtype=float)
    lag = np.asarray(stan_data["pts_lag1"], dtype=float)
    form = np.asarray(stan_data["ppg_last10"], dtype=float)

    if model == "static":
        beta_pts = fit.stan_variable("beta_pts")
        sk = beta_pts[:, None] * skill[:, team_idx]
    else:
        season_idx = np.asarray(stan_data["season"], dtype=int) - 1
        sk = skill[:, season_idx, team_idx]

    mu = (
        intercept[:, None]
        + sk
        + beta_sot[:, None] * sot[None, :]
        + beta_lag[:, None] * lag[None, :]
        + beta_form[:, None] * form[None, :]
    )
    noise = sigma_pts[:, None] * rng.standard_t(nu, size=mu.shape)
    return mu + noise


def predict_table(
    fit,
    teams: list[str],
    team_to_idx: dict[str, int],
    *,
    model: str = "static",
    last_season_index: int | None = None,
    team_features: dict[str, dict[str, float]] | None = None,
    n_sims: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Predict a full league table from a fitted **table-level** Stan model.

    Samples posterior predictive points per team, ranks into positions.
    Teams not in training index get skill = 0 (mid-table prior).
    Pass z-scored covariates via team_features (from build_forecast_features).

    Returns summary with pos_median, pts_median, etc.
    """
    if model not in {"static", "hierarchical"}:
        raise ValueError("model must be 'static' or 'hierarchical'")
    if model == "hierarchical" and last_season_index is None:
        raise ValueError("last_season_index required for hierarchical model")

    zero_feat = {"sot_diff_pg": 0.0, "pts_lag1": 0.0, "ppg_last10": 0.0}

    rng = np.random.default_rng(seed)
    intercept = fit.stan_variable("intercept")
    beta_sot = fit.stan_variable("beta_sot")
    beta_lag = fit.stan_variable("beta_lag")
    beta_form = fit.stan_variable("beta_form")
    sigma_pts = fit.stan_variable("sigma_pts")
    n_draws = intercept.shape[0]

    skill_draws = fit.stan_variable("skill")
    beta_pts_draws = (
        fit.stan_variable("beta_pts") if model == "static" else None
    )
    s_idx = (last_season_index - 1) if model == "hierarchical" else None

    position_records = {t: [] for t in teams}
    points_records = {t: [] for t in teams}

    for _ in range(n_sims):
        d = rng.integers(0, n_draws)
        sim_pts: dict[str, float] = {}
        for team in teams:
            if team in team_to_idx:
                j = team_to_idx[team] - 1
                if model == "static":
                    team_skill = beta_pts_draws[d] * skill_draws[d, j]
                else:
                    team_skill = skill_draws[d, s_idx, j]
            else:
                team_skill = 0.0

            feat = (team_features or {}).get(team, zero_feat)
            mu = (
                intercept[d]
                + team_skill
                + beta_sot[d] * feat["sot_diff_pg"]
                + beta_lag[d] * feat["pts_lag1"]
                + beta_form[d] * feat["ppg_last10"]
            )
            sim_pts[team] = mu + sigma_pts[d] * rng.standard_t(STUDENT_T_NU)

        ranked = (
            pd.DataFrame({"team": teams, "Pts": [sim_pts[t] for t in teams]})
            .sort_values(["Pts"], ascending=False)
        )
        ranked["position"] = np.arange(1, len(ranked) + 1)
        for _, row in ranked.iterrows():
            position_records[row["team"]].append(int(row["position"]))
            points_records[row["team"]].append(float(row["Pts"]))

    summary = pd.DataFrame({"team": teams})
    summary["pos_median"] = [np.median(position_records[t]) for t in teams]
    summary["pos_mean"] = [np.mean(position_records[t]) for t in teams]
    summary["pts_median"] = [np.median(points_records[t]) for t in teams]
    summary["pts_mean"] = [np.mean(points_records[t]) for t in teams]
    return summary.sort_values("pos_median").reset_index(drop=True)


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
