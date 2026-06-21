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
                "is_promoted": 0,
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
                    "is_promoted": 0,
                }
            else:
                feature_row = {
                    "season": season,
                    "team": team,
                    "sot_diff_pg": 0.0,
                    "pts_lag1": 0.0,
                    "ppg_last10": 0.0,
                    "is_promoted": 1,
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

    Teams absent in feature_season use the same raw-zero convention as promoted
    teams in training, then get z-scored with the training feature statistics.
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
                "is_promoted": 0,
            })
        else:
            raw_rows.append({
                "team": team,
                "sot_diff_pg": 0.0,
                "pts_lag1": 0.0,
                "ppg_last10": 0.0,
                "is_promoted": 1,
            })
    raw_df = pd.DataFrame(raw_rows)
    z_df, _ = _standardize_features(raw_df, stats=feature_stats)
    return {
        row["team"]: {
            "sot_diff_pg": float(row["sot_diff_pg"]),
            "pts_lag1": float(row["pts_lag1"]),
            "ppg_last10": float(row["ppg_last10"]),
            "is_promoted": float(row["is_promoted"]),
        }
        for _, row in z_df.iterrows()
    }


def prepare_table_stan_static(
    tables: pd.DataFrame,
    train_seasons: Iterable[str],
) -> tuple[dict, dict[str, int], list[str], dict[str, tuple[float, float]]]:
    """Stan data for team_static.stan from historical tables + features."""
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
        "is_promoted": train["is_promoted"].astype(int).to_numpy(),
    }
    return stan_data, team_to_idx, teams, feature_stats


def prepare_table_stan_hierarchical(
    tables: pd.DataFrame,
    train_seasons: Iterable[str],
) -> tuple[dict, dict[str, int], list[str], dict[str, int], dict[str, tuple[float, float]]]:
    """Stan data for team_hierarchical.stan from historical tables.

    Model 2 intentionally excludes process/lag covariates from its likelihood;
    they are still standardized and returned in ``feature_stats`` only because
    shared forecasting helpers use the same feature-building path as Model 1.
    """
    ordered_seasons = list(train_seasons)
    season_to_idx = {s: i + 1 for i, s in enumerate(ordered_seasons)}

    train = tables[tables["season"].isin(ordered_seasons)].copy()
    teams = sorted(train["team"].unique())
    team_to_idx = {name: i + 1 for i, name in enumerate(teams)}

    train, feature_stats = _standardize_features(train)

    stan_data = {
        "N": len(train),
        "S": len(ordered_seasons),
        "T": len(teams),
        "nu": STUDENT_T_NU,
        "season": train["season"].map(season_to_idx).astype(int).to_numpy(),
        "team": train["team"].map(team_to_idx).astype(int).to_numpy(),
        "pts": train["Pts"].astype(float).to_numpy(),
        "is_promoted": train["is_promoted"].astype(int).to_numpy(),
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
    beta_sot = fit.stan_variable("beta_sot") if model == "static" else None
    beta_lag = fit.stan_variable("beta_lag") if model == "static" else None
    beta_form = fit.stan_variable("beta_form") if model == "static" else None
    beta_promoted = fit.stan_variable("beta_promoted")
    sigma_pts = fit.stan_variable("sigma_pts")
    skill = fit.stan_variable("skill")

    team_idx = np.asarray(stan_data["team"], dtype=int) - 1
    promoted = np.asarray(stan_data["is_promoted"], dtype=float)

    if model == "static":
        beta_pts = fit.stan_variable("beta_pts")
        sot = np.asarray(stan_data["sot_diff_pg"], dtype=float)
        lag = np.asarray(stan_data["pts_lag1"], dtype=float)
        form = np.asarray(stan_data["ppg_last10"], dtype=float)
        sk = beta_pts[:, None] * skill[:, team_idx]
    else:
        season_idx = np.asarray(stan_data["season"], dtype=int) - 1
        sk = skill[:, season_idx, team_idx]

    mu = intercept[:, None] + sk + beta_promoted[:, None] * promoted[None, :]
    if model == "static":
        mu = (
            mu
            + beta_sot[:, None] * sot[None, :]
            + beta_lag[:, None] * lag[None, :]
            + beta_form[:, None] * form[None, :]
        )
    noise = sigma_pts[:, None] * rng.standard_t(nu, size=mu.shape)
    return mu + noise


def predict_team_points(
    fit,
    team: str,
    team_to_idx: dict[str, int],
    *,
    model: str = "static",
    team_features: dict[str, float] | None = None,
    n_sims: int = 500,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """
    Posterior predictive points for **one team** (model input = single club).

    Model 1 (static): ``beta_pts * skill[team]`` — one latent strength over all seasons.

    Model 2 (hierarchical): long-run ``team_skill[team]`` plus a fresh
    season-level effect ``tau_season * z`` and promoted-team effect. It does
    not use process/lag covariates.

    Returns summary stats and optional replicate array.
    """
    if model not in {"static", "hierarchical"}:
        raise ValueError("model must be 'static' or 'hierarchical'")

    zero_feat = {
        "sot_diff_pg": 0.0,
        "pts_lag1": 0.0,
        "ppg_last10": 0.0,
        "is_promoted": 0.0,
    }
    feat = team_features or zero_feat

    rng = np.random.default_rng(seed)
    intercept = fit.stan_variable("intercept")
    beta_sot = fit.stan_variable("beta_sot") if model == "static" else None
    beta_lag = fit.stan_variable("beta_lag") if model == "static" else None
    beta_form = fit.stan_variable("beta_form") if model == "static" else None
    beta_promoted = fit.stan_variable("beta_promoted")
    sigma_pts = fit.stan_variable("sigma_pts")
    n_draws = intercept.shape[0]

    beta_pts_draws = fit.stan_variable("beta_pts") if model == "static" else None
    static_skill_draws = (
        fit.stan_variable("skill") if model == "static" else None
    )
    team_skill_draws = (
        fit.stan_variable("team_skill") if model == "hierarchical" else None
    )
    tau_season_draws = (
        fit.stan_variable("tau_season") if model == "hierarchical" else None
    )

    sims = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        d = rng.integers(0, n_draws)
        if team in team_to_idx:
            j = team_to_idx[team] - 1
            if model == "static":
                team_skill = beta_pts_draws[d] * static_skill_draws[d, j]
            else:
                season_effect = tau_season_draws[d] * rng.standard_normal()
                team_skill = team_skill_draws[d, j] + season_effect
        else:
            if model == "hierarchical":
                season_effect = tau_season_draws[d] * rng.standard_normal()
                team_skill = season_effect
            else:
                team_skill = 0.0

        mu = intercept[d] + team_skill + beta_promoted[d] * feat.get("is_promoted", 0.0)
        if model == "static":
            mu = (
                mu
                + beta_sot[d] * feat["sot_diff_pg"]
                + beta_lag[d] * feat["pts_lag1"]
                + beta_form[d] * feat["ppg_last10"]
            )
        sims[i] = mu + sigma_pts[d] * rng.standard_t(STUDENT_T_NU)

    return {
        "team": team,
        "pts_mean": float(np.mean(sims)),
        "pts_median": float(np.median(sims)),
        "pts_q05": float(np.quantile(sims, 0.05)),
        "pts_q95": float(np.quantile(sims, 0.95)),
        "pts_sims": sims,
    }


def build_predicted_table(
    fit,
    teams: list[str],
    team_to_idx: dict[str, int],
    *,
    model: str = "static",
    team_features: dict[str, dict[str, float]] | None = None,
    n_sims: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a league table by predicting points for each team separately, then ranking.

    Each team is an independent model input; position comes from sorting predicted points.
    Model 1 uses static ``beta_pts * skill`` plus process/lag covariates.
    Model 2 uses long-run hierarchical ``team_skill``, a fresh common
    season-level effect for the target season, and promoted status only. Both
    also include Student-t predictive residual noise.
    """
    if model == "hierarchical":
        rng = np.random.default_rng(seed)
        intercept = fit.stan_variable("intercept")
        beta_promoted = fit.stan_variable("beta_promoted")
        sigma_pts = fit.stan_variable("sigma_pts")
        team_skill_draws = fit.stan_variable("team_skill")
        tau_season_draws = fit.stan_variable("tau_season")
        n_draws = intercept.shape[0]

        draw_idx = rng.integers(0, n_draws, size=n_sims)
        season_effect = tau_season_draws[draw_idx] * rng.standard_normal(n_sims)

        rows = []
        for team in teams:
            feat = (team_features or {}).get(team, {})
            promoted = float(feat.get("is_promoted", 0.0))
            if team in team_to_idx:
                j = team_to_idx[team] - 1
                skill = team_skill_draws[draw_idx, j]
            else:
                skill = 0.0

            sims = (
                intercept[draw_idx]
                + skill
                + season_effect
                + beta_promoted[draw_idx] * promoted
                + sigma_pts[draw_idx] * rng.standard_t(STUDENT_T_NU, size=n_sims)
            )
            rows.append({
                "team": team,
                "pts_median": float(np.median(sims)),
                "pts_mean": float(np.mean(sims)),
                "pts_q05": float(np.quantile(sims, 0.05)),
                "pts_q95": float(np.quantile(sims, 0.95)),
            })

        summary = pd.DataFrame(rows).sort_values(
            ["pts_median", "pts_mean"], ascending=False
        )
        summary["pos_median"] = np.arange(1, len(summary) + 1)
        summary["pos_mean"] = summary["pts_mean"].rank(ascending=False, method="average")
        return summary.reset_index(drop=True)

    rows = []
    for i, team in enumerate(teams):
        feat = (team_features or {}).get(team)
        pred = predict_team_points(
            fit,
            team,
            team_to_idx,
            model=model,
            team_features=feat,
            n_sims=n_sims,
            seed=seed + i,
        )
        rows.append({
            "team": team,
            "pts_median": pred["pts_median"],
            "pts_mean": pred["pts_mean"],
            "pts_q05": pred["pts_q05"],
            "pts_q95": pred["pts_q95"],
        })

    summary = pd.DataFrame(rows).sort_values(
        ["pts_median", "pts_mean"], ascending=False
    )
    summary["pos_median"] = np.arange(1, len(summary) + 1)
    summary["pos_mean"] = summary["pts_mean"].rank(ascending=False, method="average")
    return summary.reset_index(drop=True)


def compare_forecast_to_actual(
    pred_table: pd.DataFrame,
    matches: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Merge predicted league table with actual season results.

    Adds `pos_error` and `pts_error` as predicted minus actual
    (negative = model under-predicted).
    """
    actual = compute_table(matches, season)[["team", "position", "Pts"]]
    actual = actual.rename(columns={"position": "pos_actual", "Pts": "pts_actual"})
    comparison = pred_table.merge(actual, on="team", how="left")
    comparison["pos_error"] = comparison["pos_median"] - comparison["pos_actual"]
    comparison["pts_error"] = comparison["pts_median"] - comparison["pts_actual"]
    comparison["pos_abs_error"] = comparison["pos_error"].abs()
    comparison["pts_abs_error"] = comparison["pts_error"].abs()
    return comparison


def forecast_team_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per-team point/position errors, sorted by absolute point error."""
    cols = [
        "team",
        "pts_median",
        "pts_actual",
        "pts_error",
        "pts_abs_error",
        "pos_median",
        "pos_actual",
        "pos_error",
        "pos_abs_error",
    ]
    if "test_season" in comparison.columns:
        cols = ["test_season"] + cols
    return comparison[cols].sort_values("pts_abs_error", ascending=False)


def forecast_season_summary(comparison: pd.DataFrame) -> pd.Series:
    """
    Season-level backtest metrics from `compare_forecast_to_actual` output.

    Returns mean predicted/actual points, MAE, and total absolute error.
    """
    return pd.Series(
        {
            "n_teams": len(comparison),
            "pts_pred_mean": comparison["pts_median"].mean(),
            "pts_actual_mean": comparison["pts_actual"].mean(),
            "pts_mae": comparison["pts_abs_error"].mean(),
            "pts_abs_error_sum": comparison["pts_abs_error"].sum(),
            "pts_bias": comparison["pts_error"].mean(),
            "pos_mae": comparison["pos_abs_error"].mean(),
            "pos_abs_error_sum": comparison["pos_abs_error"].sum(),
            "pos_bias": comparison["pos_error"].mean(),
        }
    )


def print_forecast_season_summary(
    summary: pd.Series,
    *,
    season: str | None = None,
    title: str = "Point forecast quality",
) -> None:
    """Print season totals and mean absolute error per team (sum / n_teams)."""
    n_teams = int(summary["n_teams"])
    pts_sum = float(summary["pts_abs_error_sum"])
    pts_mean = pts_sum / n_teams
    pos_sum = float(summary["pos_abs_error_sum"])
    pos_mean = pos_sum / n_teams
    season_label = f"Season {season} — " if season else ""

    print(f"{season_label}{title}")
    print(f"  Sum |point error| over all teams:  {pts_sum:.0f} pts")
    print(f"  Mean |point error| per team:        {pts_mean:.2f} pts  ({pts_sum:.0f} / {n_teams})")
    print(f"  Mean signed point error (bias):     {summary['pts_bias']:+.2f} pts")
    print(f"  Mean predicted / actual points:     {summary['pts_pred_mean']:.1f} / {summary['pts_actual_mean']:.1f}")
    print(f"  Sum |position error|:               {pos_sum:.0f} places")
    print(f"  Mean |position error| per team:     {pos_mean:.2f} places  ({pos_sum:.0f} / {n_teams})")


def plot_forecast_team_errors(
    comparison: pd.DataFrame,
    *,
    season: str | None = None,
    title: str | None = None,
    ax=None,
):
    """
    Bar chart of signed and absolute point errors per team.

    Returns (fig, axes). Red = model over-predicted; blue = under-predicted.
    """
    import matplotlib.pyplot as plt

    data = forecast_team_errors(comparison).sort_values("pts_abs_error", ascending=True)
    season_label = season or (
        str(comparison["test_season"].iloc[0])
        if "test_season" in comparison.columns
        else ""
    )
    plot_title = title or f"Point forecast errors{f' — {season_label}' if season_label else ''}"

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(data))))
    else:
        fig = ax.figure
        axes = [ax] if not hasattr(ax, "__len__") else ax

    colors = np.where(data["pts_error"] >= 0, "tomato", "steelblue")
    axes[0].barh(data["team"], data["pts_error"], color=colors, alpha=0.85)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Point error (predicted − actual)")
    axes[0].set_title("Signed error per team")

    axes[1].barh(data["team"], data["pts_abs_error"], color="gray", alpha=0.75)
    axes[1].set_xlabel("|Point error|")
    axes[1].set_title("Absolute error per team")

    summary = forecast_season_summary(comparison)
    n_teams = int(summary["n_teams"])
    pts_sum = float(summary["pts_abs_error_sum"])
    pts_mean = pts_sum / n_teams
    fig.suptitle(
        f"{plot_title}\n"
        f"sum |error| = {pts_sum:.0f} pkt | "
        f"mean per team = {pts_mean:.2f} pkt ({pts_sum:.0f} / {n_teams})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return fig, axes


def summarize_models(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Overall table metrics for one or more fitted models."""
    return pd.DataFrame(
        {name: forecast_season_summary(cmp) for name, cmp in comparisons.items()}
    )


def compare_models_team_errors(
    comparison_a: pd.DataFrame,
    comparison_b: pd.DataFrame,
    *,
    label_a: str = "m1",
    label_b: str = "m2",
) -> pd.DataFrame:
    """Side-by-side per-team absolute point errors for two models."""
    a = forecast_team_errors(comparison_a)[
        ["team", "pts_error", "pts_abs_error", "pos_abs_error"]
    ].rename(
        columns={
            "pts_error": f"pts_error_{label_a}",
            "pts_abs_error": f"pts_abs_error_{label_a}",
            "pos_abs_error": f"pos_abs_error_{label_a}",
        }
    )
    b = forecast_team_errors(comparison_b)[
        ["team", "pts_error", "pts_abs_error", "pos_abs_error"]
    ].rename(
        columns={
            "pts_error": f"pts_error_{label_b}",
            "pts_abs_error": f"pts_abs_error_{label_b}",
            "pos_abs_error": f"pos_abs_error_{label_b}",
        }
    )
    merged = a.merge(b, on="team", how="outer")
    merged[f"pts_abs_error_diff_{label_a}_minus_{label_b}"] = (
        merged[f"pts_abs_error_{label_a}"] - merged[f"pts_abs_error_{label_b}"]
    )
    return merged.sort_values(f"pts_abs_error_{label_a}", ascending=False)


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
