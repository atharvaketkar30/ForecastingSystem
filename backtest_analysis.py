"""
Systematic backtesting and error analysis for the hierarchical forecasting setup.

What this script does:
1. Runs rolling-origin backtests across multiple cutoffs.
2. Evaluates each forecast method by series, hierarchy level, and horizon step.
3. Extracts diagnostic views to show where the current methodology underperforms.
4. Writes JSON outputs plus a markdown summary with actionable findings.

Run with:
MPLCONFIGDIR=/tmp/matplotlib Forecasting/bin/python backtest_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from basic_forecasts import (
    FORECASTS_DIR,
    HORIZON,
    ROOT_ID,
    TopDownStaticProportions,
    load_series,
    reconcile_forecasts,
)


ANALYSIS_DIR = FORECASTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

INTERVENTIONS_PATH = Path("interventions.json")
DEFAULT_HORIZON = HORIZON
DEFAULT_N_WINDOWS = 6
DEFAULT_STEP_DAYS = 14


def infer_level(unique_id: str) -> str:
    depth = unique_id.count("/")
    if depth == 0:
        return "total"
    if depth == 1:
        return "project"
    if depth == 2:
        return "category"
    return "article"


def make_cutoffs(df: pd.DataFrame, horizon: int, n_windows: int, step_days: int) -> list[pd.Timestamp]:
    all_dates = sorted(pd.to_datetime(df["ds"].drop_duplicates()))
    max_cutoff_idx = len(all_dates) - horizon - 1
    if max_cutoff_idx < 60:
        raise ValueError("Not enough history to run the requested backtest windows.")

    cutoffs: list[pd.Timestamp] = []
    idx = max_cutoff_idx
    while idx >= 30 and len(cutoffs) < n_windows:
        cutoffs.append(pd.Timestamp(all_dates[idx]))
        idx -= step_days

    return sorted(cutoffs)


def _stack_reconciled_forecasts(reconciled: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in reconciled.columns if c not in {"unique_id", "ds"}]
    stacked = reconciled.melt(
        id_vars=["unique_id", "ds"],
        value_vars=value_cols,
        var_name="model",
        value_name="y_hat",
    )
    model_map = {
        "baseline": "seasonal_naive_base",
        "baseline/BottomUp": "bottom_up",
        "baseline/TopDown_method-proportion_averages": "reconciled_top_down",
        "baseline/MinTrace_method-ols": "mint_ols",
    }
    stacked["model"] = stacked["model"].map(model_map).fillna(stacked["model"])
    return stacked


def _topdown_forecasts(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    fcst = TopDownStaticProportions(horizon=horizon).fit(train).predict().copy()
    fcst["model"] = "static_top_down"
    fcst = fcst.rename(columns={"y_hat": "y_hat"})
    return fcst[["unique_id", "ds", "model", "y_hat"]]


def generate_forecasts_for_cutoff(
    full_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = full_df[full_df["ds"] <= cutoff].copy()
    future_dates = sorted(full_df.loc[full_df["ds"] > cutoff, "ds"].drop_duplicates())[:horizon]
    test = full_df[full_df["ds"].isin(future_dates)].copy()
    if test.empty:
        raise ValueError(f"No future observations available after cutoff {cutoff}.")

    topdown = _topdown_forecasts(train=train, horizon=len(future_dates))
    reconciled = reconcile_forecasts(train=train, S_df=S_df, tags=tags, horizon=len(future_dates))
    reconciled_long = _stack_reconciled_forecasts(reconciled)

    forecasts = pd.concat([topdown, reconciled_long], ignore_index=True)
    forecasts["cutoff"] = pd.Timestamp(cutoff)
    return forecasts, test


def load_interventions() -> list[dict]:
    if not INTERVENTIONS_PATH.exists():
        return []
    return json.loads(INTERVENTIONS_PATH.read_text())


def add_context_columns(df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["level"] = out["unique_id"].map(infer_level)
    out["horizon_step"] = out.groupby(["cutoff", "unique_id", "model"]).cumcount() + 1
    out["weekday"] = pd.to_datetime(out["ds"]).dt.day_name()

    history = full_df.sort_values(["unique_id", "ds"]).copy()
    history["lag_7"] = history.groupby("unique_id")["y"].shift(7)
    history["rollmean_7"] = history.groupby("unique_id")["y"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    history["pct_vs_lag7"] = np.where(history["lag_7"] > 0, history["y"] / history["lag_7"] - 1, np.nan)
    history["pct_vs_roll7"] = np.where(history["rollmean_7"] > 0, history["y"] / history["rollmean_7"] - 1, np.nan)
    history = history[["unique_id", "ds", "lag_7", "rollmean_7", "pct_vs_lag7", "pct_vs_roll7"]]

    out = out.merge(history, on=["unique_id", "ds"], how="left")
    out["regime"] = np.select(
        [
            out["pct_vs_roll7"] >= 0.25,
            out["pct_vs_roll7"] <= -0.25,
        ],
        [
            "spike",
            "drop",
        ],
        default="stable",
    )

    total_actual = (
        full_df[full_df["unique_id"] == ROOT_ID][["ds", "y"]]
        .rename(columns={"y": "network_actual"})
        .sort_values("ds")
    )
    total_actual["volume_bucket"] = pd.qcut(
        total_actual["network_actual"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    )
    out = out.merge(total_actual[["ds", "volume_bucket"]], on="ds", how="left")
    return out


def attach_intervention_flags(df: pd.DataFrame) -> pd.DataFrame:
    interventions = load_interventions()
    if not interventions:
        df["intervention_flag"] = "none"
        return df

    out = df.copy()
    out["intervention_flag"] = "none"
    ds = pd.to_datetime(out["ds"])

    for iv in interventions:
        iv_date = pd.to_datetime(iv["date"])
        if iv["type"] == "temporary_suppression":
            end_date = iv_date + pd.Timedelta(days=iv["duration_days"])
            mask = ds.between(iv_date, end_date)
        else:
            mask = ds >= iv_date

        for project in iv["affected_projects"]:
            project_mask = out["unique_id"].str.contains(project, regex=False)
            for category in iv["affected_categories"]:
                scoped = project_mask & out["unique_id"].str.contains(category, regex=False)
                out.loc[mask & scoped, "intervention_flag"] = iv["id"]

    return out


def compute_errors(forecasts: pd.DataFrame, actuals: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    merged = forecasts.merge(actuals, on=["unique_id", "ds"], how="inner")
    merged = add_context_columns(merged, full_df=full_df)
    merged = attach_intervention_flags(merged)

    merged["error"] = merged["y_hat"] - merged["y"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"].pow(2)
    merged["ape"] = np.where(merged["y"] > 0, merged["abs_error"] / merged["y"], np.nan)
    merged["bias_pct"] = np.where(merged["y"] > 0, merged["error"] / merged["y"], np.nan)
    return merged


def summarize_errors(errors: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        errors.groupby(group_cols, as_index=False)
        .agg(
            n=("y", "size"),
            actual_sum=("y", "sum"),
            forecast_sum=("y_hat", "sum"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            mape=("ape", "mean"),
            bias=("error", "mean"),
            abs_error_sum=("abs_error", "sum"),
            over_forecast_rate=("error", lambda s: float(np.mean(s > 0))),
        )
    )
    summary["wape"] = np.where(summary["actual_sum"] > 0, summary["abs_error_sum"] / summary["actual_sum"], np.nan)
    summary["bias_pct_of_actual"] = np.where(
        summary["actual_sum"] > 0,
        (summary["forecast_sum"] - summary["actual_sum"]) / summary["actual_sum"],
        np.nan,
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def best_model_table(summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    idx = summary.groupby(group_col)["wape"].idxmin()
    return summary.loc[idx].sort_values(group_col).reset_index(drop=True)


def build_insight_report(errors: pd.DataFrame) -> str:
    overall = summarize_errors(errors, ["model"])
    by_level = summarize_errors(errors, ["model", "level"])
    by_horizon = summarize_errors(errors, ["model", "horizon_step"])
    by_regime = summarize_errors(errors, ["model", "regime"])
    by_intervention = summarize_errors(errors, ["model", "intervention_flag"])
    by_weekday = summarize_errors(errors, ["model", "weekday"])
    by_series = summarize_errors(errors, ["model", "unique_id", "level"])

    overall_best = overall.sort_values("wape").iloc[0]
    worst_articles = (
        by_series[by_series["level"] == "article"]
        .sort_values(["wape", "mae"], ascending=[False, False])
        .head(8)
    )
    late_horizon = by_horizon[by_horizon["horizon_step"].between(15, 28)]
    early_horizon = by_horizon[by_horizon["horizon_step"].between(1, 7)]
    horizon_delta = (
        late_horizon.groupby("model", as_index=False)["wape"].mean()
        .merge(early_horizon.groupby("model", as_index=False)["wape"].mean(), on="model", suffixes=("_late", "_early"))
    )
    horizon_delta["wape_drift"] = horizon_delta["wape_late"] - horizon_delta["wape_early"]
    horizon_delta = horizon_delta.sort_values("wape_drift", ascending=False)

    stable_vs_spike = by_regime.pivot(index="model", columns="regime", values="wape").reset_index()
    if {"stable", "spike"}.issubset(stable_vs_spike.columns):
        stable_vs_spike["spike_penalty"] = stable_vs_spike["spike"] - stable_vs_spike["stable"]
        stable_vs_spike = stable_vs_spike.sort_values("spike_penalty", ascending=False)

    lines: list[str] = []
    lines.append("# Backtest Error Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append(
        f"- Best overall WAPE: `{overall_best['model']}` at `{overall_best['wape']:.3f}` across `{int(overall_best['n'])}` forecast points."
    )
    for _, row in overall.sort_values("wape").iterrows():
        lines.append(
            f"- `{row['model']}`: WAPE `{row['wape']:.3f}`, MAE `{row['mae']:.1f}`, bias_pct `{row['bias_pct_of_actual']:.3f}`."
        )

    lines.append("")
    lines.append("## Where The Methodology Misses")

    score_spread = overall["wape"].max() - overall["wape"].min()
    near_ties = overall.groupby(["wape", "mae"]).size().reset_index(name="n_models")
    if (near_ties["n_models"] > 1).any():
        tied = int(near_ties["n_models"].max())
        lines.append(
            f"- At least `{tied}` methods are effectively tied on aggregate metrics. That usually means reconciliation is not adding value because the base forecasts are already coherent or too simplistic to differentiate downstream methods."
        )
    elif score_spread < 0.01:
        lines.append(
            "- The methods are very tightly clustered on WAPE, which suggests model class choice matters less right now than improving the underlying signal model."
        )

    level_best = best_model_table(by_level, "level")
    for _, row in level_best.iterrows():
        lines.append(f"- Best model at `{row['level']}` level: `{row['model']}` with WAPE `{row['wape']:.3f}`.")

    if not horizon_delta.empty:
        worst_drift = horizon_delta.iloc[0]
        lines.append(
            f"- Largest early-to-late horizon degradation: `{worst_drift['model']}` worsens by `{worst_drift['wape_drift']:.3f}` WAPE points from days 1-7 to days 15-28."
        )

    if not stable_vs_spike.empty and "spike_penalty" in stable_vs_spike.columns:
        spike_row = stable_vs_spike.iloc[0]
        lines.append(
            f"- Most sensitive to spikes: `{spike_row['model']}` with spike penalty `{spike_row['spike_penalty']:.3f}` relative to stable periods."
        )

    worst_bias = overall.iloc[overall["bias_pct_of_actual"].abs().argmax()]
    direction = "over-forecasting" if worst_bias["bias_pct_of_actual"] > 0 else "under-forecasting"
    lines.append(
        f"- Largest systematic bias: `{worst_bias['model']}` is `{direction}` by `{abs(worst_bias['bias_pct_of_actual']):.3f}` of actual volume."
    )

    lines.append("")
    lines.append("## Hardest Series")
    for _, row in worst_articles.iterrows():
        lines.append(
            f"- `{row['unique_id']}` with `{row['model']}`: WAPE `{row['wape']:.3f}`, MAE `{row['mae']:.1f}`."
        )

    lines.append("")
    lines.append("## Intervention And Regime Readout")
    top_regime = by_regime.sort_values(["regime", "wape"], ascending=[True, True]).groupby("regime").head(1)
    for _, row in top_regime.iterrows():
        lines.append(f"- Best model during `{row['regime']}` periods: `{row['model']}` with WAPE `{row['wape']:.3f}`.")

    active_interventions = by_intervention[by_intervention["intervention_flag"] != "none"].copy()
    if not active_interventions.empty:
        hardest_iv = active_interventions.sort_values("wape", ascending=False).iloc[0]
        easiest_iv = active_interventions.sort_values("wape", ascending=True).iloc[0]
        lines.append(
            f"- Hardest intervention segment: `{hardest_iv['intervention_flag']}` for `{hardest_iv['model']}` with WAPE `{hardest_iv['wape']:.3f}`."
        )
        lines.append(
            f"- Best intervention handling observed: `{easiest_iv['intervention_flag']}` for `{easiest_iv['model']}` with WAPE `{easiest_iv['wape']:.3f}`."
        )

    lines.append("")
    lines.append("## Recommended Next Directions")
    lines.append("- Add explicit event/intervention regressors into the forecast model rather than expecting seasonal-naive repeats to absorb structural shifts.")
    lines.append("- Compare weekly seasonality-only models against models with local trend, because large late-horizon drift usually signals missing trend or regime adaptation.")
    lines.append("- Evaluate model choice by hierarchy level instead of forcing a single method everywhere; top-down and reconciled methods can win on different slices.")
    lines.append("- Focus feature work first on the worst article-level series listed above, since they will likely produce the largest marginal gains.")

    return "\n".join(lines) + "\n"


def run_backtest(
    horizon: int = DEFAULT_HORIZON,
    n_windows: int = DEFAULT_N_WINDOWS,
    step_days: int = DEFAULT_STEP_DAYS,
) -> dict[str, pd.DataFrame | str]:
    full_df, S_df, tags = load_series()
    cutoffs = make_cutoffs(full_df, horizon=horizon, n_windows=n_windows, step_days=step_days)

    forecast_frames: list[pd.DataFrame] = []
    actual_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []

    for cutoff in cutoffs:
        forecasts, actuals = generate_forecasts_for_cutoff(
            full_df=full_df,
            S_df=S_df,
            tags=tags,
            cutoff=cutoff,
            horizon=horizon,
        )
        errors = compute_errors(forecasts, actuals, full_df=full_df)
        forecast_frames.append(forecasts)
        actual_frames.append(actuals.assign(cutoff=cutoff))
        error_frames.append(errors)

    all_forecasts = pd.concat(forecast_frames, ignore_index=True)
    all_actuals = pd.concat(actual_frames, ignore_index=True)
    all_errors = pd.concat(error_frames, ignore_index=True)

    summaries = {
        "overall": summarize_errors(all_errors, ["model"]),
        "by_level": summarize_errors(all_errors, ["model", "level"]),
        "by_horizon": summarize_errors(all_errors, ["model", "horizon_step"]),
        "by_regime": summarize_errors(all_errors, ["model", "regime"]),
        "by_weekday": summarize_errors(all_errors, ["model", "weekday"]),
        "by_intervention": summarize_errors(all_errors, ["model", "intervention_flag"]),
        "by_series": summarize_errors(all_errors, ["model", "unique_id", "level"]),
        "by_cutoff": summarize_errors(all_errors, ["model", "cutoff"]),
        "forecasts": all_forecasts,
        "actuals": all_actuals,
        "errors": all_errors,
        "report": build_insight_report(all_errors),
    }
    return summaries


def write_outputs(results: dict[str, pd.DataFrame | str]) -> None:
    for name, obj in results.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_json(
                ANALYSIS_DIR / f"{name}.json",
                orient="records",
                indent=2,
                date_format="iso",
            )
        else:
            (ANALYSIS_DIR / "insight_report.md").write_text(obj)


def main() -> None:
    results = run_backtest()
    write_outputs(results)

    print("Saved analysis outputs to", ANALYSIS_DIR.resolve())
    print("\nOverall summary:")
    print(results["overall"].to_string(index=False))
    print("\nTop findings:")
    report_lines = str(results["report"]).splitlines()
    for line in report_lines[3:10]:
        print(line)


if __name__ == "__main__":
    main()
