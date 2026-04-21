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
from html import escape
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
DASHBOARD_PATH = ANALYSIS_DIR / "dashboard.html"

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
    DASHBOARD_PATH.write_text(build_dashboard_html(results), encoding="utf-8")


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%d")
    frame = frame.replace({np.nan: None})
    return frame.to_dict(orient="records")


def _dashboard_payload(results: dict[str, pd.DataFrame | str]) -> dict:
    overall = results["overall"].copy()
    by_level = results["by_level"].copy()
    by_horizon = results["by_horizon"].copy()
    errors = results["errors"].copy()
    forecasts = results["forecasts"].copy()
    actuals = results["actuals"].copy()

    overall = overall.sort_values("wape").reset_index(drop=True)
    top_series = (
        results["by_series"]
        .copy()
        .sort_values(["wape", "mae"], ascending=[False, False])
        .head(20)[["model", "unique_id", "level", "wape", "mae", "bias_pct_of_actual"]]
        .reset_index(drop=True)
    )

    merged_trends = (
        forecasts.merge(actuals[["unique_id", "ds", "cutoff", "y"]], on=["unique_id", "ds", "cutoff"], how="inner")
        .sort_values(["unique_id", "model", "cutoff", "ds"])
        .reset_index(drop=True)
    )
    merged_trends["error"] = merged_trends["y_hat"] - merged_trends["y"]
    merged_trends["abs_error"] = merged_trends["error"].abs()
    merged_trends["level"] = merged_trends["unique_id"].map(infer_level)

    all_series = sorted(errors["unique_id"].dropna().unique().tolist())
    level_nodes = ["total", "project", "category", "article"]
    series_by_level = {
        level: sorted(
            merged_trends.loc[merged_trends["level"] == level, "unique_id"].dropna().unique().tolist()
        )
        for level in level_nodes
    }

    model_options = sorted(errors["model"].dropna().unique().tolist())
    cutoff_options = sorted(pd.to_datetime(errors["cutoff"]).dt.strftime("%Y-%m-%d").unique().tolist())

    best_row = overall.iloc[0]
    default_level = "total" if ROOT_ID in all_series else next(
        (level for level in level_nodes if series_by_level[level]),
        level_nodes[0],
    )
    default_series = ROOT_ID if ROOT_ID in series_by_level.get(default_level, []) else series_by_level[default_level][0]
    report_lines = [line for line in str(results["report"]).splitlines() if line.strip()]

    return {
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "best_model": best_row["model"],
            "best_wape": round(float(best_row["wape"]), 4),
            "forecast_points": int(best_row["n"]),
            "models_compared": int(overall["model"].nunique()),
            "series_count": int(errors["unique_id"].nunique()),
            "cutoff_count": int(errors["cutoff"].nunique()),
        },
        "filters": {
            "levels": level_nodes,
            "series_by_level": series_by_level,
            "models": model_options,
            "cutoffs": cutoff_options,
            "default_level": default_level,
            "default_series": default_series,
            "default_model": best_row["model"],
            "default_cutoff": cutoff_options[-1],
        },
        "tables": {
            "overall": _json_safe_records(overall[["model", "wape", "mae", "rmse", "bias_pct_of_actual", "over_forecast_rate", "n"]]),
            "hardest_series": _json_safe_records(top_series),
        },
        "charts": {
            "by_level": _json_safe_records(by_level[["model", "level", "wape", "mae"]].sort_values(["level", "wape"])),
            "by_horizon": _json_safe_records(by_horizon[["model", "horizon_step", "wape", "mae"]].sort_values(["model", "horizon_step"])),
            "trends": _json_safe_records(merged_trends[["unique_id", "level", "model", "cutoff", "ds", "y", "y_hat", "error", "abs_error"]]),
            "errors_daily": _json_safe_records(
                errors.groupby(["model", "ds"], as_index=False)
                .agg(mae=("abs_error", "mean"), bias=("error", "mean"), wape=("abs_error", "sum"))
                .sort_values(["model", "ds"])
            ),
        },
        "report_lines": report_lines,
    }


def build_dashboard_html(results: dict[str, pd.DataFrame | str]) -> str:
    dashboard_payload = _dashboard_payload(results)
    payload = json.dumps(dashboard_payload)
    report_html = "\n".join(
        f"<li>{escape(line.lstrip('- ').strip())}</li>"
        for line in dashboard_payload["report_lines"][1:]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Forecast Backtest Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.88);
      --panel-strong: #fffdf9;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: rgba(31, 41, 51, 0.12);
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --accent-2: #c2410c;
      --accent-3: #1d4ed8;
      --good: #15803d;
      --bad: #b91c1c;
      --shadow: 0 24px 50px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 65, 12, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 24%),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1400px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,250,242,0.92));
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 22px;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      color: var(--accent);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      font-weight: 700;
    }}
    .subtle {{
      color: var(--muted);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(10px);
    }}
    .kpi {{
      grid-column: span 3;
      min-height: 128px;
    }}
    .kpi-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .kpi-value {{
      margin-top: 12px;
      font-size: 2rem;
      font-weight: 700;
    }}
    .wide {{ grid-column: span 6; }}
    .full {{ grid-column: 1 / -1; }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: var(--panel-strong);
      color: var(--ink);
      font-size: 15px;
    }}
    .section-title {{
      margin: 0 0 14px;
      font-size: 1.2rem;
    }}
    .chart-wrap {{
      position: relative;
      width: 100%;
      min-height: 320px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 12px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .metric-good {{ color: var(--good); }}
    .metric-bad {{ color: var(--bad); }}
    ul {{
      margin: 0;
      padding-left: 18px;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
    }}
    .footnote {{
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    @media (max-width: 980px) {{
      .kpi, .wide {{ grid-column: span 12; }}
      .filters {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <p class="eyebrow">Interactive Forecast Review</p>
      <h1>Backtest dashboard for forecasts, trends, and error patterns.</h1>
      <p class="subtle" id="hero-summary"></p>
    </section>

    <section class="grid">
      <article class="card kpi">
        <div class="kpi-label">Best Model</div>
        <div class="kpi-value" id="best-model"></div>
        <div class="subtle" id="best-model-detail"></div>
      </article>
      <article class="card kpi">
        <div class="kpi-label">Forecast Points</div>
        <div class="kpi-value" id="forecast-points"></div>
        <div class="subtle">Evaluated across all rolling windows.</div>
      </article>
      <article class="card kpi">
        <div class="kpi-label">Series Covered</div>
        <div class="kpi-value" id="series-count"></div>
        <div class="subtle">Hierarchy nodes included in the backtest.</div>
      </article>
      <article class="card kpi">
        <div class="kpi-label">Cutoff Windows</div>
        <div class="kpi-value" id="cutoff-count"></div>
        <div class="subtle">Rolling-origin backtest windows analyzed.</div>
      </article>

      <article class="card full">
        <h2 class="section-title">Prediction Explorer</h2>
        <div class="filters">
          <div>
            <label for="level-select">Level</label>
            <select id="level-select"></select>
          </div>
          <div>
            <label for="series-select">Node</label>
            <select id="series-select"></select>
          </div>
          <div>
            <label for="model-select">Model</label>
            <select id="model-select"></select>
          </div>
          <div>
            <label for="cutoff-select">Cutoff</label>
            <select id="cutoff-select"></select>
          </div>
        </div>
        <div class="chart-wrap"><svg id="trend-chart" viewBox="0 0 860 320" preserveAspectRatio="none"></svg></div>
        <div class="legend" id="trend-legend"></div>
        <div class="footnote" id="trend-summary"></div>
      </article>

      <article class="card wide">
        <h2 class="section-title">WAPE By Horizon</h2>
        <div class="chart-wrap"><svg id="horizon-chart" viewBox="0 0 860 320" preserveAspectRatio="none"></svg></div>
        <div class="legend" id="horizon-legend"></div>
      </article>

      <article class="card wide">
        <h2 class="section-title">WAPE By Hierarchy Level</h2>
        <div class="chart-wrap"><svg id="level-chart" viewBox="0 0 860 320" preserveAspectRatio="none"></svg></div>
        <div class="legend" id="level-legend"></div>
      </article>

      <article class="card wide">
        <h2 class="section-title">Daily Error Pattern</h2>
        <div class="chart-wrap"><svg id="error-chart" viewBox="0 0 860 320" preserveAspectRatio="none"></svg></div>
        <div class="legend" id="error-legend"></div>
        <div class="footnote">Bias above zero means over-forecasting on average. MAE stays positive and highlights volatility even when bias nets out.</div>
      </article>

      <article class="card wide">
        <h2 class="section-title">Overall Metrics</h2>
        <table id="overall-table"></table>
      </article>

      <article class="card wide">
        <h2 class="section-title">Hardest Series</h2>
        <table id="series-table"></table>
      </article>

      <article class="card wide">
        <h2 class="section-title">Key Findings</h2>
        <ul>
          {report_html}
        </ul>
        <div class="footnote" id="generated-at"></div>
      </article>
    </section>
  </div>

  <script>
    const payload = {payload};

    const palette = ['#0f766e', '#c2410c', '#1d4ed8', '#b45309', '#7c3aed', '#15803d', '#be123c'];
    const levelColors = {{ total: '#0f766e', project: '#1d4ed8', category: '#c2410c', article: '#7c3aed' }};

    function formatNumber(value, digits = 0) {{
      return new Intl.NumberFormat('en-US', {{
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      }}).format(value);
    }}

    function formatPct(value) {{
      return `${{(value * 100).toFixed(1)}}%`;
    }}

    function clearSvg(svg) {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }}

    function linePath(points, xScale, yScale) {{
      return points.map((point, index) => `${{index === 0 ? 'M' : 'L'}} ${{xScale(point.x)}} ${{yScale(point.y)}}`).join(' ');
    }}

    function appendText(svg, text, x, y, options = {{}}) {{
      const node = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      node.textContent = text;
      node.setAttribute('x', x);
      node.setAttribute('y', y);
      node.setAttribute('fill', options.fill || '#5f6c7b');
      node.setAttribute('font-size', options.size || '12');
      node.setAttribute('font-family', '"Helvetica Neue", Helvetica, Arial, sans-serif');
      if (options.anchor) node.setAttribute('text-anchor', options.anchor);
      svg.appendChild(node);
      return node;
    }}

    function drawAxes(svg, width, height, margin, yTicks, xTicks, xFormatter, yFormatter) {{
      const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      svg.appendChild(grid);
      yTicks.forEach((tick) => {{
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', margin.left);
        line.setAttribute('x2', width - margin.right);
        line.setAttribute('y1', tick.position);
        line.setAttribute('y2', tick.position);
        line.setAttribute('stroke', 'rgba(31, 41, 51, 0.12)');
        grid.appendChild(line);
        appendText(svg, yFormatter(tick.value), margin.left - 10, tick.position + 4, {{ anchor: 'end' }});
      }});
      xTicks.forEach((tick) => {{
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', tick.position);
        line.setAttribute('x2', tick.position);
        line.setAttribute('y1', height - margin.bottom);
        line.setAttribute('y2', height - margin.bottom + 6);
        line.setAttribute('stroke', 'rgba(31, 41, 51, 0.22)');
        grid.appendChild(line);
        appendText(svg, xFormatter(tick.value), tick.position, height - margin.bottom + 22, {{ anchor: 'middle' }});
      }});
    }}

    function renderLegend(node, items) {{
      node.innerHTML = '';
      items.forEach((item) => {{
        const wrap = document.createElement('span');
        wrap.innerHTML = `<i class="swatch" style="background:${{item.color}}"></i>${{item.label}}`;
        node.appendChild(wrap);
      }});
    }}

    function renderMultiLineChart(svgId, legendId, series, opts) {{
      const svg = document.getElementById(svgId);
      const legend = document.getElementById(legendId);
      clearSvg(svg);

      const width = 860;
      const height = 320;
      const margin = {{ top: 18, right: 20, bottom: 40, left: 58 }};
      const xValues = Array.from(new Set(series.flatMap((line) => line.values.map((p) => p.x)))).sort((a, b) => a - b);
      const yValues = series.flatMap((line) => line.values.map((p) => p.y));
      const yMax = Math.max(...yValues, 0.0001);
      const yMin = opts.allowNegative ? Math.min(...yValues, 0) : 0;

      const xScale = (value) => margin.left + ((value - xValues[0]) / Math.max(xValues[xValues.length - 1] - xValues[0], 1)) * (width - margin.left - margin.right);
      const yScale = (value) => height - margin.bottom - ((value - yMin) / Math.max(yMax - yMin, 0.0001)) * (height - margin.top - margin.bottom);

      const yTicks = Array.from({{ length: 5 }}, (_, i) => {{
        const value = yMin + ((yMax - yMin) * i) / 4;
        return {{ value, position: yScale(value) }};
      }});
      const xTicks = xValues.filter((_, idx) => idx === 0 || idx === xValues.length - 1 || idx % Math.ceil(xValues.length / 6) === 0).map((value) => ({{
        value,
        position: xScale(value),
      }}));
      drawAxes(svg, width, height, margin, yTicks, xTicks, opts.xFormatter, opts.yFormatter);

      if (opts.allowNegative && yMin < 0 && yMax > 0) {{
        const zero = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        zero.setAttribute('x1', margin.left);
        zero.setAttribute('x2', width - margin.right);
        zero.setAttribute('y1', yScale(0));
        zero.setAttribute('y2', yScale(0));
        zero.setAttribute('stroke', 'rgba(31,41,51,0.35)');
        zero.setAttribute('stroke-dasharray', '4 4');
        svg.appendChild(zero);
      }}

      series.forEach((line, idx) => {{
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', linePath(line.values, xScale, yScale));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', line.color || palette[idx % palette.length]);
        path.setAttribute('stroke-width', '3');
        path.setAttribute('stroke-linecap', 'round');
        path.setAttribute('stroke-linejoin', 'round');
        svg.appendChild(path);
      }});

      renderLegend(legend, series.map((line, idx) => ({{
        label: line.label,
        color: line.color || palette[idx % palette.length]
      }})));
    }}

    function renderBarChart(svgId, legendId, rows) {{
      const svg = document.getElementById(svgId);
      const legend = document.getElementById(legendId);
      clearSvg(svg);

      const width = 860;
      const height = 320;
      const margin = {{ top: 20, right: 20, bottom: 58, left: 58 }};
      const levels = ['total', 'project', 'category', 'article'];
      const models = Array.from(new Set(rows.map((row) => row.model)));
      const yMax = Math.max(...rows.map((row) => row.wape), 0.0001);
      const band = (width - margin.left - margin.right) / levels.length;
      const groupWidth = band * 0.72;
      const barWidth = groupWidth / Math.max(models.length, 1);
      const yScale = (value) => height - margin.bottom - (value / yMax) * (height - margin.top - margin.bottom);

      Array.from({{ length: 5 }}, (_, i) => {{
        const value = (yMax * i) / 4;
        const y = yScale(value);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', margin.left);
        line.setAttribute('x2', width - margin.right);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(31, 41, 51, 0.12)');
        svg.appendChild(line);
        appendText(svg, formatPct(value), margin.left - 10, y + 4, {{ anchor: 'end' }});
      }});

      levels.forEach((level, levelIndex) => {{
        appendText(svg, level, margin.left + band * levelIndex + band / 2, height - margin.bottom + 24, {{ anchor: 'middle' }});
        models.forEach((model, modelIndex) => {{
          const row = rows.find((item) => item.level === level && item.model === model);
          if (!row) return;
          const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          const x = margin.left + band * levelIndex + (band - groupWidth) / 2 + modelIndex * barWidth;
          rect.setAttribute('x', x);
          rect.setAttribute('y', yScale(row.wape));
          rect.setAttribute('width', Math.max(barWidth - 4, 6));
          rect.setAttribute('height', height - margin.bottom - yScale(row.wape));
          rect.setAttribute('rx', '6');
          rect.setAttribute('fill', palette[modelIndex % palette.length]);
          svg.appendChild(rect);
        }});
      }});

      renderLegend(legend, models.map((model, idx) => ({{ label: model, color: palette[idx % palette.length] }})));
    }}

    function renderTable(nodeId, rows, columns) {{
      const table = document.getElementById(nodeId);
      table.innerHTML = '';
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      columns.forEach((col) => {{
        const th = document.createElement('th');
        th.textContent = col.label;
        headRow.appendChild(th);
      }});
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        columns.forEach((col) => {{
          const td = document.createElement('td');
          td.textContent = col.format ? col.format(row[col.key]) : row[col.key];
          if (col.className) td.className = col.className(row[col.key], row);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});
      table.appendChild(tbody);
    }}

    function setOptions(selectId, options, selected) {{
      const select = document.getElementById(selectId);
      select.innerHTML = '';
      options.forEach((value) => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        option.selected = value === selected;
        select.appendChild(option);
      }});
    }}

    function syncSeriesOptions() {{
      const level = document.getElementById('level-select').value;
      const seriesOptions = payload.filters.series_by_level[level] || [];
      const select = document.getElementById('series-select');
      const current = select.value;
      const selected = seriesOptions.includes(current) ? current : (seriesOptions[0] || '');
      setOptions('series-select', seriesOptions, selected);
    }}

    function renderPredictionExplorer() {{
      const level = document.getElementById('level-select').value;
      const series = document.getElementById('series-select').value;
      const model = document.getElementById('model-select').value;
      const cutoff = document.getElementById('cutoff-select').value;
      const rows = payload.charts.trends
        .filter((row) => row.level === level && row.unique_id === series && row.model === model && row.cutoff === cutoff)
        .sort((a, b) => a.ds.localeCompare(b.ds));

      const lineSeries = [
        {{
          label: 'Actual',
          color: '#1f2933',
          values: rows.map((row, idx) => ({{ x: idx + 1, y: row.y }})),
        }},
        {{
          label: 'Prediction',
          color: '#0f766e',
          values: rows.map((row, idx) => ({{ x: idx + 1, y: row.y_hat }})),
        }},
        {{
          label: 'Absolute error',
          color: '#c2410c',
          values: rows.map((row, idx) => ({{ x: idx + 1, y: row.abs_error }})),
        }},
      ];

      renderMultiLineChart('trend-chart', 'trend-legend', lineSeries, {{
        allowNegative: false,
        xFormatter: (value) => rows[value - 1] ? rows[value - 1].ds.slice(5) : value,
        yFormatter: (value) => formatNumber(value, 0),
      }});

      if (rows.length) {{
        const avgAbs = rows.reduce((sum, row) => sum + row.abs_error, 0) / rows.length;
        const avgBias = rows.reduce((sum, row) => sum + row.error, 0) / rows.length;
        document.getElementById('trend-summary').textContent =
          `${{level}} | ${{series}} | ${{model}} | cutoff ${{cutoff}} | average absolute error ${{formatNumber(avgAbs, 0)}} | average bias ${{formatNumber(avgBias, 0)}}`;
      }} else {{
        document.getElementById('trend-summary').textContent =
          `No rows found for ${{level}} | ${{series}} | ${{model}} | cutoff ${{cutoff}}.`;
      }}
    }}

    function renderDashboard() {{
      document.getElementById('hero-summary').textContent =
        `Generated ${{payload.generated_at}}. This dashboard replaces digging through raw backtest files and gives you a direct view of model quality, drift by horizon, and where forecast errors cluster.`;
      document.getElementById('best-model').textContent = payload.summary.best_model;
      document.getElementById('best-model-detail').textContent = `Best overall WAPE: ${{formatPct(payload.summary.best_wape)}}`;
      document.getElementById('forecast-points').textContent = formatNumber(payload.summary.forecast_points);
      document.getElementById('series-count').textContent = formatNumber(payload.summary.series_count);
      document.getElementById('cutoff-count').textContent = formatNumber(payload.summary.cutoff_count);
      document.getElementById('generated-at').textContent = `Dashboard written to data/forecasts/analysis/dashboard.html on ${{payload.generated_at}}.`;

      setOptions('level-select', payload.filters.levels, payload.filters.default_level);
      setOptions(
        'series-select',
        payload.filters.series_by_level[payload.filters.default_level] || [],
        payload.filters.default_series
      );
      setOptions('model-select', payload.filters.models, payload.filters.default_model);
      setOptions('cutoff-select', payload.filters.cutoffs, payload.filters.default_cutoff);

      const horizonSeries = payload.filters.models.map((model, idx) => ({{
        label: model,
        color: palette[idx % palette.length],
        values: payload.charts.by_horizon
          .filter((row) => row.model === model)
          .map((row) => ({{ x: row.horizon_step, y: row.wape }}))
      }}));
      renderMultiLineChart('horizon-chart', 'horizon-legend', horizonSeries, {{
        allowNegative: false,
        xFormatter: (value) => `H${{value}}`,
        yFormatter: (value) => formatPct(value),
      }});

      renderBarChart('level-chart', 'level-legend', payload.charts.by_level);

      const errorSeries = payload.filters.models.slice(0, 4).flatMap((model, idx) => {{
        const values = payload.charts.errors_daily.filter((row) => row.model === model);
        return [
          {{
            label: `${{model}} MAE`,
            color: palette[idx % palette.length],
            values: values.map((row, rowIdx) => ({{ x: rowIdx + 1, y: row.mae }})),
          }},
          {{
            label: `${{model}} bias`,
            color: palette[(idx + 3) % palette.length],
            values: values.map((row, rowIdx) => ({{ x: rowIdx + 1, y: row.bias }})),
          }},
        ];
      }});
      renderMultiLineChart('error-chart', 'error-legend', errorSeries, {{
        allowNegative: true,
        xFormatter: (value) => value,
        yFormatter: (value) => formatNumber(value, 0),
      }});

      renderTable('overall-table', payload.tables.overall, [
        {{ key: 'model', label: 'Model' }},
        {{ key: 'wape', label: 'WAPE', format: formatPct }},
        {{ key: 'mae', label: 'MAE', format: (value) => formatNumber(value, 0) }},
        {{ key: 'rmse', label: 'RMSE', format: (value) => formatNumber(value, 0) }},
        {{ key: 'bias_pct_of_actual', label: 'Bias %', format: formatPct, className: (value) => value > 0 ? 'metric-bad' : 'metric-good' }},
        {{ key: 'over_forecast_rate', label: 'Over-Forecast Rate', format: formatPct }},
        {{ key: 'n', label: 'Points', format: (value) => formatNumber(value, 0) }},
      ]);

      renderTable('series-table', payload.tables.hardest_series, [
        {{ key: 'unique_id', label: 'Series' }},
        {{ key: 'model', label: 'Model' }},
        {{ key: 'level', label: 'Level' }},
        {{ key: 'wape', label: 'WAPE', format: formatPct }},
        {{ key: 'mae', label: 'MAE', format: (value) => formatNumber(value, 0) }},
        {{ key: 'bias_pct_of_actual', label: 'Bias %', format: formatPct }},
      ]);

      document.getElementById('level-select').addEventListener('change', () => {{
        syncSeriesOptions();
        renderPredictionExplorer();
      }});
      ['series-select', 'model-select', 'cutoff-select'].forEach((id) => {{
        document.getElementById(id).addEventListener('change', renderPredictionExplorer);
      }});
      renderPredictionExplorer();
    }}

    renderDashboard();
  </script>
</body>
</html>
"""


def main() -> None:
    results = run_backtest()
    write_outputs(results)

    print("Saved analysis outputs to", ANALYSIS_DIR.resolve())
    print("Dashboard:", DASHBOARD_PATH.resolve())
    print("\nOverall summary:")
    print(results["overall"].to_string(index=False))
    print("\nTop findings:")
    report_lines = str(results["report"]).splitlines()
    for line in report_lines[3:10]:
        print(line)


if __name__ == "__main__":
    main()
