"""
Phase 3 causal effect recovery.

This script reads the current repo data products, estimates intervention
effects where the observed window supports identification, and writes a
machine-readable effect library plus dashboard artifacts.

Run with:
MPLCONFIGDIR=/tmp/matplotlib Forecasting/bin/python causal_analysis.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


INTERVENTIONS_PATH = Path("interventions.json")
PROCESSED_PATH = Path("data/processed/processed_data.parquet")
FEATURES_PATH = Path("data/features/features.parquet")
CAUSAL_DIR = Path("data/causal")
CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_PATH = CAUSAL_DIR / "dashboard.html"
EFFECTS_PATH = CAUSAL_DIR / "causal_effects.json"
PAYLOAD_PATH = CAUSAL_DIR / "dashboard_payload.json"


@dataclass
class MethodArtifacts:
    summary: dict
    artifact_payload: dict


def load_interventions() -> list[dict]:
    return json.loads(INTERVENTIONS_PATH.read_text())


def load_processed() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%d")
    frame = frame.replace({np.nan: None})
    return frame.to_dict(orient="records")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def _affected_pairs(intervention: dict) -> list[tuple[str, str]]:
    return [
        (project, category)
        for project in intervention["affected_projects"]
        for category in intervention["affected_categories"]
    ]


def _affected_ids(intervention: dict) -> list[str]:
    return [f"{project}/{category}" for project, category in _affected_pairs(intervention)]


def _aggregate_category_views(processed: pd.DataFrame) -> pd.DataFrame:
    out = (
        processed.groupby(["project", "category", "date"], as_index=False)["views_injected"]
        .sum()
        .rename(columns={"views_injected": "views"})
    )
    out["series_id"] = out["project"] + "/" + out["category"]
    return out


def _aggregate_feature_panel(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "log_views",
        "day_of_week",
        "month",
        "weekofyear",
        "is_weekend",
        "doy_sine",
        "doy_cosine",
        "dow_sine",
        "dow_cosine",
        "log_views_lag_1",
        "log_views_lag_7",
        "log_views_lag_30",
        "log_views_rollmean_7",
        "log_views_rollmean_30",
        "log_views_rollstd_7",
        "log_views_rollstd_30",
        "is_launch_event",
        "is_price_change",
        "is_outage_event",
    ]
    out = features.groupby(["project", "category", "date"], as_index=False)[numeric_cols].mean()
    out["series_id"] = out["project"] + "/" + out["category"]
    return out


def _interval(values: pd.Series | np.ndarray, lower: float = 0.025, upper: float = 0.975) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, lower)), float(np.quantile(arr, upper))


def _make_unavailable_result(intervention: dict, method: str, diagnostics: dict, artifact_payload: dict | None = None) -> MethodArtifacts:
    payload = artifact_payload or {}
    summary = {
        "status": "unavailable",
        "type": intervention["type"],
        "method": method,
        "affected": _affected_ids(intervention),
        "p_value": None,
        "diagnostics": diagnostics,
    }
    return MethodArtifacts(summary=summary, artifact_payload=payload)


def _intervention_active_for_series(intervention: dict, series_id: str, start_date: pd.Timestamp) -> bool:
    project, category = series_id.split("/", 1)
    in_scope = (
        project in intervention["affected_projects"]
        and category in intervention["affected_categories"]
    )
    if not in_scope:
        return False
    iv_date = pd.Timestamp(intervention["date"])
    if intervention["type"] == "temporary_suppression":
        end_date = iv_date + pd.Timedelta(days=intervention["duration_days"] - 1)
        return end_date >= start_date
    return iv_date <= start_date


def eligible_synth_donors(category_views: pd.DataFrame, intervention: dict, interventions: list[dict]) -> list[str]:
    all_ids = sorted(category_views["series_id"].unique().tolist())
    treated_ids = set(_affected_ids(intervention))
    donors: list[str] = []
    launch_date = pd.Timestamp(intervention["date"])
    for series_id in all_ids:
        if series_id in treated_ids:
            continue
        if any(
            other["id"] != intervention["id"] and _intervention_active_for_series(other, series_id, launch_date)
            for other in interventions
        ):
            continue
        donors.append(series_id)
    return donors


def _solve_synth_weights(treated_pre: np.ndarray, donor_pre: np.ndarray) -> np.ndarray:
    donor_count = donor_pre.shape[1]
    init = np.full(donor_count, 1.0 / donor_count)

    def objective(weights: np.ndarray) -> float:
        residual = treated_pre - donor_pre @ weights
        return float(np.sum(residual ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(donor_count)]
    result = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Synthetic control optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _fit_synthetic_control(treated: pd.Series, donors: pd.DataFrame, intervention_date: pd.Timestamp) -> dict:
    aligned = pd.concat([treated.rename("treated"), donors], axis=1, join="inner").dropna()
    pre = aligned.index < intervention_date
    post = aligned.index >= intervention_date
    if pre.sum() < 14:
        raise ValueError("Need at least 14 pre-period observations for synthetic control.")
    if post.sum() < 7:
        raise ValueError("Need at least 7 post-period observations for synthetic control.")

    treated_pre_mean = float(aligned.loc[pre, "treated"].mean())
    if treated_pre_mean <= 0:
        raise ValueError("Treated pre-period mean must be positive.")

    donor_pre_means = aligned.loc[pre, donors.columns].mean(axis=0)
    valid_donors = donor_pre_means[donor_pre_means > 0].index.tolist()
    if len(valid_donors) < 1:
        raise ValueError("No donors have positive pre-period mean.")
    aligned = aligned[["treated"] + valid_donors]
    donor_pre_means = donor_pre_means.loc[valid_donors]

    treated_norm = aligned["treated"] / treated_pre_mean
    donors_norm = aligned[valid_donors].divide(donor_pre_means, axis=1)
    weights = _solve_synth_weights(treated_norm.loc[pre].to_numpy(), donors_norm.loc[pre].to_numpy())

    cf_norm = donors_norm.to_numpy() @ weights
    counterfactual = pd.Series(cf_norm * treated_pre_mean, index=aligned.index, name="counterfactual")
    effect = aligned["treated"] - counterfactual
    pct_effect = np.where(counterfactual > 0, effect / counterfactual, np.nan)

    pre_actual = aligned.loc[pre, "treated"]
    pre_cf = counterfactual.loc[pre]
    pre_rmse = float(np.sqrt(np.mean((pre_actual - pre_cf) ** 2)))
    pre_wape = float(np.sum(np.abs(pre_actual - pre_cf)) / np.sum(np.abs(pre_actual))) if np.sum(np.abs(pre_actual)) else np.nan
    pre_r2 = float(r2_score(pre_actual, pre_cf)) if len(pre_actual) > 1 else np.nan

    return {
        "dates": aligned.index,
        "treated": aligned["treated"],
        "counterfactual": counterfactual,
        "effect": effect,
        "pct_effect": pd.Series(pct_effect, index=aligned.index),
        "weights": dict(zip(valid_donors, weights)),
        "pre_mask": pre,
        "post_mask": post,
        "pre_rmse": pre_rmse,
        "pre_wape": pre_wape,
        "pre_r2": pre_r2,
    }


def run_synthetic_control(processed: pd.DataFrame, intervention: dict, interventions: list[dict]) -> MethodArtifacts:
    category_views = _aggregate_category_views(processed)
    treated_ids = _affected_ids(intervention)
    if len(treated_ids) != 1:
        return _make_unavailable_result(
            intervention,
            method="synthetic_control",
            diagnostics={"reason": "expected_single_treated_category", "treated_ids": treated_ids},
        )

    treated_id = treated_ids[0]
    treated_df = category_views[category_views["series_id"] == treated_id].sort_values("date")
    donors = eligible_synth_donors(category_views, intervention, interventions)
    diagnostics = {
        "treated_series": treated_id,
        "candidate_donors": donors,
        "observed_date_min": category_views["date"].min(),
        "observed_date_max": category_views["date"].max(),
    }
    if len(donors) < 1:
        diagnostics["reason"] = "no_eligible_donors"
        return _make_unavailable_result(intervention, "synthetic_control", diagnostics)

    pivot = (
        category_views[category_views["series_id"].isin([treated_id] + donors)]
        .pivot(index="date", columns="series_id", values="views")
        .sort_index()
    )
    iv_date = pd.Timestamp(intervention["date"])
    if pivot.index.min() >= iv_date:
        diagnostics["reason"] = "no_pre_period"
        diagnostics["pre_days"] = 0
        diagnostics["post_days"] = int((pivot.index >= iv_date).sum())
        return _make_unavailable_result(intervention, "synthetic_control", diagnostics)
    if (pivot.index >= iv_date).sum() == 0:
        diagnostics["reason"] = "no_post_period"
        diagnostics["pre_days"] = int((pivot.index < iv_date).sum())
        diagnostics["post_days"] = 0
        return _make_unavailable_result(intervention, "synthetic_control", diagnostics)

    try:
        fit = _fit_synthetic_control(pivot[treated_id], pivot[donors], iv_date)
    except ValueError as exc:
        diagnostics["reason"] = "fit_failed"
        diagnostics["error"] = str(exc)
        return _make_unavailable_result(intervention, "synthetic_control", diagnostics)

    post_mask = fit["post_mask"]
    post_effect = fit["effect"].loc[post_mask]
    post_pct_effect = fit["pct_effect"].loc[post_mask]
    magnitude = float(np.nanmean(post_pct_effect))
    ci_lower, ci_upper = _interval(post_pct_effect)
    avg_abs_effect = float(np.mean(post_effect))

    placebo_rows: list[dict] = []
    for donor_id in donors:
        placebo_pool = [candidate for candidate in donors if candidate != donor_id]
        if not placebo_pool:
            continue
        try:
            placebo_fit = _fit_synthetic_control(pivot[donor_id], pivot[placebo_pool], iv_date)
        except ValueError:
            continue
        placebo_pct = placebo_fit["pct_effect"].loc[placebo_fit["post_mask"]]
        placebo_rows.append(
            {
                "series_id": donor_id,
                "avg_pct_effect": float(np.nanmean(placebo_pct)),
                "abs_avg_pct_effect": float(np.abs(np.nanmean(placebo_pct))),
            }
        )

    treated_abs = float(np.abs(magnitude))
    placebo_abs = [row["abs_avg_pct_effect"] for row in placebo_rows if row["abs_avg_pct_effect"] is not None]
    p_value = float((sum(value >= treated_abs for value in placebo_abs) + 1) / (len(placebo_abs) + 1)) if placebo_abs else None

    artifact_rows = pd.DataFrame(
        {
            "date": fit["dates"],
            "treated": fit["treated"].to_numpy(),
            "counterfactual": fit["counterfactual"].to_numpy(),
            "effect": fit["effect"].to_numpy(),
            "pct_effect": fit["pct_effect"].to_numpy(),
            "period": np.where(fit["pre_mask"], "pre", "post"),
        }
    )
    weights_df = pd.DataFrame(
        [{"series_id": series_id, "weight": weight} for series_id, weight in fit["weights"].items()]
    ).sort_values("weight", ascending=False)

    status = "ok" if fit["pre_r2"] is not None and fit["pre_r2"] >= 0.8 else "degraded"
    diagnostics.update(
        {
            "pre_days": int(fit["pre_mask"].sum()),
            "post_days": int(fit["post_mask"].sum()),
            "pre_rmse": fit["pre_rmse"],
            "pre_wape": fit["pre_wape"],
            "pre_r2": fit["pre_r2"],
            "placebo_count": len(placebo_rows),
        }
    )
    summary = {
        "status": status,
        "type": intervention["type"],
        "method": "synthetic_control",
        "affected": treated_ids,
        "magnitude": magnitude,
        "avg_daily_effect": avg_abs_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ground_truth": float(intervention.get("magnitude")) if intervention.get("magnitude") is not None else None,
        "recovery_error": abs(magnitude - float(intervention["magnitude"])) if intervention.get("magnitude") is not None else None,
        "p_value": p_value,
        "diagnostics": diagnostics,
    }
    artifact_payload = {
        "series": _json_safe_records(artifact_rows),
        "weights": _json_safe_records(weights_df),
        "placebos": placebo_rows,
    }
    return MethodArtifacts(summary=summary, artifact_payload=artifact_payload)


def _make_design_matrix(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    numeric = df[feature_cols].copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    categorical = pd.get_dummies(df[categorical_cols].astype(str), prefix=categorical_cols, drop_first=False)
    return pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)


def _fit_linear_ols(df: pd.DataFrame, target_col: str, feature_cols: list[str], categorical_cols: list[str]) -> tuple[LinearRegression, pd.DataFrame, np.ndarray]:
    X = _make_design_matrix(df, feature_cols=feature_cols, categorical_cols=categorical_cols)
    y = df[target_col].to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


def _crossfit_residuals(df: pd.DataFrame, treatment_col: str, outcome_col: str, confounders: list[str]) -> dict:
    clean = df.dropna(subset=[outcome_col]).copy()
    X = clean[confounders].copy()
    X = X.fillna(X.median(numeric_only=True))
    y_t = clean[treatment_col].to_numpy(dtype=float)
    y_y = clean[outcome_col].to_numpy(dtype=float)
    n_splits = min(5, max(2, int(len(clean) / 30)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    treatment_hat = np.zeros(len(clean))
    outcome_hat = np.zeros(len(clean))
    classifier = LogisticRegression(max_iter=500)
    if len(np.unique(y_t)) < 2:
        raise ValueError("Treatment has no variation for DML.")

    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_t_train = y_t[train_idx]
        y_y_train = y_y[train_idx]

        if len(np.unique(y_t_train)) < 2:
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_t_train)
            treatment_hat[test_idx] = clf.predict_proba(X_test)[:, -1]
        else:
            classifier.fit(X_train, y_t_train)
            treatment_hat[test_idx] = classifier.predict_proba(X_test)[:, -1]

        reg = RandomForestRegressor(n_estimators=300, random_state=42)
        reg.fit(X_train, y_y_train)
        outcome_hat[test_idx] = reg.predict(X_test)

    clean["treatment_resid"] = y_t - treatment_hat
    clean["outcome_resid"] = y_y - outcome_hat
    return {"frame": clean, "n_splits": n_splits}


def run_did_dml(features: pd.DataFrame, intervention: dict) -> MethodArtifacts:
    panel = _aggregate_feature_panel(features)
    treated_ids = set(_affected_ids(intervention))
    control_ids = sorted([sid for sid in panel["series_id"].unique().tolist() if sid not in treated_ids])
    iv_date = pd.Timestamp(intervention["date"])

    panel = panel[panel["series_id"].isin(sorted(treated_ids | set(control_ids)))].copy()
    panel["treated"] = panel["series_id"].isin(treated_ids).astype(int)
    panel["post"] = (panel["date"] >= iv_date).astype(int)
    panel["did_term"] = panel["treated"] * panel["post"]
    panel["time_index"] = (panel["date"] - panel["date"].min()).dt.days

    diagnostics = {
        "treated_groups": sorted(treated_ids),
        "control_groups": control_ids,
        "observed_date_min": panel["date"].min(),
        "observed_date_max": panel["date"].max(),
        "pre_rows": int((panel["date"] < iv_date).sum()),
        "post_rows": int((panel["date"] >= iv_date).sum()),
    }
    if diagnostics["pre_rows"] == 0:
        diagnostics["reason"] = "no_pre_period"
        return _make_unavailable_result(intervention, "did_double_ml", diagnostics)
    if diagnostics["post_rows"] == 0:
        diagnostics["reason"] = "no_post_period"
        return _make_unavailable_result(intervention, "did_double_ml", diagnostics)
    if len(control_ids) == 0:
        diagnostics["reason"] = "no_controls"
        return _make_unavailable_result(intervention, "did_double_ml", diagnostics)
    if panel["did_term"].nunique() < 2:
        diagnostics["reason"] = "no_treatment_variation"
        return _make_unavailable_result(intervention, "did_double_ml", diagnostics)

    feature_cols = ["treated", "post", "did_term", "time_index"]
    categorical_cols = ["series_id", "day_of_week", "month", "weekofyear"]
    did_model, did_X, _ = _fit_linear_ols(panel, "log_views", feature_cols, categorical_cols)
    did_coef = float(did_model.coef_[list(did_X.columns).index("did_term")])

    pre_panel = panel[panel["date"] < iv_date].copy()
    pre_panel["treated_time"] = pre_panel["treated"] * pre_panel["time_index"]
    pre_model, pre_X, _ = _fit_linear_ols(pre_panel, "log_views", ["treated_time", "time_index"], ["series_id", "day_of_week", "month"])
    pretrend_coef = float(pre_model.coef_[list(pre_X.columns).index("treated_time")])

    confounders = [
        "log_views_lag_1",
        "log_views_lag_7",
        "log_views_lag_30",
        "log_views_rollmean_7",
        "log_views_rollmean_30",
        "log_views_rollstd_7",
        "log_views_rollstd_30",
        "is_weekend",
        "doy_sine",
        "doy_cosine",
        "dow_sine",
        "dow_cosine",
    ]
    try:
        dml = _crossfit_residuals(panel, "did_term", "log_views", confounders)
    except ValueError as exc:
        diagnostics["reason"] = "dml_unavailable"
        diagnostics["error"] = str(exc)
        return _make_unavailable_result(intervention, "did_double_ml", diagnostics)

    resid_model = LinearRegression()
    resid_model.fit(dml["frame"][["treatment_resid"]], dml["frame"]["outcome_resid"])
    ate_dml = float(resid_model.coef_[0])
    pct_demand_change = float(np.exp(ate_dml) - 1)
    assumed_price_change = 0.10
    elasticity = float(pct_demand_change / assumed_price_change)
    ci_lower, ci_upper = ate_dml - 1.96 * np.std(dml["frame"]["outcome_resid"]), ate_dml + 1.96 * np.std(dml["frame"]["outcome_resid"])
    diagnostics.update(
        {
            "parallel_trend_coef": pretrend_coef,
            "parallel_trends_ok": abs(pretrend_coef) < 0.01,
            "dml_rows": int(len(dml["frame"])),
            "crossfit_splits": dml["n_splits"],
        }
    )
    summary = {
        "status": "ok",
        "type": intervention["type"],
        "method": "did_double_ml",
        "affected": sorted(treated_ids),
        "ate_did": did_coef,
        "ate_dml": ate_dml,
        "elasticity": elasticity,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ground_truth_ate": float(np.log1p(intervention["magnitude"])),
        "p_value": None,
        "diagnostics": diagnostics,
    }
    trend_rows = (
        panel.groupby(["date", "treated"], as_index=False)["log_views"]
        .mean()
        .assign(group=lambda df: np.where(df["treated"] == 1, "treated", "control"))
    )
    artifact_payload = {
        "trend_rows": _json_safe_records(trend_rows[["date", "group", "log_views"]]),
        "residual_rows": _json_safe_records(dml["frame"][["date", "series_id", "treatment_resid", "outcome_resid"]]),
        "coefficients": {
            "ate_did": did_coef,
            "ate_dml": ate_dml,
            "parallel_trend_coef": pretrend_coef,
        },
    }
    return MethodArtifacts(summary=summary, artifact_payload=artifact_payload)


def run_outage_attribution(processed: pd.DataFrame, intervention: dict) -> MethodArtifacts:
    category_views = _aggregate_category_views(processed)
    iv_date = pd.Timestamp(intervention["date"])
    end_date = iv_date + pd.Timedelta(days=intervention["duration_days"] - 1)
    treated_ids = _affected_ids(intervention)
    diagnostics = {
        "treated_groups": treated_ids,
        "event_start": iv_date,
        "event_end": end_date,
        "observed_date_min": category_views["date"].min(),
        "observed_date_max": category_views["date"].max(),
    }

    event_mask = category_views["date"].between(iv_date, end_date)
    if event_mask.sum() == 0:
        diagnostics["reason"] = "no_event_window_in_observed_data"
        diagnostics["processed_labels_present"] = bool((processed["interventions_id"] == intervention["id"]).any())
        return _make_unavailable_result(intervention, "regression_counterfactual", diagnostics)

    treated = (
        category_views[category_views["series_id"].isin(treated_ids)]
        .groupby("date", as_index=True)["views"]
        .sum()
        .sort_index()
        .rename("treated")
    )
    donor_ids = sorted([sid for sid in category_views["series_id"].unique().tolist() if sid not in treated_ids])
    donor_panel = (
        category_views[category_views["series_id"].isin(donor_ids)]
        .pivot(index="date", columns="series_id", values="views")
        .sort_index()
    )
    aligned = pd.concat([treated, donor_panel], axis=1, join="inner").dropna()
    pre_mask = aligned.index < iv_date
    post_mask = aligned.index.between(iv_date, end_date)
    if pre_mask.sum() < 14:
        diagnostics["reason"] = "insufficient_pre_period"
        diagnostics["pre_days"] = int(pre_mask.sum())
        return _make_unavailable_result(intervention, "regression_counterfactual", diagnostics)

    X_pre = donor_panel.loc[pre_mask].fillna(donor_panel.loc[pre_mask].median())
    y_pre = treated.loc[pre_mask]
    reg = LinearRegression()
    reg.fit(X_pre, y_pre)
    X_full = donor_panel.loc[aligned.index].fillna(donor_panel.loc[aligned.index].median())
    counterfactual = pd.Series(reg.predict(X_full), index=aligned.index, name="counterfactual")
    effect = treated.loc[aligned.index] - counterfactual
    pct_effect = np.where(counterfactual > 0, effect / counterfactual, np.nan)
    post_pct = pd.Series(pct_effect, index=aligned.index).loc[post_mask]
    ci_lower, ci_upper = _interval(post_pct)
    magnitude = float(np.nanmean(post_pct))
    summary = {
        "status": "degraded",
        "type": intervention["type"],
        "method": "regression_counterfactual",
        "affected": treated_ids,
        "magnitude": magnitude,
        "duration_days": int(intervention["duration_days"]),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "recovery_shape": "snap_back",
        "ground_truth": float(intervention["magnitude"]),
        "recovery_error": abs(magnitude - float(intervention["magnitude"])),
        "p_value": None,
        "diagnostics": diagnostics,
    }
    artifact_payload = {
        "series": _json_safe_records(
            pd.DataFrame(
                {
                    "date": aligned.index,
                    "treated": treated.loc[aligned.index].to_numpy(),
                    "counterfactual": counterfactual.to_numpy(),
                    "effect": effect.to_numpy(),
                    "pct_effect": pd.Series(pct_effect, index=aligned.index).to_numpy(),
                    "period": np.where(pre_mask, "pre", np.where(post_mask, "event", "post")),
                }
            )
        ),
    }
    return MethodArtifacts(summary=summary, artifact_payload=artifact_payload)


def build_dashboard_payload(results: dict[str, MethodArtifacts]) -> dict:
    summaries = []
    chart_payload: dict[str, dict] = {}
    diagnostics_rows: list[dict] = []
    for iv_id, result in results.items():
        summaries.append({"intervention_id": iv_id, **result.summary})
        diagnostics = result.summary.get("diagnostics", {})
        diagnostics_rows.append(
            {
                "intervention_id": iv_id,
                "method": result.summary.get("method"),
                "status": result.summary.get("status"),
                "reason": diagnostics.get("reason"),
                "pre_days": diagnostics.get("pre_days"),
                "post_days": diagnostics.get("post_days"),
                "placebo_count": diagnostics.get("placebo_count"),
                "pre_r2": diagnostics.get("pre_r2"),
            }
        )
        chart_payload[iv_id] = result.artifact_payload

    return {
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "summary_rows": _clean(summaries),
        "diagnostics_rows": _clean(diagnostics_rows),
        "charts": _clean(chart_payload),
    }


def build_dashboard_html(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Causal Effect Recovery Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255,252,247,0.92);
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: rgba(31,41,51,0.12);
      --accent: #0f766e;
      --warn: #c2410c;
      --bad: #b91c1c;
      --good: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; color: var(--ink); background: linear-gradient(180deg, #f7f2ea, #efe7dc); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ margin-bottom: 24px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 36px; }}
    .hero p {{ margin: 0; color: var(--muted); max-width: 760px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 18px 38px rgba(15,23,42,0.07); }}
    .kicker {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .metric {{ font-size: 30px; margin: 6px 0; }}
    .muted {{ color: var(--muted); }}
    .ok {{ color: var(--good); }}
    .degraded {{ color: var(--warn); }}
    .unavailable {{ color: var(--bad); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    svg {{ width: 100%; height: 280px; border: 1px solid var(--line); border-radius: 14px; background: white; }}
    .section {{ margin-top: 24px; }}
    .section h2 {{ margin: 0 0 12px; font-size: 24px; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(15,118,110,0.12); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Phase 3 Causal Effect Recovery</h1>
      <p id="hero-copy"></p>
    </section>
    <section class="grid" id="summary-cards"></section>
    <section class="section card">
      <h2>Recovered Effects</h2>
      <table id="summary-table"></table>
    </section>
    <section class="section card">
      <h2>Synthetic Control</h2>
      <svg id="launch-chart" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
      <p class="muted" id="launch-copy"></p>
    </section>
    <section class="section card">
      <h2>DiD / Double ML</h2>
      <svg id="did-chart" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
      <p class="muted" id="did-copy"></p>
    </section>
    <section class="section card">
      <h2>Outage Attribution</h2>
      <svg id="outage-chart" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
      <p class="muted" id="outage-copy"></p>
    </section>
    <section class="section card">
      <h2>Diagnostics</h2>
      <table id="diag-table"></table>
    </section>
  </div>
  <script>
    const payload = {payload_json};

    function fmt(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
      return Number(value).toFixed(digits);
    }}

    function statusClass(status) {{
      return status === 'ok' ? 'ok' : (status === 'degraded' ? 'degraded' : 'unavailable');
    }}

    function renderCards() {{
      const node = document.getElementById('summary-cards');
      payload.summary_rows.forEach((row) => {{
        const div = document.createElement('article');
        div.className = 'card';
        div.innerHTML = `
          <div class="kicker">${{row.intervention_id}}</div>
          <div class="pill ${'{'}statusClass(row.status){'}'}">${{row.status}}</div>
          <div class="metric">${{row.magnitude !== undefined && row.magnitude !== null ? fmt(row.magnitude) : fmt(row.elasticity)}}</div>
          <div class="muted">${{row.method}}</div>
        `;
        node.appendChild(div);
      }});
    }}

    function renderTable(id, rows, columns) {{
      const table = document.getElementById(id);
      const thead = document.createElement('thead');
      const headTr = document.createElement('tr');
      columns.forEach((col) => {{
        const th = document.createElement('th');
        th.textContent = col.label;
        headTr.appendChild(th);
      }});
      thead.appendChild(headTr);
      table.appendChild(thead);
      const tbody = document.createElement('tbody');
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        columns.forEach((col) => {{
          const td = document.createElement('td');
          const value = row[col.key];
          td.textContent = col.format ? col.format(value, row) : (value ?? 'n/a');
          if (col.className) td.className = col.className(value, row);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});
      table.appendChild(tbody);
    }}

    function clearSvg(svg) {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }}

    function drawLineChart(svgId, rows, xKey, seriesDefs, copyId, emptyCopy) {{
      const svg = document.getElementById(svgId);
      clearSvg(svg);
      const copy = document.getElementById(copyId);
      if (!rows || !rows.length) {{
        copy.textContent = emptyCopy;
        return;
      }}
      const width = 900;
      const height = 280;
      const margin = {{ top: 20, right: 20, bottom: 38, left: 56 }};
      const xVals = rows.map((_, idx) => idx);
      const yVals = seriesDefs.flatMap((s) => rows.map((r) => r[s.key]).filter((v) => v !== null && v !== undefined));
      const yMin = Math.min(...yVals, 0);
      const yMax = Math.max(...yVals, 1);
      const xScale = (i) => margin.left + (i / Math.max(rows.length - 1, 1)) * (width - margin.left - margin.right);
      const yScale = (v) => height - margin.bottom - ((v - yMin) / Math.max(yMax - yMin, 1e-6)) * (height - margin.top - margin.bottom);

      const axis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      axis.setAttribute('x1', margin.left);
      axis.setAttribute('x2', width - margin.right);
      axis.setAttribute('y1', height - margin.bottom);
      axis.setAttribute('y2', height - margin.bottom);
      axis.setAttribute('stroke', '#94a3b8');
      svg.appendChild(axis);

      seriesDefs.forEach((series) => {{
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const d = rows.map((row, idx) => `${{idx === 0 ? 'M' : 'L'}} ${{xScale(idx)}} ${{yScale(row[series.key])}}`).join(' ');
        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', series.color);
        path.setAttribute('stroke-width', '3');
        svg.appendChild(path);
      }});

      copy.textContent = `${{seriesDefs.map((s) => s.label).join(' vs ')}} across ${{rows.length}} observations.`;
    }}

    document.getElementById('hero-copy').textContent =
      `Generated ${{payload.generated_at}}. The dashboard shows recovered causal effects where the current repo data supports them and explicit unavailable statuses where it does not.`;

    renderCards();
    renderTable('summary-table', payload.summary_rows, [
      {{ key: 'intervention_id', label: 'Intervention' }},
      {{ key: 'status', label: 'Status', className: (value) => statusClass(value) }},
      {{ key: 'method', label: 'Method' }},
      {{ key: 'magnitude', label: 'Magnitude', format: (value, row) => value !== undefined && value !== null ? fmt(value) : fmt(row.elasticity) }},
      {{ key: 'ci_lower', label: 'CI Lower', format: fmt }},
      {{ key: 'ci_upper', label: 'CI Upper', format: fmt }},
      {{ key: 'p_value', label: 'P-Value', format: fmt }},
    ]);
    renderTable('diag-table', payload.diagnostics_rows, [
      {{ key: 'intervention_id', label: 'Intervention' }},
      {{ key: 'status', label: 'Status', className: (value) => statusClass(value) }},
      {{ key: 'reason', label: 'Reason' }},
      {{ key: 'pre_days', label: 'Pre Days' }},
      {{ key: 'post_days', label: 'Post Days' }},
      {{ key: 'placebo_count', label: 'Placebos' }},
      {{ key: 'pre_r2', label: 'Pre R2', format: fmt }},
    ]);

    drawLineChart(
      'launch-chart',
      payload.charts.launch_event ? payload.charts.launch_event.series : [],
      'date',
      [{{ key: 'treated', label: 'Observed', color: '#1f2933' }}, {{ key: 'counterfactual', label: 'Counterfactual', color: '#0f766e' }}],
      'launch-copy',
      'Launch synthetic-control series is unavailable.'
    );
    drawLineChart(
      'did-chart',
      payload.charts.price_change ? payload.charts.price_change.trend_rows : [],
      'date',
      [{{ key: 'log_views', label: 'Group mean', color: '#c2410c' }}],
      'did-copy',
      'DiD / DML trends are unavailable for the current files.'
    );
    drawLineChart(
      'outage-chart',
      payload.charts.outage_event ? payload.charts.outage_event.series : [],
      'date',
      [{{ key: 'treated', label: 'Observed', color: '#1f2933' }}, {{ key: 'counterfactual', label: 'Counterfactual', color: '#7c2d12' }}],
      'outage-copy',
      'Outage attribution is unavailable for the current files.'
    );
  </script>
</body>
</html>
"""


def write_method_artifacts(results: dict[str, MethodArtifacts]) -> None:
    for intervention_id, result in results.items():
        path = CAUSAL_DIR / f"{intervention_id}.json"
        path.write_text(json.dumps(_clean({"summary": result.summary, "artifacts": result.artifact_payload}), indent=2))


def run_causal_analysis() -> dict[str, MethodArtifacts]:
    interventions = load_interventions()
    processed = load_processed()
    features = load_features()

    interventions_by_id = {iv["id"]: iv for iv in interventions}
    results = {
        "launch_event": run_synthetic_control(processed, interventions_by_id["launch_event"], interventions),
        "price_change": run_did_dml(features, interventions_by_id["price_change"]),
        "outage_event": run_outage_attribution(processed, interventions_by_id["outage_event"]),
    }
    return results


def write_outputs(results: dict[str, MethodArtifacts]) -> None:
    effects = {intervention_id: result.summary for intervention_id, result in results.items()}
    EFFECTS_PATH.write_text(json.dumps(_clean(effects), indent=2))
    write_method_artifacts(results)
    payload = build_dashboard_payload(results)
    PAYLOAD_PATH.write_text(json.dumps(_clean(payload), indent=2))
    DASHBOARD_PATH.write_text(build_dashboard_html(payload), encoding="utf-8")


def main() -> None:
    results = run_causal_analysis()
    write_outputs(results)
    summary_rows = [
        f"{intervention_id}: {result.summary['status']} via {result.summary['method']}"
        for intervention_id, result in results.items()
    ]
    print("Causal analysis complete.")
    for row in summary_rows:
        print(f"- {row}")
    print(f"- Wrote {EFFECTS_PATH}")
    print(f"- Wrote {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
