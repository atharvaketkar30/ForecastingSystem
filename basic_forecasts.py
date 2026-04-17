"""
Part A - Top-down forecasts with static proportion allocation.
Part B - Hierarchical reconciliation with simple baseline forecasts.

Hierarchy:
Level 0: total_network
Level 1: project
Level 2: category
Level 3: article

This version avoids external forecasting dependencies that are not installed
locally and instead uses a seasonal-naive baseline that works with the current
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
from hierarchicalforecast.utils import aggregate


PROCESSED_DIR = Path("data/processed")
FORECASTS_DIR = Path("data/forecasts")
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = PROCESSED_DIR / "processed_data.parquet"
ROOT_ID = "total_network"
HORIZON = 28
FREQ = "D"
SEASON_LENGTH = 7
TEST_CUTOFF = datetime.now() - pd.Timedelta(days=90)


def _id_depth(unique_id: str) -> int:
    return unique_id.count("/")


def _build_bottom_level_frame() -> pd.DataFrame:
    """Load bottom-level series and fill missing dates with zeros."""
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT
            project,
            category,
            article,
            CAST(date AS DATE) AS ds,
            CAST(views_injected AS DOUBLE) AS y
        FROM '{PROCESSED_PATH}'
        ORDER BY project, category, article, ds
        """
    ).fetchdf()

    df["ds"] = pd.to_datetime(df["ds"])
    full_dates = pd.date_range(df["ds"].min(), df["ds"].max(), freq=FREQ)

    pieces: list[pd.DataFrame] = []
    for (project, category, article), grp in df.groupby(
        ["project", "category", "article"], sort=True
    ):
        aligned = (
            grp.set_index("ds")[["y"]]
            .reindex(full_dates, fill_value=0.0)
            .rename_axis("ds")
            .reset_index()
        )
        aligned["project"] = project
        aligned["category"] = category
        aligned["article"] = article
        pieces.append(aligned[["project", "category", "article", "ds", "y"]])

    balanced = pd.concat(pieces, ignore_index=True)
    balanced["network"] = ROOT_ID
    return balanced


def load_series() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Build the full hierarchy and summing matrix.

    Returns:
        Y_df: long-format hierarchical dataframe with columns [unique_id, ds, y]
        S_df: summing matrix dataframe
        tags: hierarchy level tags for reconciliation
    """
    bottom = _build_bottom_level_frame()
    spec = [
        ["network"],
        ["network", "project"],
        ["network", "project", "category"],
        ["network", "project", "category", "article"],
    ]
    Y_df, S_df, tags = aggregate(bottom, spec=spec)
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return Y_df, S_df, tags


def train_test_split(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each series so the test set contains at most the next `horizon` dates.
    """
    train = df[df["ds"] < TEST_CUTOFF].copy()
    future_dates = sorted(df.loc[df["ds"] >= TEST_CUTOFF, "ds"].drop_duplicates())
    eval_dates = future_dates[:horizon]
    test = df[df["ds"].isin(eval_dates)].copy()
    return train, test


def _seasonal_naive_values(history: pd.Series, horizon: int, season_length: int = SEASON_LENGTH) -> np.ndarray:
    """Repeat the last seasonal pattern into the forecast horizon."""
    clean = history.astype(float).to_numpy()
    if clean.size == 0:
        return np.zeros(horizon, dtype=float)

    if clean.size >= season_length:
        template = clean[-season_length:]
    else:
        template = np.repeat(clean[-1], season_length)

    reps = int(np.ceil(horizon / len(template)))
    forecast = np.tile(template, reps)[:horizon]
    return np.clip(forecast, a_min=0.0, a_max=None)


def make_base_forecast(train: pd.DataFrame, horizon: int = HORIZON, model_name: str = "baseline") -> pd.DataFrame:
    """Create seasonal-naive forecasts for every hierarchy node."""
    last_train_date = train["ds"].max()
    future_dates = pd.date_range(last_train_date + pd.Timedelta(days=1), periods=horizon, freq=FREQ)

    rows: list[pd.DataFrame] = []
    for unique_id, grp in train.groupby("unique_id", sort=True):
        y_hat = _seasonal_naive_values(grp.sort_values("ds")["y"], horizon=horizon)
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_dates,
                    model_name: y_hat,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


@dataclass
class TopDownStaticProportions:
    """
    Forecast the top node, then allocate down using trailing 42-day proportions.
    """

    horizon: int = HORIZON
    lookback: int = 42
    season_length: int = SEASON_LENGTH

    def __post_init__(self) -> None:
        self._props: dict[str, float] = {}
        self._top_history: pd.Series | None = None
        self._bottom_ids: list[str] = []

    def fit(self, train: pd.DataFrame) -> "TopDownStaticProportions":
        total = (
            train.loc[train["unique_id"] == ROOT_ID, ["ds", "y"]]
            .sort_values("ds")
            .set_index("ds")["y"]
        )
        self._top_history = total
        self._bottom_ids = [uid for uid in train["unique_id"].unique() if _id_depth(uid) == 3]

        total_window = total.tail(self.lookback)
        total_mean = float(total_window.mean()) if not total_window.empty else 0.0

        raw_props: dict[str, float] = {}
        for uid in self._bottom_ids:
            series = (
                train.loc[train["unique_id"] == uid, ["ds", "y"]]
                .sort_values("ds")
                .set_index("ds")["y"]
                .reindex(total.index, fill_value=0.0)
            )
            prop = float(series.tail(self.lookback).mean() / total_mean) if total_mean > 0 else 0.0
            raw_props[uid] = max(prop, 0.0)

        prop_sum = sum(raw_props.values())
        if prop_sum <= 0:
            equal_weight = 1.0 / len(self._bottom_ids)
            self._props = {uid: equal_weight for uid in self._bottom_ids}
        else:
            self._props = {uid: prop / prop_sum for uid, prop in raw_props.items()}

        return self

    def predict(self) -> pd.DataFrame:
        if self._top_history is None:
            raise ValueError("Call fit before predict.")

        future_dates = pd.date_range(
            self._top_history.index.max() + pd.Timedelta(days=1),
            periods=self.horizon,
            freq=FREQ,
        )
        top_hat = _seasonal_naive_values(
            self._top_history,
            horizon=self.horizon,
            season_length=self.season_length,
        )

        bottom_frames: list[pd.DataFrame] = []
        for uid in self._bottom_ids:
            bottom_frames.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": future_dates,
                        "y_hat": top_hat * self._props[uid],
                    }
                )
            )
        bottom_fcst = pd.concat(bottom_frames, ignore_index=True)

        all_rows = [
            pd.DataFrame({"unique_id": ROOT_ID, "ds": future_dates, "y_hat": top_hat})
        ]
        for depth in (1, 2):
            agg = bottom_fcst.copy()
            agg["unique_id"] = agg["unique_id"].str.split("/").str[: depth + 1].str.join("/")
            agg = agg.groupby(["unique_id", "ds"], as_index=False)["y_hat"].sum()
            all_rows.append(agg)
        all_rows.append(bottom_fcst)

        out = pd.concat(all_rows, ignore_index=True)
        out["y_hat"] = out["y_hat"].clip(lower=0.0)
        return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def reconcile_forecasts(
    train: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Generate reconciled hierarchical forecasts from seasonal-naive base forecasts."""
    base_fcst = make_base_forecast(train=train, horizon=horizon, model_name="baseline")
    reconcilers = [
        BottomUp(),
        TopDown(method="proportion_averages"),
        MinTrace(method="ols"),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled = hrec.reconcile(
        Y_hat_df=base_fcst,
        Y_df=train,
        S_df=S_df,
        tags=tags,
        is_balanced=True,
    )
    return reconciled.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def evaluate_forecasts(actuals: pd.DataFrame, forecasts: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return MAE, RMSE, and WAPE by hierarchy node."""
    merged = actuals.merge(
        forecasts[["unique_id", "ds", value_col]],
        on=["unique_id", "ds"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["unique_id", "mae", "rmse", "wape"])

    err = merged[value_col] - merged["y"]
    summary = (
        merged.assign(abs_err=err.abs(), sq_err=err.pow(2))
        .groupby("unique_id", as_index=False)
        .agg(
            mae=("abs_err", "mean"),
            rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
            actual_sum=("y", "sum"),
            abs_err_sum=("abs_err", "sum"),
        )
    )
    summary["wape"] = np.where(summary["actual_sum"] > 0, summary["abs_err_sum"] / summary["actual_sum"], np.nan)
    return summary[["unique_id", "mae", "rmse", "wape"]].sort_values("unique_id").reset_index(drop=True)


def main() -> None:
    Y_df, S_df, tags = load_series()
    train, test = train_test_split(Y_df, horizon=HORIZON)

    topdown = TopDownStaticProportions(horizon=test["ds"].nunique()).fit(train)
    topdown_fcst = topdown.predict()

    reconciled_fcst = reconcile_forecasts(
        train=train,
        S_df=S_df,
        tags=tags,
        horizon=test["ds"].nunique(),
    )

    topdown_metrics = evaluate_forecasts(test, topdown_fcst, "y_hat")
    baseline_metrics = evaluate_forecasts(test, reconciled_fcst, "baseline")
    bottomup_metrics = evaluate_forecasts(test, reconciled_fcst, "baseline/BottomUp")
    mint_metrics = evaluate_forecasts(test, reconciled_fcst, "baseline/MinTrace_method-ols")

    topdown_fcst.to_parquet(FORECASTS_DIR / "topdown_static.parquet", index=False)
    reconciled_fcst.to_parquet(FORECASTS_DIR / "hierarchical_reconciled.parquet", index=False)

    metric_table = (
        topdown_metrics.rename(columns={"mae": "topdown_mae", "rmse": "topdown_rmse", "wape": "topdown_wape"})
        .merge(
            baseline_metrics.rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse", "wape": "baseline_wape"}),
            on="unique_id",
            how="outer",
        )
        .merge(
            bottomup_metrics.rename(columns={"mae": "bottomup_mae", "rmse": "bottomup_rmse", "wape": "bottomup_wape"}),
            on="unique_id",
            how="outer",
        )
        .merge(
            mint_metrics.rename(columns={"mae": "mint_mae", "rmse": "mint_rmse", "wape": "mint_wape"}),
            on="unique_id",
            how="outer",
        )
        .sort_values("unique_id")
        .reset_index(drop=True)
    )
    metric_table.to_csv(FORECASTS_DIR / "forecast_metrics.csv", index=False)

    focus_ids = [ROOT_ID, "total_network/en.wikipedia", "total_network/de.wikipedia"]
    print("Saved forecasts to", FORECASTS_DIR.resolve())
    print("\nTop-down preview:")
    print(topdown_fcst[topdown_fcst["unique_id"].isin(focus_ids)].head(9).to_string(index=False))
    print("\nMetrics preview:")
    print(metric_table[metric_table["unique_id"].isin(focus_ids)].to_string(index=False))


if __name__ == "__main__":
    main()
