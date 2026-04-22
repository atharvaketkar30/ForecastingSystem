# ForecastingSystem

Creating a forecasting system that uses forecasting techniques and causal analysis.

## Backtest dashboard

Run the backtest and generate the interactive dashboard:

```bash
MPLCONFIGDIR=/tmp/matplotlib Forecasting/bin/python backtest_analysis.py
```

That writes the analysis outputs to `data/forecasts/analysis/`, including:

- `dashboard.html` for interactive backtest results, forecast trends, and error-pattern charts
- `overall.json`, `by_horizon.json`, `by_level.json`, `errors.json`, and related files that power the dashboard
- `insight_report.md` for the text summary

Open `data/forecasts/analysis/dashboard.html` in your browser to explore the results.

## Causal effect recovery

Run the Phase 3 causal analysis pipeline:

```bash
MPLCONFIGDIR=/tmp/matplotlib Forecasting/bin/python causal_analysis.py
```

That writes outputs to `data/causal/`, including:

- `causal_effects.json` as the machine-readable effect library for downstream scenario modeling
- `dashboard.html` for causal recovery diagnostics and availability status by intervention
- per-intervention JSON artifacts such as `launch_event.json`, `price_change.json`, and `outage_event.json`

Open `data/causal/dashboard.html` in your browser to inspect the causal diagnostics.
