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
