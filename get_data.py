import requests
import pandas as pd
import numpy as np
import json 
import duckdb
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import time
from pathlib import Path

##### Configuration #####

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
IV_PATH = Path("interventions.json")

for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Heirarchy: project -> category -> articles

SERIES_HEIRARCHY = {
    "en.wikipedia": {
        "tech": [
            "Python_(programming_language)",
            "Machine_learning",
            "Artificial_intelligence",
            "Cloud_computing",
            "Application_programming_interface",
        ],
        "finance": [
            "Stock_market",
            "Hedge_fund",
            "Venture_capital",
            "Cryptocurrency",
            "Interest_rate",
        ],
    },
    "de.wikipedia": {
        "tech": [
            "Python_(Programmiersprache)",
            "Maschinelles_Lernen",
            "Künstliche_Intelligenz",
        ],
        "finance": [
            "Aktienmarkt",
            "Hedge-Fonds",
            "Kryptowährung",
        ],
    },
}

INTERVENTIONS = [
    {
        "id": "launch_event",
        "type": "step_change",
        "date": "2025-06-01",
        "magnitude": 0.35,
        "affected_projects": ["en.wikipedia"],
        "affected_categories": ["tech"],
        "description": "New model launch → 35% sustained uplift in tech API usage",
    },
    {
        "id": "price_change",
        "type": "elasticity_shift",
        "date": "2023-09-01",
        "magnitude": -0.20,
        "affected_projects": ["en.wikipedia"],
        "affected_categories": ["tech", "finance"],
        "description": "Price increase → 20% demand reduction (elasticity response)",
    },
    {
        "id": "outage_event",
        "type": "temporary_suppression",
        "date": "2024-11-15",
        "duration_days": 3,
        "magnitude": -0.80,
        "affected_projects": ["en.wikipedia", "de.wikipedia"],
        "affected_categories": ["tech", "finance"],
        "description": "3-day outage → 80% traffic suppression",
    },
]

## ----- Layer 1 Raw Pull to Parquet ----- ##

def fetch_pageviews(project: str, article: str, start_date: str, end_date: str) -> list[dict]:
    """Hit Wikimedia REST API, return raw items list"""
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{article}/daily/{start_date}/{end_date}"
    headers = {"User-Agent": "APIForecastingProject/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json().get("items", [])
        elif response.status_code == 404:
            print(f"Warning: Data not found for {project} - {article}")
            return []
        else:
            print(f"Error {response.status_code} for {project} - {article}: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request error for {project} - {article}: {e}")
        return []
    
def raw_path(project: str, category: str, article: str) -> Path:
    """Construct raw data path for given article"""
    safe_article = article.replace("/", "_")  # Sanitize article name for filesystem
    return RAW_DIR / project / category / f"{safe_article}.parquet"

def pull_all_series(start_date: str, end_date: str):
    """Iterate over heirarchy, pull data, save to Parquet"""
    pull_date = datetime.now(timezone.utc).isoformat()
    total = sum(len(arts) for cats in SERIES_HEIRARCHY.values() for arts in cats.values())
    print(f"Pulling {total} series from {start_date} to {end_date} at {pull_date}")

    for project, categories in SERIES_HEIRARCHY.items():
        for category, articles in categories.items():
            for article in tqdm(articles, desc=f"Pulling {project} - {category}"):
                out_path = raw_path(project, category, article)
                if out_path.exists():
                    continue  # Skip if already pulled
                
                data = fetch_pageviews(project, article, start_date, end_date)
                if data:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df = pd.DataFrame([{
                        "project": project,
                        "category": category,
                        "article": article,
                        "date": pd.to_datetime(item["timestamp"][:8], format="%Y%m%d"),  # YYYYMMDD
                        "views": item["views"],
                        "pull_date": pull_date
                    } for item in data])
                    df.to_parquet(out_path, index=False)
                    time.sleep(0.2)  # Rate limit to avoid hitting API limits

    n_files = len(list(RAW_DIR.glob("**/*.parquet")))
    print(f"Completed pulling. Total series pulled: {n_files} in {RAW_DIR}")


## ----- Layer 2: Processed Data + Apply Interventions ----- ##

def apply_interventions():
    """
    Read all raw Parquet with DuckDB, apply interventions
    write single processed Parquet (partitoned by project).
    """
    con = duckdb.connect()

    df = con.execute(f"""
        SELECT project, category, article, date, views, pull_date
        FROM read_parquet('{RAW_DIR}/**/*.parquet')
        ORDER BY project, category, article, date
    """).fetchdf()

    df["views_injected"] = df["views"].astype(float)
    df["interventions_id"] = None

    for iv in INTERVENTIONS:
        iv_date = pd.to_datetime(iv["date"])
        mask = (
            df["date"] >= iv_date
        ) & (
            df["project"].isin(iv["affected_projects"])
        ) & (
            df["category"].isin(iv["affected_categories"])
        )
        if iv["type"] == "step_change":
            df.loc[mask, "views_injected"] *= (1 + iv["magnitude"])
        elif iv["type"] == "elasticity_shift":
            df.loc[mask, "views_injected"] *= (1 + iv["magnitude"])
        elif iv["type"] == "temporary_suppression":
            end_date = iv_date + pd.Timedelta(days=iv["duration_days"])
            temp_mask = mask & (df["date"] <= end_date)
            df.loc[temp_mask, "views_injected"] *= (1 + iv["magnitude"])
        
        df.loc[mask, "interventions_id"] = iv["id"]

    np.random.seed(42)
    df["views_injected"] = (
        df["views_injected"] * (1 + np.random.uniform(0, 0.05, size=len(df)))
    ).clip(lower=0).round().astype(int)

    df["log_views"] = np.log1p(df["views_injected"])
    df["processed_date"] = datetime.now(timezone.utc).isoformat()

    out_path = PROCESSED_DIR / "processed_data.parquet"
    df.to_parquet(out_path, index=False)

    IV_PATH.write_text(json.dumps(INTERVENTIONS, indent=2))
    print(f"Processed data with interventions applied. Output at {out_path}")

### ----- Layer 3: Feature Engineering + DuckDB SQL ----- ###
def build_features():
    """
    Build feature table : 
    - Lag features: 1, 7, 30 day lags of log_views
    - Rolling mean features: 7-day rolling mean of log_views
    - Calendar features: day of week, month
    - Event flags

    Output: features/features.parquet
    """
    con = duckdb.connect()

    interventions_by_id = {iv["id"]: iv for iv in INTERVENTIONS}
    iv_dates = {iv_id: pd.to_datetime(iv["date"]) for iv_id, iv in interventions_by_id.items()}
    outage_iv = interventions_by_id["outage_event"]
    outage_end = iv_dates["outage_event"] + pd.Timedelta(days=outage_iv["duration_days"])

    df = con.execute(f"""
        SELECT project, category, article, date, log_views, interventions_id
        FROM read_parquet('{PROCESSED_DIR}/processed_data.parquet')
        ORDER BY project, category, article, date
    """).fetchdf()

    df["date"] = pd.to_datetime(df["date"])

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["doy_sine"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cosine"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["dow_sine"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cosine"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features
    for lag in [1, 7, 30]:
        df[f"log_views_lag_{lag}"] = df.groupby(["project", "category", "article"])["log_views"].shift(lag)

    df["log_views_rollmean_7"] = df.groupby(["project", "category", "article"])["log_views"].transform(lambda x: x.rolling(window=7).mean())
    df["log_views_rollmean_30"] = df.groupby(["project", "category", "article"])["log_views"].transform(lambda x: x.rolling(window=30).mean())
    df["log_views_rollstd_7"] = df.groupby(["project", "category", "article"])["log_views"].transform(lambda x: x.rolling(window=7).std())
    df["log_views_rollstd_30"] = df.groupby(["project", "category", "article"])["log_views"].transform(lambda x: x.rolling(window=30).std())

    # Event flags are based on intervention scope/date, not interventions_id,
    # because multiple interventions can apply to the same row over time.
    launch_iv = interventions_by_id["launch_event"]
    price_iv = interventions_by_id["price_change"]

    launch_mask = (
        df["project"].isin(launch_iv["affected_projects"])
        & df["category"].isin(launch_iv["affected_categories"])
        & df["date"].ge(iv_dates["launch_event"])
    )
    price_mask = (
        df["project"].isin(price_iv["affected_projects"])
        & df["category"].isin(price_iv["affected_categories"])
        & df["date"].ge(iv_dates["price_change"])
    )
    outage_mask = (
        df["project"].isin(outage_iv["affected_projects"])
        & df["category"].isin(outage_iv["affected_categories"])
        & df["date"].ge(iv_dates["outage_event"])
        & df["date"].le(outage_end)
    )

    df["is_launch_event"] = launch_mask.astype(int)
    df["is_price_change"] = price_mask.astype(int)
    df["is_outage_event"] = outage_mask.astype(int)

    out_path = FEATURES_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Feature engineering completed. Output at {out_path}")
    return df

#### ----- Sanity Check: Run all layers ----- ####
def run_sanity_check():
    con = duckdb.connect()
    raw_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{RAW_DIR}/**/*.parquet')").fetchone()[0]
    processed_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PROCESSED_DIR}/processed_data.parquet')").fetchone()[0]
    features_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{FEATURES_DIR}/features.parquet')").fetchone()[0]
    print(f"Sanity Check: Raw={raw_n} rows, Processed={processed_n} rows, Features={features_n} rows")

    iv_check = con.execute(f"""
        SELECT interventions_id, COUNT(*) AS n_rows
        FROM read_parquet('{PROCESSED_DIR}/processed_data.parquet')
        WHERE interventions_id IS NOT NULL
        GROUP BY interventions_id
    """).fetchdf()
    print(iv_check.to_string(index=False))

    print("\n Launch injection check (should see uplift post vs pre):")
    launch_check = con.execute(f"""
        SELECT 
            CASE WHEN date < '2023-06-01' THEN 'pre_launch' ELSE 'post_launch' END AS period,
            AVG(views_injected) AS avg_views,
            AVG(views) AS avg_views_original
        FROM read_parquet('{PROCESSED_DIR}/processed_data.parquet')
        WHERE project = 'en.wikipedia' AND category = 'tech'
        GROUP BY period
    """).fetchdf()
    print(launch_check.to_string(index=False))
                           

if __name__ == "__main__":
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y%m%d")
    pull_all_series(start_date, end_date)
    apply_interventions()
    build_features()
    run_sanity_check()
