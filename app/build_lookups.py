"""
Build lookup tables for the Streamlit FTT inference app.

Reads `Data/01_cleaned_data.csv` (which has raw AIRLINE / ORIGIN / DEST identifiers
plus base weather/congestion features) and produces per-group statistics matching
the feature engineering recipe in `Notebooks/03_feature_engineering_final_v2.ipynb`.

Outputs go into `lookups/` at the project root. Original data files are not modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# This file lives in <project_root>/app/. The cleaned dataset is at the
# project root; lookup outputs are bundled alongside the app.
APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent
DATA = ROOT / "Data" / "01_cleaned_data.csv"
OUT = APP_DIR / "lookups"
OUT.mkdir(exist_ok=True)

# Month → season bucket, derived from monthly delay rates in the EDA chapter.
# Buckets: 0 = low, 1 = neutral, 2 = medium, 3 = high.
SEASON_MAP = {1: 1, 2: 3, 3: 1, 4: 1, 5: 2, 6: 3, 7: 3, 8: 3, 9: 0, 10: 0, 11: 1, 12: 2}

# WMO weather codes flagged as adverse in the EDA (drizzle, rain, snow intensities).
ADVERSE_CODES = [51, 53, 55, 61, 63, 65, 71, 73, 75]


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-row temporal / weather / route derivations used at training time."""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["MONTH"] = df["DATE"].dt.month
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    df["SEASON"] = df["MONTH"].map(SEASON_MAP)
    df["LOG_DISTANCE"] = np.log1p(df["DISTANCE"])
    df["IS_FREEZING"] = (df["temp"] < -5).astype(int)
    df["IS_RAINING"] = (df["precip"] > 0).astype(int)
    df["IS_SNOWING"] = (df["snowfall"] > 0).astype(int)
    df["IS_ADVERSE_WEATHER"] = df["weather_code"].isin(ADVERSE_CODES).astype(int)
    df["ROUTE"] = df["ORIGIN_AIRPORT"] + "-" + df["DESTINATION_AIRPORT"]
    return df


def compute_rate(df: pd.DataFrame, group_cols, min_samples: int) -> pd.DataFrame:
    """Group-level mean of TARGET (delay rate), filtered to groups with ≥ min_samples rows.

    The min_samples filter excludes rare groups whose mean would be unreliable.
    Thresholds mirror those used in the training feature engineering.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    g = df.groupby(group_cols)["TARGET"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    g = g[g["count"] >= min_samples][["rate"]].reset_index()
    return g


def main():
    print(f"Reading {DATA} …")
    df = pd.read_csv(DATA)
    print(f"  loaded {len(df):,} rows")
    df = add_safe_features(df)

    global_rate = float(df["TARGET"].mean())
    global_dep_delay = float(df["DEPARTURE_DELAY"].mean())
    print(f"  global delay rate: {global_rate:.4f}")
    print(f"  global mean dep delay: {global_dep_delay:.2f} min")

    # ------------------------------------------------------------------
    # Per-airline / origin / dest stats. Mean of TARGET = historical delay
    # rate; mean of DEPARTURE_DELAY = static stand-in for the rolling-7-day
    # delay feature that a streaming pipeline would compute in production.
    # ------------------------------------------------------------------
    airline_stats = (
        df.groupby("AIRLINE")
        .agg(AIRLINE_RATE=("TARGET", "mean"), AIRLINE_ROLL7_DELAY=("DEPARTURE_DELAY", "mean"), n=("TARGET", "size"))
        .reset_index()
    )
    airline_stats = airline_stats[airline_stats["n"] >= 20].drop(columns="n")
    airline_stats.to_parquet(OUT / "airline_stats.parquet", index=False)
    print(f"  airline_stats: {len(airline_stats)} airlines")

    origin_stats = (
        df.groupby("ORIGIN_AIRPORT")
        .agg(
            ORIGIN_RATE=("TARGET", "mean"),
            ORIGIN_AIRPORT_ROLL7_DELAY=("DEPARTURE_DELAY", "mean"),
            ORIGIN_CONGESTION=("ORIGIN_CONGESTION", "mean"),
            n=("TARGET", "size"),
        )
        .reset_index()
    )
    origin_stats = origin_stats[origin_stats["n"] >= 20].drop(columns="n").rename(columns={"ORIGIN_AIRPORT": "ORIGIN"})
    origin_stats.to_parquet(OUT / "origin_stats.parquet", index=False)
    print(f"  origin_stats: {len(origin_stats)} origins")

    dest_stats = (
        df.groupby("DESTINATION_AIRPORT")
        .agg(DEST_RATE=("TARGET", "mean"), DEST_CONGESTION=("DEST_CONGESTION", "mean"), n=("TARGET", "size"))
        .reset_index()
    )
    dest_stats = dest_stats[dest_stats["n"] >= 20].drop(columns="n").rename(columns={"DESTINATION_AIRPORT": "DEST"})
    dest_stats.to_parquet(OUT / "dest_stats.parquet", index=False)
    print(f"  dest_stats: {len(dest_stats)} destinations")

    # ------------------------------------------------------------------
    # Interaction rate tables
    # ------------------------------------------------------------------
    route_rates = compute_rate(df, "ROUTE", min_samples=20).rename(columns={"rate": "ROUTE_RATE"})
    route_rates[["ORIGIN", "DEST"]] = route_rates["ROUTE"].str.split("-", expand=True)
    route_rates = route_rates[["ORIGIN", "DEST", "ROUTE_RATE"]]
    route_rates.to_parquet(OUT / "route_stats.parquet", index=False)
    print(f"  route_stats: {len(route_rates)} routes")

    route_season = compute_rate(df, ["ROUTE", "SEASON"], min_samples=5).rename(columns={"rate": "ROUTE_SEASON_RATE"})
    route_season[["ORIGIN", "DEST"]] = route_season["ROUTE"].str.split("-", expand=True)
    route_season = route_season[["ORIGIN", "DEST", "SEASON", "ROUTE_SEASON_RATE"]]
    route_season.to_parquet(OUT / "route_season_stats.parquet", index=False)
    print(f"  route_season_stats: {len(route_season)} (route, season) pairs")

    airline_month = compute_rate(df, ["AIRLINE", "MONTH"], min_samples=15).rename(
        columns={"rate": "AIRLINE_MONTH_RATE"}
    )
    airline_month.to_parquet(OUT / "airline_month_stats.parquet", index=False)
    print(f"  airline_month_stats: {len(airline_month)} (airline, month) pairs")

    origin_hour = compute_rate(df, ["ORIGIN_AIRPORT", "HOUR"], min_samples=10).rename(
        columns={"rate": "ORIGIN_HOUR_RATE", "ORIGIN_AIRPORT": "ORIGIN"}
    )
    origin_hour.to_parquet(OUT / "origin_hour_stats.parquet", index=False)
    print(f"  origin_hour_stats: {len(origin_hour)} (origin, hour) pairs")

    # ------------------------------------------------------------------
    # Weather presets — pick realistic numeric values per qualitative label
    # ------------------------------------------------------------------
    def stats_of(mask, label):
        sub = df[mask]
        if len(sub) == 0:
            return None
        return {
            "label": label,
            "n": int(len(sub)),
            "weather_code": int(sub["weather_code"].mode().iloc[0]),
            "temp": float(sub["temp"].median()),
            "precip": float(sub["precip"].median()),
            "snowfall": float(sub["snowfall"].median()),
            "wind_speed": float(sub["wind_speed"].median()),
            "wind_gusts": float(sub["wind_gusts"].median()),
            "IS_FREEZING": int((sub["temp"].median() < -5)),
            "IS_RAINING": int((sub["precip"].median() > 0)),
            "IS_SNOWING": int((sub["snowfall"].median() > 0)),
            "IS_ADVERSE_WEATHER": int(sub["weather_code"].mode().iloc[0] in ADVERSE_CODES),
        }

    presets = {
        "Auto (typical for season)": None,  # filled per-season at inference time
        "Clear / Sunny": stats_of(df["weather_code"].isin([0, 1]), "Clear / Sunny"),
        "Cloudy": stats_of(df["weather_code"].isin([2, 3]), "Cloudy"),
    }

    rain_mask = df["IS_RAINING"] == 1
    if rain_mask.any():
        rain_med = df.loc[rain_mask, "precip"].median()
        presets["Light rain"] = stats_of(rain_mask & (df["precip"] < rain_med), "Light rain")
        presets["Heavy rain"] = stats_of(rain_mask & (df["precip"] >= rain_med), "Heavy rain")

    snow_mask = df["IS_SNOWING"] == 1
    if snow_mask.any():
        presets["Snow"] = stats_of(snow_mask, "Snow")

    fog_mask = df["weather_code"].isin([45, 48])
    if fog_mask.any():
        presets["Fog"] = stats_of(fog_mask, "Fog")

    # Per-season "Auto" defaults
    seasonal = {}
    for season in sorted(df["SEASON"].unique()):
        s = stats_of(df["SEASON"] == season, f"Auto-season-{season}")
        seasonal[str(int(season))] = s

    presets_payload = {"presets": {k: v for k, v in presets.items() if v is not None}, "seasonal_auto": seasonal}
    (OUT / "weather_presets.json").write_text(json.dumps(presets_payload, indent=2))
    print(f"  weather_presets: {len(presets_payload['presets'])} presets + {len(seasonal)} seasonal autos")

    # ------------------------------------------------------------------
    # Defaults for features the user cannot reasonably provide. The -1.0
    # sentinel matches the convention established during training (NB03)
    # for flights with no traceable previous leg.
    # ------------------------------------------------------------------
    defaults = {
        "global_delay_rate": global_rate,
        "global_dep_delay_mean": global_dep_delay,
        "TURNAROUND_MIN": -1.0,
        "PREV_LEG_DEP_DELAY": -1.0,
        "PREV_LEG_ARR_DELAY": -1.0,
        "PREV_LEG_DELAYED": -1.0,
        "ORIGIN_CONGESTION_global": float(df["ORIGIN_CONGESTION"].mean()),
        "DEST_CONGESTION_global": float(df["DEST_CONGESTION"].mean()),
    }
    (OUT / "defaults.json").write_text(json.dumps(defaults, indent=2))
    print(f"  defaults.json written")

    print("\nLookups built successfully.")


if __name__ == "__main__":
    main()
