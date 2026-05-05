"""
Inference helpers for the FT-Transformer flight delay predictor.

Loads the saved pytorch-tabular model and threshold, plus the lookup tables
produced by `build_lookups.py`, and exposes a single `predict_delay()` function
that takes user-friendly inputs and returns (probability, is_delayed).
"""
from __future__ import annotations

import json
from math import asin, cos, log1p, radians, sin, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CPU-load patches — the FTT model was trained on Colab GPU. The saved
# checkpoint references CUDA storage and the saved config.yml requests
# accelerator='gpu'. We map weights to CPU and force the inference Trainer
# to CPU. Neither patch modifies any file on disk.
# ---------------------------------------------------------------------------
import torch

_orig_torch_load = torch.load


def _torch_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_cpu

# pytorch-tabular reads accelerator/devices from the saved config when it
# rebuilds the inference Trainer. Wrap _prepare_trainer to coerce CPU.
from pytorch_tabular.tabular_model import TabularModel as _TM

_orig_prep_trainer = _TM._prepare_trainer


def _prep_trainer_cpu(self, *args, **kwargs):
    self.config.accelerator = "cpu"
    self.config.devices = 1
    return _orig_prep_trainer(self, *args, **kwargs)


_TM._prepare_trainer = _prep_trainer_cpu

# This file lives in <project_root>/app/. Models and Data sit at the project
# root; the lookups folder is bundled with the app.
APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent
MODEL_DIR = ROOT / "Models" / "ftt_model"
THRESHOLD_PATH = ROOT / "Models" / "ftt_threshold.pkl"
LOOKUPS = APP_DIR / "lookups"
AIRPORTS_CSV = ROOT / "Data" / "airports.csv"

SEASON_MAP = {1: 1, 2: 3, 3: 1, 4: 1, 5: 2, 6: 3, 7: 3, 8: 3, 9: 0, 10: 0, 11: 1, 12: 2}

CONTINUOUS_COLS = [
    "LOG_DISTANCE", "temp", "precip", "snowfall", "wind_speed", "wind_gusts",
    "ORIGIN_CONGESTION", "DEST_CONGESTION", "TURNAROUND_MIN",
    "PREV_LEG_DEP_DELAY", "PREV_LEG_ARR_DELAY",
    "AIRLINE_ROLL7_DELAY", "ORIGIN_AIRPORT_ROLL7_DELAY",
    "AIRLINE_RATE", "ORIGIN_RATE", "DEST_RATE", "ROUTE_RATE",
    "ROUTE_SEASON_RATE", "AIRLINE_MONTH_RATE", "ORIGIN_HOUR_RATE",
]
CATEGORICAL_COLS = [
    "MONTH", "DAY_OF_WEEK", "HOUR", "IS_WEEKEND", "SEASON",
    "weather_code", "IS_FREEZING", "IS_RAINING", "IS_SNOWING",
    "IS_ADVERSE_WEATHER", "PREV_LEG_DELAYED",
]


@st.cache_resource(show_spinner="Loading FT-Transformer model…")
def load_model():
    from pytorch_tabular import TabularModel

    model = TabularModel.load_model(str(MODEL_DIR))
    threshold = float(joblib.load(THRESHOLD_PATH))
    return model, threshold


@st.cache_data(show_spinner=False)
def load_lookups() -> dict:
    return {
        "airline": pd.read_parquet(LOOKUPS / "airline_stats.parquet"),
        "origin": pd.read_parquet(LOOKUPS / "origin_stats.parquet"),
        "dest": pd.read_parquet(LOOKUPS / "dest_stats.parquet"),
        "route": pd.read_parquet(LOOKUPS / "route_stats.parquet"),
        "route_season": pd.read_parquet(LOOKUPS / "route_season_stats.parquet"),
        "airline_month": pd.read_parquet(LOOKUPS / "airline_month_stats.parquet"),
        "origin_hour": pd.read_parquet(LOOKUPS / "origin_hour_stats.parquet"),
        "weather": json.loads((LOOKUPS / "weather_presets.json").read_text()),
        "defaults": json.loads((LOOKUPS / "defaults.json").read_text()),
    }


@st.cache_data(show_spinner=False)
def load_airports() -> pd.DataFrame:
    df = pd.read_csv(AIRPORTS_CSV)
    return df.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) coordinates."""
    r_km = 6371.0
    a1, a2 = radians(lat1), radians(lat2)
    da = radians(lat2 - lat1)
    do = radians(lon2 - lon1)
    h = sin(da / 2) ** 2 + cos(a1) * cos(a2) * sin(do / 2) ** 2
    km = 2 * r_km * asin(sqrt(h))
    return km * 0.621371


def _lookup(df: pd.DataFrame, where: dict, value_col: str, fallback: float) -> float:
    """Fetch `value_col` from the row matching all `where` key/value pairs, else fallback.

    Used to retrieve precomputed group statistics from the parquet lookup tables.
    """
    mask = np.ones(len(df), dtype=bool)
    for k, v in where.items():
        mask &= df[k].values == v
    if mask.any():
        return float(df.loc[mask, value_col].iloc[0])
    return float(fallback)


def build_feature_row(
    *,
    origin: str,
    dest: str,
    airline: str,
    month: int,
    day_of_week: int,
    hour: int,
    weather_label: str,
    lookups: dict,
    airports: pd.DataFrame,
) -> pd.DataFrame:
    """Construct a single-row DataFrame with all 31 features the FTT expects.

    `month` is 1-12; `day_of_week` follows the BTS convention (Monday=1 .. Sunday=7).
    The model never consumes a year — only month/day-of-week-derived features — so
    the dashboard input is intentionally year-less.
    """
    is_weekend = int(day_of_week in (6, 7))
    season = int(SEASON_MAP[month])

    # Distance via airport coordinates
    o_row = airports[airports["IATA_CODE"] == origin]
    d_row = airports[airports["IATA_CODE"] == dest]
    if len(o_row) and len(d_row):
        miles = _haversine_miles(
            float(o_row["LATITUDE"].iloc[0]), float(o_row["LONGITUDE"].iloc[0]),
            float(d_row["LATITUDE"].iloc[0]), float(d_row["LONGITUDE"].iloc[0]),
        )
    else:
        miles = 800.0  # reasonable US median fallback
    log_distance = log1p(miles)

    defaults = lookups["defaults"]
    g_rate = defaults["global_delay_rate"]
    g_delay = defaults["global_dep_delay_mean"]

    # Rate / rolling lookups. If a key is missing (unseen airline/route),
    # fall back to a progressively broader statistic — narrower-grain rates
    # cascade up to the global mean rather than failing the prediction.
    airline_rate = _lookup(lookups["airline"], {"AIRLINE": airline}, "AIRLINE_RATE", g_rate)
    airline_roll = _lookup(lookups["airline"], {"AIRLINE": airline}, "AIRLINE_ROLL7_DELAY", g_delay)
    origin_rate = _lookup(lookups["origin"], {"ORIGIN": origin}, "ORIGIN_RATE", g_rate)
    origin_roll = _lookup(lookups["origin"], {"ORIGIN": origin}, "ORIGIN_AIRPORT_ROLL7_DELAY", g_delay)
    origin_cong = _lookup(lookups["origin"], {"ORIGIN": origin}, "ORIGIN_CONGESTION", defaults["ORIGIN_CONGESTION_global"])
    dest_rate = _lookup(lookups["dest"], {"DEST": dest}, "DEST_RATE", g_rate)
    dest_cong = _lookup(lookups["dest"], {"DEST": dest}, "DEST_CONGESTION", defaults["DEST_CONGESTION_global"])
    route_rate = _lookup(lookups["route"], {"ORIGIN": origin, "DEST": dest}, "ROUTE_RATE", origin_rate)
    route_season_rate = _lookup(
        lookups["route_season"], {"ORIGIN": origin, "DEST": dest, "SEASON": season},
        "ROUTE_SEASON_RATE", route_rate,
    )
    airline_month_rate = _lookup(
        lookups["airline_month"], {"AIRLINE": airline, "MONTH": month},
        "AIRLINE_MONTH_RATE", airline_rate,
    )
    origin_hour_rate = _lookup(
        lookups["origin_hour"], {"ORIGIN": origin, "HOUR": hour},
        "ORIGIN_HOUR_RATE", origin_rate,
    )

    # The weather preset expands one user choice into all 10 weather features
    # consistently, avoiding an incoherent mix the user couldn't be expected
    # to specify (e.g. "Heavy rain" but with sub-zero temperature).
    presets = lookups["weather"]["presets"]
    if weather_label.startswith("Auto") or weather_label not in presets:
        w = lookups["weather"]["seasonal_auto"][str(season)]
    else:
        w = presets[weather_label]

    row = {
        # Temporal
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "HOUR": hour,
        "IS_WEEKEND": is_weekend,
        "SEASON": season,
        # Distance
        "LOG_DISTANCE": log_distance,
        # Weather
        "temp": w["temp"],
        "precip": w["precip"],
        "snowfall": w["snowfall"],
        "wind_speed": w["wind_speed"],
        "wind_gusts": w["wind_gusts"],
        "weather_code": w["weather_code"],
        "IS_FREEZING": w["IS_FREEZING"],
        "IS_RAINING": w["IS_RAINING"],
        "IS_SNOWING": w["IS_SNOWING"],
        "IS_ADVERSE_WEATHER": w["IS_ADVERSE_WEATHER"],
        # Operational
        "ORIGIN_CONGESTION": origin_cong,
        "DEST_CONGESTION": dest_cong,
        "TURNAROUND_MIN": defaults["TURNAROUND_MIN"],
        "PREV_LEG_DEP_DELAY": defaults["PREV_LEG_DEP_DELAY"],
        "PREV_LEG_DELAYED": defaults["PREV_LEG_DELAYED"],
        "PREV_LEG_ARR_DELAY": defaults["PREV_LEG_ARR_DELAY"],
        # Rolling histories
        "AIRLINE_ROLL7_DELAY": airline_roll,
        "ORIGIN_AIRPORT_ROLL7_DELAY": origin_roll,
        # Rates
        "AIRLINE_RATE": airline_rate,
        "ORIGIN_RATE": origin_rate,
        "DEST_RATE": dest_rate,
        "ROUTE_RATE": route_rate,
        "ROUTE_SEASON_RATE": route_season_rate,
        "AIRLINE_MONTH_RATE": airline_month_rate,
        "ORIGIN_HOUR_RATE": origin_hour_rate,
    }
    df = pd.DataFrame([row])
    # pytorch-tabular requires categorical columns as strings at predict time.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
    return df


def predict_delay(row_df: pd.DataFrame, model, threshold: float) -> tuple[float, bool]:
    """Run the FTT forward pass and apply the locked threshold.

    Returns (probability, is_delayed) where is_delayed is the binary verdict.
    """
    preds = model.predict(row_df)
    prob = float(preds["TARGET_1_probability"].iloc[0])
    return prob, prob >= threshold
