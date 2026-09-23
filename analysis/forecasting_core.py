"""Dependency-light shared forecasting features and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/Bogota"
HORIZONS = (1, 2, 3, 4)
FREQUENCY_MINUTES = 15


def build_examples(series: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict] = []
    for station, frame in series.groupby("station_id", sort=True, observed=True):
        frame = frame.sort_values("observed_at").set_index("observed_at")
        demand = frame["demand"]
        for origin in origins:
            if origin not in demand.index:
                continue
            hist = demand.loc[:origin]
            if len(hist) < 672:
                continue
            y0 = float(hist.iloc[-1])
            lag4 = float(hist.iloc[-4])
            lag96 = float(hist.iloc[-96])
            lag672 = float(hist.iloc[-672])
            roll4 = float(hist.iloc[-4:].mean())
            roll96 = float(hist.iloc[-96:].mean())
            for horizon in HORIZONS:
                target = origin + pd.Timedelta(minutes=horizon * FREQUENCY_MINUTES)
                if target not in demand.index:
                    continue
                local_target = target.tz_convert(TZ)
                rows.append(
                    {
                        "station_id": station,
                        "horizon": horizon,
                        "origin": origin,
                        "target_at": target,
                        "y": float(demand.loc[target]),
                        "persistence": y0,
                        "daily": float(demand.loc[target - pd.Timedelta(days=1)]),
                        "weekly": float(demand.loc[target - pd.Timedelta(days=7)]),
                        "lag_1": y0,
                        "lag_4": lag4,
                        "lag_96": lag96,
                        "lag_672": lag672,
                        "rolling_4": roll4,
                        "rolling_96": roll96,
                        "hour_sin": np.sin(2 * np.pi * local_target.hour / 24),
                        "hour_cos": np.cos(2 * np.pi * local_target.hour / 24),
                        "dow_sin": np.sin(2 * np.pi * local_target.dayofweek / 7),
                        "dow_cos": np.cos(2 * np.pi * local_target.dayofweek / 7),
                    }
                )
    return pd.DataFrame(rows)


def score(predictions: pd.DataFrame, prediction_column: str) -> tuple[dict, pd.DataFrame]:
    result = predictions[["station_id", "horizon", "y", prediction_column]].copy()
    result["absolute_error"] = (result["y"] - result[prediction_column]).abs()
    station = result.groupby("station_id", observed=True).agg(
        absolute_error=("absolute_error", "sum"), actual=("y", "sum")
    )
    station["wape"] = station["absolute_error"] / station["actual"].clip(lower=1)
    station["accuracy"] = (100 * (1 - station["wape"])).clip(lower=0)
    horizon = result.groupby("horizon", observed=True).agg(
        absolute_error=("absolute_error", "sum"), actual=("y", "sum")
    )
    horizon["wape"] = horizon["absolute_error"] / horizon["actual"].clip(lower=1)
    horizon["accuracy"] = (100 * (1 - horizon["wape"])).clip(lower=0)
    return {
        "official_accuracy": float(station["accuracy"].mean()),
        "mean_station_wape": float(station["wape"].mean()),
        "mae": float(result["absolute_error"].mean()),
    }, horizon
