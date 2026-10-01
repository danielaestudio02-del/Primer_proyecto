"""Prophet + LightGBM ensemble with a recent-level adjustment.

The same class is used by the backtest and by the production pipeline, and the
fitted object is what gets stored in the joblib artifact. Prophet models are
kept as JSON strings so the artifact does not depend on pickling Stan objects.

Every feature for a (station, origin, target) row only reads observations at or
before the origin, so the model can be scored on many origins at once without
looking into the future.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

TZ = "America/Bogota"
STEP = pd.Timedelta(minutes=15)
HORIZONS = (1, 2, 3, 4)

FEATURES = [
    "station_code", "horizon",
    # Calendar of the target.
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "min_to_am_peak", "min_to_pm_peak",
    # Recent history at the origin.
    "lag_0", "lag_3", "lag_95", "lag_671", "rolling_4", "rolling_96", "d1", "d2", "accel", "peak_x_slope",
    # Same slot yesterday / last week and averaged seasonal profiles.
    "daily", "weekly", "weekly_mean2", "weekly_mean4", "daytype_mean",
    # How the origin compares with its own past.
    "ratio_vs_week", "delta_vs_week", "weekly_change", "daily_change", "seasonal_persistence",
    "weekly_scaled", "origin_vs_profile", "profile_persistence",
]

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def _quiet_prophet() -> None:
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").disabled = True


def _prophet_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    # Bogotá has no daylight saving time, so naive local time is unambiguous.
    ds = times.tz_convert(TZ).tz_localize(None)
    return pd.DataFrame({"ds": ds, "weekday": ds.dayofweek < 5})


def _series_lookup(history: pd.DataFrame) -> pd.Series:
    frame = history[["station_id", "observed_at", "demand"]].copy()
    frame["station_id"] = frame["station_id"].astype(str)
    return frame.set_index(["station_id", "observed_at"])["demand"].astype(float).sort_index()


def build_features(history: pd.DataFrame, rows: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """Add model features to rows with station_id, origin, target_at (UTC) and horizon."""
    lookup = _series_lookup(history)
    out = rows.copy()
    out["station_id"] = out["station_id"].astype(str)
    station_ids = out["station_id"].to_numpy()

    def at(base: str, steps: int) -> np.ndarray:
        times = pd.DatetimeIndex(out[base]) + steps * STEP
        index = pd.MultiIndex.from_arrays([station_ids, times])
        return lookup.reindex(index).to_numpy(dtype=float)

    def nanmean(arrays: list[np.ndarray]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN
            return np.nanmean(np.vstack(arrays), axis=0)

    codes = {station: code for code, station in enumerate(stations)}
    out["station_code"] = [codes.get(station, -1) for station in station_ids]

    local = pd.DatetimeIndex(out["target_at"]).tz_convert(TZ)
    minutes = local.hour * 60 + local.minute
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["min_to_am_peak"] = minutes - (7 * 60 + 30)
    out["min_to_pm_peak"] = minutes - (17 * 60 + 30)

    y0, y1, y2 = at("origin", 0), at("origin", -1), at("origin", -2)
    week_origin = [at("origin", -672 * k) for k in (1, 2, 3, 4)]
    out["lag_0"] = y0
    out["lag_3"] = at("origin", -3)
    out["lag_95"] = at("origin", -95)
    out["lag_671"] = at("origin", -671)
    out["rolling_4"] = nanmean([at("origin", -k) for k in range(4)])
    out["rolling_96"] = nanmean([at("origin", -k) for k in range(96)])
    out["d1"] = y0 - y1
    out["d2"] = y1 - y2
    out["accel"] = out["d1"] - out["d2"]
    in_peak = (np.abs(out["min_to_am_peak"]) < 120) | (np.abs(out["min_to_pm_peak"]) < 120)
    out["peak_x_slope"] = in_peak * out["d1"]

    week_target = [at("target_at", -672 * k) for k in (1, 2, 3, 4)]
    out["daily"] = at("target_at", -96)
    out["weekly"] = week_target[0]
    out["weekly_mean2"] = nanmean(week_target[:2])
    out["weekly_mean4"] = nanmean(week_target)
    same_type = []
    for k in range(1, 8):
        value = at("target_at", -96 * k)
        past_weekend = (local - pd.Timedelta(days=k)).dayofweek >= 5
        same_type.append(np.where(past_weekend == (local.dayofweek >= 5), value, np.nan))
    out["daytype_mean"] = nanmean(same_type)

    out["ratio_vs_week"] = y0 / np.maximum(week_origin[0], 1)
    out["delta_vs_week"] = y0 - week_origin[0]
    out["weekly_change"] = out["weekly"] - week_origin[0]
    out["daily_change"] = out["daily"] - at("origin", -96)
    out["seasonal_persistence"] = y0 + out["weekly_change"]
    out["weekly_scaled"] = out["weekly"] * np.clip(out["ratio_vs_week"], 0.5, 2)
    out["origin_vs_profile"] = y0 - nanmean(week_origin)
    out["profile_persistence"] = out["weekly_mean4"] + out["origin_vs_profile"]
    return out


def training_rows(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Hourly origins x 4 horizons whose target is observed at or before the cutoff."""
    frame = history.loc[history["observed_at"] <= cutoff, ["station_id", "observed_at", "demand"]].copy()
    frame["station_id"] = frame["station_id"].astype(str)
    start = frame.groupby("station_id")["observed_at"].min()
    local = frame["observed_at"].dt.tz_convert(TZ)
    origins = frame.loc[
        (local.dt.minute == 0)
        & (frame["observed_at"] >= frame["station_id"].map(start) + pd.Timedelta(days=7)),
        ["station_id", "observed_at"],
    ].rename(columns={"observed_at": "origin"})
    rows = pd.concat(
        [origins.assign(horizon=h, target_at=origins["origin"] + h * STEP) for h in HORIZONS],
        ignore_index=True,
    )
    lookup = _series_lookup(frame)
    rows["y"] = lookup.reindex(pd.MultiIndex.from_arrays([rows["station_id"], rows["target_at"]])).to_numpy()
    return rows.loc[rows["y"].notna() & (rows["target_at"] <= cutoff)].reset_index(drop=True)


# Components blended by the adaptive mode. prophet_slot (Prophet scaled by the level
# observed around the same slot yesterday) was added for peak-shape drift; models saved
# before it keep their original four components (see _components).
ADAPTIVE_COMPONENTS = ("prophet", "lightgbm", "daily_lvl", "persist", "prophet_slot")
LEGACY_ADAPTIVE_COMPONENTS = ("prophet", "lightgbm", "daily_lvl", "persist")


class PulsoEnsemble:
    """Prophet (level-adjusted, one per station) + LightGBM.

    Fixed mode: 65% Prophet + 35% LightGBM. Adaptive mode (default for new models):
    for every station and origin, each component (Prophet, LightGBM, same slot
    yesterday scaled to today's level, persistence and Prophet scaled by the level
    observed around the same slot yesterday) is scored on the previous
    `adaptive_lookback_h` hours using only targets already observed at the origin
    (errors weighted by exp(-age / adaptive_decay_h), so the last hours count most),
    and the components are blended with weights proportional to 1 / WAPE^2.
    """

    def __init__(
        self,
        prophet_weight: float = 0.65,
        level_window: int = 8,
        level_clip: tuple[float, float] = (0.5, 2.0),
        lgbm_params: dict | None = None,
        adaptive: bool = True,
        adaptive_lookback_h: int = 24,
        adaptive_min_points: int = 16,
        adaptive_decay_h: float | None = 3.0,
    ) -> None:
        self.adaptive = adaptive
        self.adaptive_lookback_h = adaptive_lookback_h
        self.adaptive_min_points = adaptive_min_points
        self.adaptive_components = ADAPTIVE_COMPONENTS
        self.adaptive_decay_h = adaptive_decay_h
        self.prophet_weight = prophet_weight
        self.level_window = level_window
        self.level_clip = level_clip
        self.lgbm_params = dict(DEFAULT_LGBM_PARAMS if lgbm_params is None else lgbm_params)
        self.prophet_json: dict[str, str] = {}
        self.stations: list[str] = []
        self.lgbm = None
        self.trained_until: pd.Timestamp | None = None
        self._prophet_cache: dict[str, object] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_prophet_cache"] = {}
        return state

    def get_config(self) -> dict:
        return {
            "prophet_weight": self.prophet_weight,
            "level_window_steps": self.level_window,
            "level_clip": list(self.level_clip),
            "prophet": {"daily_seasonality": 25, "weekly_seasonality": 10, "weekday_daily_fourier": 20,
                        "changepoint_prior_scale": 0.05},
            "lightgbm": self.lgbm_params,
            "adaptive": getattr(self, "adaptive", False),
            "adaptive_lookback_h": getattr(self, "adaptive_lookback_h", None),
            "adaptive_components": list(self._components()),
            "adaptive_decay_h": getattr(self, "adaptive_decay_h", None),
        }

    # ----- training -------------------------------------------------------
    def fit(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> "PulsoEnsemble":
        import lightgbm as lgb
        from prophet import Prophet
        from prophet.serialize import model_to_json

        _quiet_prophet()
        history = history.loc[history["observed_at"] <= cutoff].copy()
        history["station_id"] = history["station_id"].astype(str)
        self.stations = sorted(history["station_id"].unique())
        self.prophet_json, self._prophet_cache = {}, {}
        for station in self.stations:
            series = history.loc[history["station_id"] == station].sort_values("observed_at")
            frame = _prophet_frame(pd.DatetimeIndex(series["observed_at"]))
            frame["y"] = series["demand"].to_numpy(dtype=float)
            model = Prophet(
                daily_seasonality=25,
                weekly_seasonality=10,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            model.add_seasonality("weekday_daily", period=1, fourier_order=20, condition_name="weekday")
            model.fit(frame)
            self.prophet_json[station] = model_to_json(model)

        rows = training_rows(history, cutoff)
        if rows.empty:
            raise ValueError("No hay ejemplos para entrenar LightGBM.")
        features = build_features(history, rows, self.stations)
        self.lgbm = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm.fit(features[FEATURES], features["y"])
        self.trained_until = pd.Timestamp(cutoff)
        return self

    # ----- inference ------------------------------------------------------
    def _components(self) -> tuple[str, ...]:
        return tuple(getattr(self, "adaptive_components", LEGACY_ADAPTIVE_COMPONENTS))

    def _prophet(self, station: str):
        if station not in self._prophet_cache:
            from prophet.serialize import model_from_json

            _quiet_prophet()
            self._prophet_cache[station] = model_from_json(self.prophet_json[station])
        return self._prophet_cache[station]

    def _prophet_curve(self, station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        times = pd.date_range(start, end, freq=STEP)
        yhat = self._prophet(station).predict(_prophet_frame(times))["yhat"].to_numpy()
        return pd.Series(yhat, index=times)

    def _level_ratio(self, observed: pd.Series, curve: pd.Series, window: int | None = None) -> pd.Series:
        """Observed / expected demand over the last `window` slots (default level_window), per timestamp."""
        window = window or self.level_window
        y = observed.reindex(curve.index)
        present = y.notna()
        num = y.fillna(0).rolling(window, min_periods=1).sum()
        den = curve.where(present, 0).clip(lower=0).rolling(window, min_periods=1).sum()
        ratio = (num / den.where(den > 0)).fillna(1.0)
        return ratio.clip(*self.level_clip)

    def predict_components(self, history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict rows (station_id, origin, target_at, horizon); returns every component."""
        if self.lgbm is None:
            raise ValueError("El modelo no está entrenado.")
        rows = rows.copy()
        rows["station_id"] = rows["station_id"].astype(str)
        unknown = set(rows["station_id"]) - set(self.stations)
        if unknown:
            raise ValueError(f"Estaciones sin modelo: {sorted(unknown)}")
        features = build_features(history, rows, self.stations)
        lgbm_pred = np.maximum(0, self.lgbm.predict(features[FEATURES]))

        prophet_pred = np.full(len(rows), np.nan)
        slot_pred = np.full(len(rows), np.nan)
        level = np.full(len(rows), np.nan)
        lookup = _series_lookup(history)
        with_slot = "prophet_slot" in self._components()
        for station, index in rows.groupby("station_id").groups.items():
            subset = rows.loc[index]
            start = subset["origin"].min() - (self.level_window - 1) * STEP
            if with_slot:
                start = min(start, subset["target_at"].min() - pd.Timedelta(days=1) - self.level_window * STEP)
            curve = self._prophet_curve(station, start, subset["target_at"].max())
            observed = lookup.xs(station, level="station_id") if station in lookup.index.get_level_values(0) else pd.Series(dtype=float)
            observed = observed[observed.index <= subset["origin"].max()]
            ratio = self._level_ratio(observed, curve)
            positions = rows.index.get_indexer(index)
            level[positions] = ratio.reindex(subset["origin"]).to_numpy()
            prophet_pred[positions] = np.maximum(0, curve.reindex(subset["target_at"]).to_numpy()) * level[positions]
            if with_slot:
                # Level of the window centred on the same slot yesterday (already observed at the
                # origin): a peak that grew or shrank yesterday is corrected from its first slot,
                # instead of waiting for the last 2 h to fill with peak data.
                anchor = subset["target_at"] - pd.Timedelta(days=1) + (self.level_window // 2) * STEP
                slot_ratio = ratio.reindex(anchor).fillna(1.0).to_numpy()
                slot_pred[positions] = np.maximum(0, curve.reindex(subset["target_at"]).to_numpy()) * slot_ratio
        ensemble = self.prophet_weight * prophet_pred + (1 - self.prophet_weight) * lgbm_pred

        # Simple recent references used by the adaptive blend.
        def at(times: pd.DatetimeIndex) -> np.ndarray:
            idx = pd.MultiIndex.from_arrays([rows["station_id"].to_numpy(), times])
            return lookup.reindex(idx).to_numpy(dtype=float)

        origin = pd.DatetimeIndex(rows["origin"])
        recent = sum(np.nan_to_num(at(origin - k * STEP)) for k in range(self.level_window))
        yesterday = sum(np.nan_to_num(at(origin - (96 + k) * STEP)) for k in range(self.level_window))
        day_ratio = np.clip(np.where(yesterday > 0, recent / np.maximum(yesterday, 1), 1.0), *self.level_clip)
        daily_lvl = np.nan_to_num(at(pd.DatetimeIndex(rows["target_at"]) - pd.Timedelta(days=1))) * day_ratio
        persist = np.nan_to_num(at(origin))

        return pd.DataFrame(
            {
                "prophet": prophet_pred,
                "lightgbm": lgbm_pred,
                "daily_lvl": daily_lvl,
                "persist": persist,
                "prophet_slot": slot_pred,
                "level_ratio": level,
                "fixed": np.maximum(0, ensemble),
                "prediction": np.maximum(0, ensemble),
            },
            index=rows.index,
        )

    def adaptive_weights(self, history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Blend weights per (station, origin) from each component's WAPE on the previous
        `adaptive_lookback_h` hourly origins whose targets were observed by the origin."""
        lookback = int(getattr(self, "adaptive_lookback_h", 24))
        components = self._components()
        keys = rows[["station_id", "origin"]].astype({"station_id": str}).drop_duplicates()
        pairs = keys.loc[keys.index.repeat(lookback)].reset_index(drop=True)
        pairs["shadow_origin"] = pairs["origin"] - pd.to_timedelta(np.tile(np.arange(1, lookback + 1), len(keys)), unit="h")
        shadow = pairs[["station_id", "shadow_origin"]].drop_duplicates().rename(columns={"shadow_origin": "origin"})
        shadow = shadow.loc[shadow.index.repeat(4)].reset_index(drop=True)
        shadow["horizon"] = np.tile([1, 2, 3, 4], len(shadow) // 4)
        shadow["target_at"] = shadow["origin"] + shadow["horizon"] * STEP
        lookup = _series_lookup(history)
        shadow["y"] = lookup.reindex(pd.MultiIndex.from_arrays([shadow["station_id"], shadow["target_at"]])).to_numpy(float)
        shadow = shadow[shadow["y"].notna()].reset_index(drop=True)
        empty = pd.DataFrame(columns=["station_id", "origin", "points", *[f"w_{k}" for k in components]])
        if shadow.empty:
            return empty
        comp = self.predict_components(history, shadow)
        for k in components:
            shadow[f"e_{k}"] = (shadow["y"] - comp[k].to_numpy()).abs()
        joined = pairs.merge(shadow.rename(columns={"origin": "shadow_origin"}), on=["station_id", "shadow_origin"])
        joined = joined[joined["target_at"] <= joined["origin"]].copy()
        decay = getattr(self, "adaptive_decay_h", None)
        if decay:
            # Recent errors count more: a drift phase lasts a few hours, so the blend must
            # follow it before the 24 h window fills with the new regime.
            age_h = (joined["origin"] - joined["shadow_origin"]) / pd.Timedelta(hours=1)
            recency = np.exp(-(age_h - 1) / decay)
            joined["y"] = joined["y"] * recency
            for k in components:
                joined[f"e_{k}"] = joined[f"e_{k}"] * recency
        agg = joined.groupby(["station_id", "origin"]).agg(
            points=("y", "size"), y=("y", "sum"), **{f"e_{k}": (f"e_{k}", "sum") for k in components}
        ).reset_index()
        inv = pd.DataFrame({
            k: 1 / np.maximum(agg[f"e_{k}"] / agg["y"].clip(lower=1), 0.02) ** 2 for k in components
        })
        weights = inv.div(inv.sum(axis=1), axis=0)
        for k in components:
            agg[f"w_{k}"] = weights[k]
        return agg[["station_id", "origin", "points", *[f"w_{k}" for k in components]]]

    def predict_detailed(self, history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Components, blend weights and final prediction for each row."""
        comp = self.predict_components(history, rows)
        if not getattr(self, "adaptive", False):
            return comp
        components = self._components()
        keyed = rows[["station_id", "origin"]].astype({"station_id": str})
        weights = keyed.merge(self.adaptive_weights(history, rows), on=["station_id", "origin"], how="left")
        weights.index = rows.index
        ok = weights["points"].fillna(0) >= getattr(self, "adaptive_min_points", 16)
        blend = sum(weights[f"w_{k}"].fillna(0) * comp[k] for k in components)
        comp["prediction"] = np.maximum(0, np.where(ok, blend, comp["fixed"]))
        for k in components:
            comp[f"w_{k}"] = np.where(ok, weights[f"w_{k}"], np.nan)
        return comp

    def predict(self, history: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        return self.predict_detailed(history, rows)["prediction"].to_numpy()

    def level_ratios(
        self, history: pd.DataFrame, cutoff: pd.Timestamp, window: int, lookbacks: int
    ) -> pd.DataFrame:
        """Observed/expected level per station over `lookbacks` non-overlapping windows of
        `window` slots ending at the cutoff (newest first); used for drift alerts."""
        lookup = _series_lookup(history)
        records = []
        for station in self.stations:
            start = cutoff - (window * lookbacks - 1) * STEP
            curve = self._prophet_curve(station, start, cutoff)
            observed = lookup.xs(station, level="station_id") if station in lookup.index.get_level_values(0) else pd.Series(dtype=float)
            ratio = self._level_ratio(observed[observed.index <= cutoff], curve, window)
            for k in range(lookbacks):
                at = cutoff - k * window * STEP
                records.append({"station_id": station, "window_end": at, "ratio": float(ratio.get(at, np.nan))})
        return pd.DataFrame(records)
