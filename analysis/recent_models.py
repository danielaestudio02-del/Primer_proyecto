"""Predictores que se ajustan solo con las últimas horas, en cada origen.

En la fase de drift la demanda dejó de seguir el perfil diario/semanal: cada
estación oscila en ciclos de pocas horas y sigue a otras estaciones con rezago.
Estos predictores se reajustan en cada origen con una ventana corta, usando
solo observaciones disponibles hasta ese origen.

- own_ar: por estación y horizonte, regresión ridge sobre sus propios 8 rezagos.
- cross_ar: por estación y horizonte, ridge sobre los 8 rezagos de las 12 estaciones.
- pooled_ar: un modelo por horizonte para todas las estaciones, con series
  normalizadas por su nivel reciente (aprende la forma de la ola).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = 8


def wide_matrix(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.assign(station_id=history["station_id"].astype(str))
    wide = frame.pivot(index="observed_at", columns="station_id", values="demand").sort_index().astype(float)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="15min")
    return wide.reindex(full).ffill()


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    ym = y.mean()
    a = z.T @ z + alpha * len(z) * np.eye(z.shape[1])
    w = np.linalg.solve(a, z.T @ (y - ym))
    return w, mu, ym, sd


def _apply(model, x: np.ndarray) -> np.ndarray:
    w, mu, ym, sd = model
    return ((x - mu) / sd) @ w + ym


def predict_recent(
    wide: pd.DataFrame, origin: pd.Timestamp, horizons=(1, 2, 3, 4), window_h: int = 24,
    kinds=("own_ar", "cross_ar", "pooled_ar"), alpha: float = 0.05,
) -> dict[str, dict[tuple[str, int], float]]:
    """Predicciones {kind: {(station, horizon): value}} para un origen; solo usa datos <= origin."""
    values = wide.loc[:origin].to_numpy()
    p = len(values) - 1
    stations = list(wide.columns)
    n_win = window_h * 4
    out: dict[str, dict[tuple[str, int], float]] = {k: {} for k in kinds}
    if p < n_win + LAGS + 4:
        return out
    lagmat = np.stack([values[p - n_win - LAGS - 4 + 1 - j: p + 1 - j] for j in range(LAGS)], axis=-1)
    # lagmat[i, s, j] = value at (start + i - j) for station s; start = p - n_win - LAGS - 4 + 1
    start = p - n_win - LAGS - 4 + 1
    idx = np.arange(start, p + 1)
    for h in horizons:
        q = idx[(idx + h <= p) & (idx >= p - n_win)]  # training origins whose target is observed
        rows = q - start
        ytr = values[q + h]                            # (n, S)
        now = lagmat[-1]                               # (S, LAGS) at origin p
        if "own_ar" in kinds:
            for s, st in enumerate(stations):
                m = _ridge(lagmat[rows, s, :], ytr[:, s], alpha)
                out["own_ar"][(st, h)] = float(max(0.0, _apply(m, now[s][None, :])[0]))
        if "cross_ar" in kinds:
            xtr = lagmat[rows].reshape(len(rows), -1)
            xnow = now.reshape(1, -1)
            for s, st in enumerate(stations):
                m = _ridge(xtr, ytr[:, s], alpha * 4)
                out["cross_ar"][(st, h)] = float(max(0.0, _apply(m, xnow)[0]))
        if "pooled_ar" in kinds:
            scale_tr = np.maximum(lagmat[rows].mean(-1), 1.0)          # (n, S) nivel de las últimas 2 h
            xtr = (lagmat[rows] / scale_tr[..., None]).reshape(-1, LAGS)
            ytr_n = (ytr / scale_tr).reshape(-1)
            m = _ridge(xtr, ytr_n, alpha)
            scale_now = np.maximum(now.mean(-1), 1.0)
            pred = _apply(m, now / scale_now[:, None]) * scale_now
            for s, st in enumerate(stations):
                out["pooled_ar"][(st, h)] = float(max(0.0, pred[s]))
    return out
