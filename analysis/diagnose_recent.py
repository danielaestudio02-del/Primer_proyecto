"""Diagnóstico con datos reales: cómo le habría ido a cada predictor en los últimos
ciclos horarios, sin información futura. Compara el champion (y sus componentes) con
predictores que solo usan las últimas horas, y con selectores que eligen por estación
el predictor que mejor acertó en las horas previas.

Uso: python analysis/diagnose_recent.py [horas_a_evaluar]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pulso_pipeline as pp
from ensemble_model import STEP

TZ = "America/Bogota"


def main() -> None:
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    client = pp.supabase_client()
    history = pp.load_observations(client)
    history["station_id"] = history["station_id"].astype(str)
    champion = pp.load_champion(client)
    model = champion["model"]
    train_end = pd.Timestamp(champion["training_data_end"])
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    last = history["observed_at"].max()
    print(f"Champion {champion['model_version']}; datos de entrenamiento hasta {train_end.tz_convert(TZ)}")
    print(f"Última observación: {last.tz_convert(TZ)}")

    # Hourly origins whose four targets are already observed; extra 6 h for the selectors.
    end = (last - 4 * STEP).floor("h")
    origins = pd.date_range(end - pd.Timedelta(hours=hours + 6 - 1), end, freq="h")
    stations = sorted(history["station_id"].unique())
    rows = pd.DataFrame(
        [(s, o, h) for o in origins for s in stations for h in (1, 2, 3, 4)],
        columns=["station_id", "origin", "horizon"],
    )
    rows["target_at"] = rows["origin"] + rows["horizon"] * STEP
    lookup = history.set_index(["station_id", "observed_at"])["demand"].astype(float).sort_index()

    def at(times) -> np.ndarray:
        return lookup.reindex(pd.MultiIndex.from_arrays([rows["station_id"], pd.DatetimeIndex(times)])).to_numpy()

    rows["y"] = at(rows["target_at"])
    det = model.predict_detailed(history, rows)
    for col in [c for c in ("prediction", "fixed", "prophet", "lightgbm", "daily_lvl", "persist", "prophet_slot") if c in det]:
        rows[col] = det[col].to_numpy()

    o = pd.DatetimeIndex(rows["origin"])
    lag = {k: np.nan_to_num(at(o - k * STEP)) for k in range(0, 12)}
    rows["mean_1h"] = sum(lag[k] for k in range(4)) / 4
    rows["mean_2h"] = sum(lag[k] for k in range(8)) / 8
    slope = (lag[0] + lag[1] - lag[4] - lag[5]) / 2 / 4  # per slot, smoothed
    rows["trend"] = np.maximum(0, lag[0] + slope * rows["horizon"])
    t = pd.DatetimeIndex(rows["target_at"])
    rows["daily"] = at(t - pd.Timedelta(days=1))
    rows["weekly"] = at(t - pd.Timedelta(days=7))
    recent = sum(lag[k] for k in range(4))
    yday = sum(np.nan_to_num(at(o - pd.Timedelta(days=1) - k * STEP)) for k in range(4))
    ratio = np.where(yday > 0, recent / np.maximum(yday, 1), 1.0)
    rows["daily_x_1h_noclip"] = np.nan_to_num(rows["daily"]) * ratio
    week = sum(np.nan_to_num(at(o - pd.Timedelta(days=7) - k * STEP)) for k in range(4))
    rows["weekly_x_1h_noclip"] = np.nan_to_num(rows["weekly"]) * np.where(week > 0, recent / np.maximum(week, 1), 1.0)

    # Submitted values, for a sanity check against what the competition scored.
    sub = pp.submitted_targets(client, history)
    if not sub.empty:
        sub = sub[["station_id", "origin", "target_at", "predicted_value"]].rename(columns={"predicted_value": "enviado"})
        rows = rows.merge(sub, on=["station_id", "origin", "target_at"], how="left")

    base = ["persist", "mean_1h", "mean_2h", "trend", "daily", "daily_x_1h_noclip", "weekly",
            "weekly_x_1h_noclip", "daily_lvl", "prophet", "lightgbm", "prediction"]
    base = [c for c in base if c in rows]

    # Selectors: per station and origin, the predictor with the lowest WAPE over the
    # previous K hourly origins (targets already observed at the origin).
    rows = rows.sort_values(["station_id", "origin", "horizon"]).reset_index(drop=True)
    for k_h in (2, 3, 6):
        chosen = np.full(len(rows), np.nan)
        blend = np.full(len(rows), np.nan)
        for station, g in rows.groupby("station_id"):
            for origin, idx in g.groupby("origin").groups.items():
                past = g[(g["origin"] >= origin - pd.Timedelta(hours=k_h)) & (g["target_at"] <= origin) & g["y"].notna()]
                if len(past) < 4:
                    continue
                err = {c: (past["y"] - past[c]).abs().sum() / max(past["y"].sum(), 1) for c in base}
                best = min(err, key=err.get)
                chosen[rows.index.get_indexer(idx)] = rows.loc[idx, best].to_numpy()
                inv = {c: 1 / max(e, 0.02) ** 2 for c, e in err.items()}
                tot = sum(inv.values())
                blend[rows.index.get_indexer(idx)] = sum(rows.loc[idx, c].fillna(0).to_numpy() * w / tot for c, w in inv.items())
        rows[f"sel_best_{k_h}h"] = chosen
        rows[f"sel_blend_{k_h}h"] = blend

    ev = rows[(rows["origin"] > origins[5]) & rows["y"].notna()]
    ev_out = ev[ev["origin"] > train_end]
    cols = [c for c in ["enviado", *base, "sel_best_2h", "sel_best_3h", "sel_best_6h",
                        "sel_blend_2h", "sel_blend_3h", "sel_blend_6h"] if c in ev]
    last6 = ev[ev["origin"] >= ev["origin"].max() - pd.Timedelta(hours=5)]
    table = pd.DataFrame({
        f"últimas {hours} h": {c: pp.official_accuracy(ev, c) for c in cols},
        "fuera de entrenamiento": {c: pp.official_accuracy(ev_out, c) for c in cols} if not ev_out.empty else {},
        "últimos 6 ciclos": {c: pp.official_accuracy(last6, c) for c in cols},
    }).round(2).sort_values("últimos 6 ciclos", ascending=False)
    print(f"\nOrígenes evaluados: {ev['origin'].min().tz_convert(TZ)} → {ev['origin'].max().tz_convert(TZ)}"
          f" ({ev['origin'].nunique()} ciclos; {ev_out['origin'].nunique()} posteriores al entrenamiento)")
    print(table.to_string())

    print("\nÚltimos 6 ciclos por estación (accuracy):")
    per = {}
    for c in ["enviado", "prediction", "persist", "mean_1h", "trend", "daily_x_1h_noclip", "sel_best_3h", "sel_blend_3h"]:
        if c in last6:
            g = last6.assign(ae=(last6["y"] - last6[c]).abs()).groupby("station_id").agg(ae=("ae", "sum"), y=("y", "sum"))
            per[c] = (100 * (1 - g["ae"] / g["y"].clip(lower=1))).clip(lower=0)
    print(pd.DataFrame(per).round(1).to_string())

    print("\nPor ciclo (últimas 12 h):")
    rec = ev[ev["origin"] >= ev["origin"].max() - pd.Timedelta(hours=11)]
    cyc = {}
    for origin, g in rec.groupby("origin"):
        cyc[origin.tz_convert(TZ).strftime("%d/%m %H:%M")] = {
            c: pp.official_accuracy(g, c) for c in ["enviado", "prediction", "persist", "trend", "daily_x_1h_noclip", "sel_best_3h", "sel_blend_3h"] if c in g
        }
    print(pd.DataFrame(cyc).T.round(1).to_string())


if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[2] == "explore"):
    main()


def explore() -> None:
    """Serie cruda reciente, rezagos, relación entre estaciones y endpoints de la API."""
    client = pp.supabase_client()
    h = pp.load_observations(client)
    h["station_id"] = h["station_id"].astype(str)
    wide = h.pivot(index="observed_at", columns="station_id", values="demand").sort_index().astype(float)
    last = wide.index.max()
    recent = wide[wide.index > last - pd.Timedelta(hours=24)]
    before = wide[(wide.index > last - pd.Timedelta(days=8)) & (wide.index <= last - pd.Timedelta(days=7))]
    print("\nSerie cruda, últimas 3 h (filas = intervalos de 15 min):")
    print(recent.tail(12).rename(index=lambda t: t.tz_convert(TZ).strftime("%d/%m %H:%M")).astype(int).to_string())
    print("\nMisma ventana hace 7 días:")
    print(before.tail(12).rename(index=lambda t: t.tz_convert(TZ).strftime("%d/%m %H:%M")).astype(int).to_string())
    print("\nMedia y coef. de variación por estación: últimas 24 h vs mismo día hace 7 días")
    def cv(f):
        return (f.diff().abs().mean() / f.mean().clip(lower=1)).round(2)
    print(pd.DataFrame({"media_24h": recent.mean().round(0), "media_semana_ant": before.mean().round(0),
                        "salto_medio_24h": cv(recent), "salto_medio_ant": cv(before)}).to_string())

    def acc(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        ok = ~np.isnan(y) & ~np.isnan(p)
        return max(0.0, 100 * (1 - np.abs(y[ok] - p[ok]).sum() / max(y[ok].sum(), 1)))

    print("\nAccuracy de y[t] ≈ y[t-k] (últimas 24 h), por rezago k en intervalos:")
    lags = {k: {s: round(acc(recent[s], wide[s].shift(k).reindex(recent.index)), 1) for s in wide} for k in (1, 2, 3, 4, 8, 96, 672)}
    print(pd.DataFrame(lags).to_string())

    print("\nMejor estación/rezago para explicar cada estación (y_s[t] ≈ c · y_o[t-k], k=1..8):")
    out = []
    for s in wide:
        best = (0, None, None)
        for o in wide:
            for k in range(1, 9):
                x = wide[o].shift(k).reindex(recent.index)
                c = recent[s].sum() / max(x.sum(), 1)
                a = acc(recent[s], c * x)
                if a > best[0]:
                    best = (a, o, k)
        out.append({"estación": s, "accuracy": round(best[0], 1), "explicada_por": best[1], "rezago": best[2],
                    "propia_k1": lags[1][s]})
    print(pd.DataFrame(out).to_string(index=False))

    print("\nEndpoints de la API (openapi):")
    try:
        r = pp.api_get("/openapi.json")
        paths = r.json().get("paths", {}) if r.status_code == 200 else {}
        for p, ops in paths.items():
            print(" ", p, sorted(ops))
    except Exception as exc:  # noqa: BLE001
        print("  no disponible:", type(exc).__name__)
    cyc = pp.api_get("/v1/forecast-cycles/current")
    print("\nCiclo actual (claves):", cyc.status_code, sorted(cyc.json().keys()) if cyc.headers.get("content-type", "").startswith("application/json") else "")
    try:
        body = cyc.json()
        extra = {k: v for k, v in body.items() if k not in ("targets",)}
        print(str(extra)[:1500])
        if body.get("targets"):
            print("target ejemplo:", body["targets"][0])
    except Exception:  # noqa: BLE001
        pass
    s = pp.api_get("/v1/stream/observations", {"limit": 2})
    print("\nStream ejemplo:", s.status_code, str(s.json())[:800] if s.status_code == 200 else "")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "explore":
    explore()
