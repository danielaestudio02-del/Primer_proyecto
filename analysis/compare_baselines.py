"""Compare leakage-safe forecasting baselines on the starter dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from forecasting_core import (
    FREQUENCY_MINUTES,
    HORIZONS,
    TZ,
    build_examples,
    score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "starter"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
def load_local_env() -> None:
    """Read local credentials without printing their values."""
    env_path = ROOT / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if name.strip() and value:
            os.environ.setdefault(name.strip(), value)


def load_observations(source: str) -> pd.DataFrame:
    if source == "local":
        frame = pd.read_csv(
            DATA / "observations.csv",
            parse_dates=["observed_at"],
            dtype={"station_id": "string"},
        )
    else:
        load_local_env()
        url = os.environ.get("SUPABASE_URL")
        secret = os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not secret:
            raise SystemExit(
                "Faltan SUPABASE_URL o SUPABASE_SECRET_KEY en .env.local; valores ocultos."
            )
        from supabase import create_client

        client = create_client(url, secret)
        page_size = 1000
        rows: list[dict] = []
        try:
            start = 0
            while True:
                response = (
                    client.table("observations")
                    .select("station_id,observed_at,demand")
                    .order("station_id")
                    .order("observed_at")
                    .range(start, start + page_size - 1)
                    .execute()
                )
                page = response.data or []
                rows.extend(page)
                if len(page) < page_size:
                    break
                start += page_size
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f"Lectura de Supabase interrumpida ({type(exc).__name__}); valores ocultos."
            ) from None
        if not rows:
            raise SystemExit("Supabase devolvió cero observaciones; benchmark cancelado.")
        frame = pd.DataFrame.from_records(rows)
        frame["station_id"] = frame["station_id"].astype("string")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)

    if frame.duplicated(["station_id", "observed_at"]).any():
        raise SystemExit("Hay claves station_id/observed_at duplicadas; benchmark cancelado.")
    if frame["demand"].isna().any() or frame["demand"].lt(0).any():
        raise SystemExit("La demanda contiene valores inválidos; benchmark cancelado.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara baselines con corte temporal.")
    parser.add_argument(
        "--source",
        choices=("local", "supabase"),
        default="local",
        help="Origen de observaciones (predeterminado: local).",
    )
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    observations = load_observations(args.source)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    end = observations["observed_at"].max()
    split = end.normalize() - pd.Timedelta(days=6)
    # Align the split to local midnight, giving exactly seven full local days.
    local_end = end.tz_convert(TZ)
    split_local = (local_end.normalize() - pd.Timedelta(days=6)).tz_localize(None)
    split = split_local.tz_localize(TZ).tz_convert("UTC")
    all_times = pd.DatetimeIndex(sorted(observations["observed_at"].unique()))
    local_times = all_times.tz_convert(TZ)
    hourly_origins = all_times[local_times.minute == 0]
    train_origins = hourly_origins[hourly_origins < split]
    test_origins = hourly_origins[hourly_origins >= split]
    train = build_examples(observations, train_origins)
    test = build_examples(observations, test_origins)
    train = train.loc[train["target_at"] < split].copy()

    categorical = ["station_id", "horizon"]
    numeric = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "lag_1", "lag_4",
        "lag_96", "lag_672", "rolling_4", "rolling_96",
    ]
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", "passthrough", numeric),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[categorical + numeric], train["y"])
    test["random_forest"] = np.maximum(0, model.predict(test[categorical + numeric]))

    model_columns = {
        "Persistencia": "persistence",
        "Naive estacional diario": "daily",
        "Naive estacional semanal": "weekly",
        "Random Forest global": "random_forest",
    }
    summaries: dict[str, dict] = {}
    horizon_scores: dict[str, pd.DataFrame] = {}
    for name, column in model_columns.items():
        summaries[name], horizon_scores[name] = score(test, column)
    comparison = pd.DataFrame(summaries).T.sort_values("official_accuracy", ascending=False)
    comparison.index.name = "Modelo"
    comparison.columns = ["Accuracy oficial (%)", "WAPE medio por estación", "MAE"]

    by_horizon = pd.concat(
        {name: frame["accuracy"] for name, frame in horizon_scores.items()}, axis=1
    )
    by_horizon.index = [f"+{int(h) * FREQUENCY_MINUTES} min" for h in by_horizon.index]
    by_horizon.index.name = "Horizonte"

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison["Accuracy oficial (%)"].sort_values().plot.barh(ax=ax, color="#257a83")
    ax.set(title="Validación temporal: accuracy oficial por modelo", xlabel="Accuracy (%)", ylabel="")
    ax.set_xlim(left=0, right=100)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparacion_modelos_accuracy.png", dpi=150)
    plt.close(fig)

    report = f"""# Comparación inicial de modelos — Pulso TransMi

## Diseño de evaluación

- Entrenamiento: orígenes horarios anteriores al corte temporal; solo se conservan ejemplos cuyos targets también preceden al corte.
- Validación: últimos siete días completos disponibles en el starter, con orígenes horarios y horizontes de +15, +30, +45 y +60 minutos.
- Ejemplos de entrenamiento: {len(train):,}; ejemplos de validación: {len(test):,}.
- Corte temporal: `{split.tz_convert(TZ)}`; fin del histórico: `{end.tz_convert(TZ)}`.
- Random Forest global: una sola estimación conjunta con identificador de estación, horizonte, calendario, rezagos hasta el origen y medias móviles pasadas.
- No se usó contexto meteorológico porque en el corte está marcado como observado y no se estableció que sus valores futuros estuvieran disponibles al emitir el pronóstico.

## Resultados agregados

La accuracy oficial se calcula con WAPE por estación y luego se promedia sin ponderar entre estaciones. Un mayor accuracy y menor MAE/WAPE indican mejor resultado.

{_markdown(comparison.round(3))}

![Comparación de accuracy](figures/comparacion_modelos_accuracy.png)

## Accuracy por horizonte

{_markdown(by_horizon.round(2))}

## Lectura

Estos resultados son una primera referencia local sobre datos sintéticos. El vencedor de este corte no debe declararse champion todavía: conviene repetir backtesting en varios cortes temporales, revisar variación por estación y horizonte, y contrastar contra los ciclos oficiales revelados por la API.

El baseline diario usa el valor de la misma hora del día anterior; el semanal usa el valor de la misma hora y día de la semana anterior. Ambos solo consultan observaciones disponibles a la hora de origen del pronóstico.
"""
    (REPORTS / "model_baselines_v1.md").write_text(report, encoding="utf-8")
    print(f"Origen de observaciones: {args.source}")
    print(f"Corte: {split.tz_convert(TZ)}")
    print(f"Filas train/test: {len(train):,}/{len(test):,}")
    print(comparison.round(2).to_string())
    print(f"Reporte: {REPORTS / 'model_baselines_v1.md'}")


def _markdown(frame: pd.DataFrame) -> str:
    headers = [str(frame.index.name or "Modelo")] + [str(column) for column in frame.columns]
    rows = [headers]
    for index, row in frame.iterrows():
        rows.append([str(index), *[str(value) for value in row.tolist()]])
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(rows[0])) + " |"]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
