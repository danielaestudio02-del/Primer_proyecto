"""Compare simple baselines, Random Forest, and XGBoost over weekly time folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from compare_baselines import (
    FIGURES,
    REPORTS,
    TZ,
    _markdown,
    build_examples,
    load_observations,
    score,
)


CATEGORICAL = ["station_id", "horizon"]
NUMERIC = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "lag_1", "lag_4",
    "lag_96", "lag_672", "rolling_4", "rolling_96",
]
FEATURES = CATEGORICAL + NUMERIC
MODEL_NAMES = {
    "Persistencia": "persistence",
    "Naive estacional diario": "daily",
    "Naive estacional semanal": "weekly",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ]
    )


def make_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor()),
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


def make_xgboost() -> Pipeline:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise SystemExit(
            "Falta XGBoost. Instálalo con: python -m pip install -r requirements-modeling.txt"
        ) from None
    return Pipeline(
        [
            ("features", make_preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=5.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def run_fold(observations: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    fold_end = split + pd.Timedelta(days=7)
    available_end = observations["observed_at"].max() + pd.Timedelta(minutes=15)
    fold_end = min(fold_end, available_end)

    all_times = pd.DatetimeIndex(sorted(observations["observed_at"].unique()))
    local_times = all_times.tz_convert(TZ)
    hourly_origins = all_times[local_times.minute == 0]
    train_origins = hourly_origins[hourly_origins < split]
    test_origins = hourly_origins[(hourly_origins >= split) & (hourly_origins < fold_end)]

    train = build_examples(observations, train_origins)
    train = train.loc[train["target_at"] < split].copy()
    test = build_examples(observations, test_origins)
    test = test.loc[(test["target_at"] >= split) & (test["target_at"] < fold_end)].copy()
    if train.empty or test.empty:
        raise SystemExit(f"El corte {split} no tiene suficientes datos de entrenamiento o validación.")

    for name, factory in (("random_forest", make_random_forest), ("xgboost", make_xgboost)):
        model = factory()
        model.fit(train[FEATURES], train["y"])
        test[name] = np.maximum(0, model.predict(test[FEATURES]))
        print(f"  {name}: entrenamiento y predicción completados ({len(test):,} filas validadas)")

    return test, {"start": split, "end": fold_end, "train_rows": len(train), "test_rows": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtesting temporal de baselines y modelos de árboles.")
    parser.add_argument("--source", choices=("local", "supabase"), default="supabase")
    parser.add_argument("--folds", type=int, default=3, help="Número de ventanas semanales, máximo 4.")
    args = parser.parse_args()
    if not 1 <= args.folds <= 4:
        raise SystemExit("--folds debe estar entre 1 y 4.")

    observations = load_observations(args.source)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    end_local = observations["observed_at"].max().tz_convert(TZ)
    most_recent_split = (end_local.normalize() - pd.Timedelta(days=6)).tz_localize(None)
    most_recent_split = most_recent_split.tz_localize(TZ).tz_convert("UTC")
    split_points = [
        most_recent_split - pd.Timedelta(days=7 * index)
        for index in reversed(range(args.folds))
    ]

    fold_frames: list[pd.DataFrame] = []
    fold_meta: list[dict] = []
    for index, split in enumerate(split_points, start=1):
        print(f"Fold {index}/{len(split_points)}: validación desde {split.tz_convert(TZ)}")
        predictions, meta = run_fold(observations, split)
        predictions["fold"] = index
        fold_frames.append(predictions)
        fold_meta.append(meta)

    all_predictions = pd.concat(fold_frames, ignore_index=True)
    records: list[dict] = []
    for fold, frame in all_predictions.groupby("fold", sort=True):
        for model_name, column in MODEL_NAMES.items():
            metrics, _ = score(frame, column)
            records.append({"fold": int(fold), "model": model_name, **metrics})
    per_fold = pd.DataFrame(records)

    summary = per_fold.groupby("model", sort=False).agg(
        mean_accuracy=("official_accuracy", "mean"),
        std_accuracy=("official_accuracy", "std"),
        min_accuracy=("official_accuracy", "min"),
        max_accuracy=("official_accuracy", "max"),
        mean_station_wape=("mean_station_wape", "mean"),
        mean_mae=("mae", "mean"),
    ).fillna(0)
    summary = summary.sort_values("mean_accuracy", ascending=False)
    summary.columns = [
        "Accuracy media (%)", "Desv. estándar (pp)", "Peor fold (%)",
        "Mejor fold (%)", "WAPE medio", "MAE medio",
    ]
    by_fold = per_fold.pivot(index="model", columns="fold", values="official_accuracy")
    by_fold.index.name = "Modelo"
    by_fold.columns = [f"Fold {column} (%)" for column in by_fold.columns]

    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ax = summary["Accuracy media (%)"].sort_values().plot.barh(
        figsize=(10, 5), color="#257a83", xlim=(0, 100)
    )
    ax.set(title=f"Accuracy medio en {args.folds} ventanas temporales", xlabel="Accuracy (%)", ylabel="")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / "backtest_modelos_accuracy.png", dpi=150)
    plt.close(ax.figure)

    fold_table = pd.DataFrame(
        [
            {
                "Fold": index,
                "Inicio validación (Bogotá)": meta["start"].tz_convert(TZ),
                "Fin validación (Bogotá)": (meta["end"] - pd.Timedelta(minutes=15)).tz_convert(TZ),
                "Filas train": f"{meta['train_rows']:,}",
                "Filas validación": f"{meta['test_rows']:,}",
            }
            for index, meta in enumerate(fold_meta, start=1)
        ]
    )
    report = f"""# Backtesting temporal de modelos — Pulso TransMi v1

## Protocolo

- Origen de los datos: `{args.source}`.
- {args.folds} ventanas semanales consecutivas. Cada fold entrena solo con targets anteriores a su corte y valida en la semana siguiente.
- Las features usan estación, horizonte, calendario, rezagos y promedios calculados hasta el origen. No se incluyen variables futuras de contexto.
- Accuracy usa la métrica oficial: WAPE calculado por estación y luego promediado sin ponderar.
- Todos los resultados usan el starter sintético; no son puntajes oficiales de competencia.

{_markdown(fold_table.set_index("Fold"))}

## Estabilidad agregada

La desviación estándar y el mínimo entre folds ayudan a ver si un modelo depende demasiado de una sola semana. Menor desviación y mayor peor-fold suelen ser señales de mayor estabilidad; se deben considerar junto con la media.

{_markdown(summary.round(3))}

## Accuracy por ventana

{_markdown(by_fold.round(2))}

![Accuracy promedio por modelo](figures/backtest_modelos_accuracy.png)

## Métodos comparados

- **Persistencia:** predice que la demanda seguirá igual que en el último intervalo conocido (15 minutos antes del origen).
- **Naive estacional diario:** repite el valor de la misma estación y franja horaria del día anterior.
- **Naive estacional semanal:** repite el valor de la misma estación, hora y día de la semana anterior.
- **Random Forest:** combina rezagos, medias móviles, calendario, estación y horizonte con muchos árboles entrenados sobre muestras/features aleatorias.
- **XGBoost:** combina árboles construidos secuencialmente; cada árbol intenta corregir errores de los anteriores.

Los tres primeros son baselines interpretables. Indican qué tan difícil es la serie y evitan atribuir valor a un modelo complejo que no supere reglas sencillas.
"""
    output = REPORTS / "backtest_modelos_v1.md"
    output.write_text(report, encoding="utf-8")
    print("\nAccuracy media y estabilidad por modelo:")
    print(summary.round(2).to_string())
    print(f"\nReporte: {output}")


if __name__ == "__main__":
    main()
