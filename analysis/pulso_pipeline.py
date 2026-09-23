"""Student-side Pulso TransMi stream, training, and forecast pipeline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from supabase import Client, create_client

from compare_baselines import build_examples, score


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
ARTIFACT_BUCKET = "model-artifacts"
PAGE_SIZE = 1000
STREAM_PAGE_SIZE = 5000
CATEGORICAL = ["station_id", "horizon"]
NUMERIC = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "lag_1", "lag_4",
    "lag_96", "lag_672", "rolling_4", "rolling_96",
]
FEATURES = CATEGORICAL + NUMERIC


def load_local_env() -> None:
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


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Falta {name} en el entorno local o en GitHub Actions Secrets.")
    return value


def supabase_client() -> Client:
    return create_client(required_env("SUPABASE_URL"), required_env("SUPABASE_SECRET_KEY"))


def api_get(path: str, params: dict[str, Any] | None = None) -> requests.Response:
    try:
        response = requests.get(
            f"{os.environ.get('PULSO_API_URL', BASE_URL).rstrip('/')}{path}",
            params=params,
            headers={"Authorization": f"Bearer {required_env('PULSO_API_KEY')}"},
            timeout=30,
        )
        return response
    except requests.RequestException as exc:
        raise RuntimeError(f"Pulso API no disponible ({type(exc).__name__}).") from None


def paged_rows(client: Client, table: str, selection: str) -> list[dict]:
    rows: list[dict] = []
    start = 0
    while True:
        result = (
            client.table(table)
            .select(selection)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def set_cursor(client: Client, cursor: str, last_observed_at: str | None) -> None:
    update: dict[str, Any] = {
        "confirmed_cursor": cursor,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_observed_at:
        update["last_observed_at"] = last_observed_at
    client.table("collector_state").update(update).eq("singleton", True).execute()


def sync_stream(client: Client) -> int:
    """Fetch released pages and only confirm a page cursor after its upsert."""
    state = (
        client.table("collector_state")
        .select("confirmed_cursor")
        .eq("singleton", True)
        .limit(1)
        .execute()
    )
    if not state.data:
        raise RuntimeError("No existe la fila singleton en collector_state.")
    cursor_before = state.data[0].get("confirmed_cursor")
    cursor = cursor_before
    cursor_after = cursor_before
    stations = {
        str(row["station_id"])
        for row in paged_rows(client, "stations", "station_id")
    }
    run = client.table("collector_runs").insert(
        {"status": "running", "cursor_before": cursor_before}
    ).execute()
    run_id = run.data[0]["run_id"] if run.data else None
    received = 0
    last_observed: str | None = None
    seen_cursors: set[str] = set()

    try:
        while True:
            params: dict[str, Any] = {"limit": STREAM_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            response = api_get("/v1/stream/observations", params)
            if response.status_code >= 400:
                raise RuntimeError(f"Stream API respondió HTTP {response.status_code}.")
            try:
                body = response.json()
            except ValueError:
                raise RuntimeError("Stream API devolvió una respuesta que no es JSON.") from None
            page = body.get("data") or []
            next_cursor = body.get("next_cursor")

            normalized: list[dict] = []
            for row in page:
                station_id = str(row.get("station_id", ""))
                observed_at = row.get("observed_at")
                demand = row.get("demand")
                if station_id not in stations or not observed_at:
                    raise RuntimeError("El stream contiene estación o timestamp inválido.")
                if isinstance(demand, bool) or not isinstance(demand, (int, float)):
                    raise RuntimeError("El stream contiene demanda no numérica.")
                if not math.isfinite(float(demand)) or demand < 0 or int(demand) != demand:
                    raise RuntimeError("El stream contiene demanda negativa o no entera.")
                timestamp = pd.Timestamp(observed_at)
                if timestamp.tzinfo is None:
                    raise RuntimeError("El stream contiene un timestamp sin zona horaria.")
                observed_utc = timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")
                normalized.append(
                    {
                        "station_id": station_id,
                        "observed_at": observed_utc,
                        "demand": int(demand),
                    }
                )
            if normalized:
                client.table("observations").upsert(
                    normalized,
                    on_conflict="station_id,observed_at",
                    returning="minimal",
                ).execute()
                received += len(normalized)
                last_observed = max(row["observed_at"] for row in normalized)

            # A cursor is committed only after the corresponding page is safely upserted.
            if next_cursor:
                next_cursor = str(next_cursor)
                if next_cursor in seen_cursors or next_cursor == cursor:
                    raise RuntimeError("El stream repitió un cursor; se detuvo para evitar un loop.")
                set_cursor(client, next_cursor, last_observed)
                cursor_after = next_cursor
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            if not next_cursor:
                break

        if run_id:
            client.table("collector_runs").update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "succeeded",
                    "cursor_after": cursor_after,
                    "rows_received": received,
                    "rows_upserted": received,
                    "metadata": {"pages": len(seen_cursors) + 1},
                }
            ).eq("run_id", run_id).execute()
    except Exception as exc:
        if run_id:
            client.table("collector_runs").update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "cursor_after": cursor_after,
                    "rows_received": received,
                    "rows_upserted": received,
                    "error_summary": type(exc).__name__,
                }
            ).eq("run_id", run_id).execute()
        raise

    if last_observed:
        client.table("collector_state").update(
            {
                "last_observed_at": last_observed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("singleton", True).execute()

    print(f"Sincronización del stream completada: {received:,} filas nuevas.")
    return received


def load_observations(client: Client, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = paged_rows(client, "observations", "station_id,observed_at,demand")
    if not rows:
        raise RuntimeError("Supabase no contiene observaciones.")
    frame = pd.DataFrame.from_records(rows)
    frame["station_id"] = frame["station_id"].astype("string")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = pd.to_numeric(frame["demand"], errors="raise")
    if frame.duplicated(["station_id", "observed_at"]).any():
        raise RuntimeError("Hay observaciones duplicadas en Supabase.")
    if frame["demand"].isna().any() or frame["demand"].lt(0).any():
        raise RuntimeError("Supabase contiene demanda inválida.")
    if cutoff is not None:
        frame = frame.loc[frame["observed_at"] <= cutoff].copy()
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def make_model() -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
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


def hourly_origins(frame: pd.DataFrame) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(sorted(frame["observed_at"].unique()))
    return times[times.tz_convert("America/Bogota").minute == 0]


def train_and_promote(client: Client, cutoff: pd.Timestamp) -> dict:
    history = load_observations(client, cutoff)
    if history.empty:
        raise RuntimeError("No hay datos anteriores al cutoff para entrenar.")
    station_latest = history.groupby("station_id")["observed_at"].max()
    too_stale = station_latest[station_latest < cutoff - pd.Timedelta(minutes=30)]
    if not too_stale.empty:
        raise RuntimeError(
            "Faltan observaciones recientes al cutoff para estaciones: "
            + ", ".join(map(str, too_stale.index.tolist()))
        )

    origins = hourly_origins(history)
    if len(origins) < 2:
        raise RuntimeError("No hay suficientes orígenes horarios para entrenar.")
    local_end = history["observed_at"].max().tz_convert("America/Bogota")
    validation_start = (local_end.normalize() - pd.Timedelta(days=6)).tz_localize(None)
    validation_start = validation_start.tz_localize("America/Bogota").tz_convert("UTC")
    if validation_start >= cutoff:
        raise RuntimeError("La ventana de validación está vacía al cutoff actual.")

    examples = build_examples(history, origins)
    train = examples.loc[examples["target_at"] < validation_start].copy()
    validation = examples.loc[
        (examples["origin"] >= validation_start)
        & (examples["target_at"] <= cutoff)
    ].copy()
    if train.empty or validation.empty:
        raise RuntimeError("No se pudo formar el holdout temporal antes de entrenar.")

    validator = make_model()
    validator.fit(train[FEATURES], train["y"])
    validation["random_forest"] = np.maximum(0, validator.predict(validation[FEATURES]))
    rf_metrics, _ = score(validation, "random_forest")
    weekly_metrics, _ = score(validation, "weekly")
    print(
        "Validación reciente (accuracy oficial): "
        f"RF={rf_metrics['official_accuracy']:.2f}%, "
        f"naive semanal={weekly_metrics['official_accuracy']:.2f}%"
    )
    if rf_metrics["official_accuracy"] <= weekly_metrics["official_accuracy"]:
        raise RuntimeError(
            "Random Forest no superó el naive semanal en el holdout reciente; "
            "se conserva el champion actual y no se promueve este candidato."
        )

    final_examples = examples.loc[examples["target_at"] <= cutoff].copy()
    if final_examples.empty:
        raise RuntimeError("No hay ejemplos etiquetados antes del cutoff final.")
    model = make_model()
    model.fit(final_examples[FEATURES], final_examples["y"])
    training_data_start = final_examples["target_at"].min()
    training_data_end = final_examples["target_at"].max()
    trained_at = datetime.now(timezone.utc)
    version = f"rf-{trained_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    git_commit = None
    try:
        candidate = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if len(candidate) == 40:
            git_commit = candidate
    except (OSError, subprocess.CalledProcessError):
        pass

    run_result = client.table("training_runs").insert(
        {
            "model_family": "random_forest_global",
            "finished_at": trained_at.isoformat(),
            "training_data_start": training_data_start.isoformat(),
            "training_data_end": training_data_end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": cutoff.isoformat(),
            "status": "completed",
            "code_commit": git_commit,
            "parameters": {
                "n_estimators": 120,
                "min_samples_leaf": 3,
                "max_features": 0.8,
                "random_state": 42,
            },
            "metrics": {
                "random_forest": rf_metrics,
                "weekly_naive": weekly_metrics,
            },
            "notes": "Candidato validado temporalmente contra naive semanal.",
        }
    ).execute()
    training_run_id = run_result.data[0]["training_run_id"]

    artifact_path = f"{version}/model.joblib"
    artifact_uri = f"supabase://{ARTIFACT_BUCKET}/{artifact_path}"
    artifact = {
        "pipeline": model,
        "model_version": version,
        "training_data_start": training_data_start.isoformat(),
        "training_data_end": training_data_end.isoformat(),
        "trained_at": trained_at.isoformat(),
        "code_commit": git_commit,
        "features": FEATURES,
        "validation_metrics": {
            "random_forest": rf_metrics,
            "weekly_naive": weekly_metrics,
        },
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    try:
        client.storage.from_(ARTIFACT_BUCKET).upload(
            artifact_path,
            buffer.getvalue(),
            file_options={"content-type": "application/octet-stream", "upsert": "false"},
        )
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo guardar el artefacto en Storage ({type(exc).__name__}); "
            f"revisa que exista el bucket privado '{ARTIFACT_BUCKET}'."
        ) from None

    client.table("model_versions").insert(
        {
            "model_version": version,
            "training_run_id": training_run_id,
            "status": "candidate",
            "training_data_end": training_data_end.isoformat(),
            "code_commit": git_commit,
            "artifact_uri": artifact_uri,
            "features": FEATURES,
            "validation_metrics": artifact["validation_metrics"],
            "notes": "Champion inicial RF; superó naive semanal en holdout temporal reciente.",
        }
    ).execute()

    client.table("model_versions").update({"status": "retired"}).eq(
        "status", "champion"
    ).execute()
    client.table("model_versions").update(
        {"status": "champion", "promoted_at": datetime.now(timezone.utc).isoformat()}
    ).eq("model_version", version).execute()
    print(f"Champion promovido: {version}; artefacto: {artifact_uri}")
    return artifact


def load_champion(client: Client) -> dict:
    result = (
        client.table("model_versions")
        .select("model_version,training_data_end,artifact_uri,features,validation_metrics")
        .eq("status", "champion")
        .limit(1)
        .execute()
    )
    if not result.data:
        raise RuntimeError("No hay un modelo champion; ejecuta primero el entrenamiento.")
    row = result.data[0]
    prefix = f"supabase://{ARTIFACT_BUCKET}/"
    if not str(row.get("artifact_uri", "")).startswith(prefix):
        raise RuntimeError("La ubicación del artefacto champion no es válida.")
    object_path = row["artifact_uri"][len(prefix) :]
    try:
        payload = client.storage.from_(ARTIFACT_BUCKET).download(object_path)
        artifact = joblib.load(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el artefacto ({type(exc).__name__}).") from None
    if artifact.get("model_version") != row["model_version"]:
        raise RuntimeError("La versión del archivo no coincide con model_versions.")
    artifact["database_row"] = row
    return artifact


def build_target_features(history: pd.DataFrame, cycle: dict) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle["data_cutoff"])
    if cutoff.tzinfo is None:
        raise RuntimeError("El cutoff del ciclo no incluye zona horaria.")
    cutoff = cutoff.tz_convert("UTC")
    station_ids = set(history["station_id"].astype(str))
    rows: list[dict] = []
    targets = cycle.get("targets") or []
    if len(targets) != cycle.get("expected_predictions"):
        raise RuntimeError("El conjunto de targets no coincide con expected_predictions.")
    seen: set[tuple[str, str]] = set()
    for target in targets:
        station_id = str(target["station_id"])
        target_at = pd.Timestamp(target["target_at"])
        if target_at.tzinfo is None:
            raise RuntimeError("Un target no incluye zona horaria.")
        target_at = target_at.tz_convert("UTC")
        horizon_minutes = int(target["horizon_minutes"])
        actual_minutes = int((target_at - cutoff).total_seconds() // 60)
        if actual_minutes != horizon_minutes:
            raise RuntimeError("Un target no coincide con el cutoff/horizonte del ciclo.")
        if (station_id, target_at.isoformat()) in seen:
            raise RuntimeError("El ciclo contiene targets duplicados.")
        seen.add((station_id, target_at.isoformat()))
        if station_id not in station_ids:
            raise RuntimeError(f"Faltan datos de la estación {station_id}.")
        series = history.loc[
            (history["station_id"].astype(str) == station_id)
            & (history["observed_at"] <= cutoff)
        ].sort_values("observed_at")["demand"]
        if len(series) < 672:
            raise RuntimeError(f"No hay 7 días de historia utilizables para {station_id}.")
        local_target = target_at.tz_convert("America/Bogota")
        values = series.astype(float).to_numpy()
        rows.append(
            {
                "station_id": station_id,
                "horizon": horizon_minutes // 15,
                "origin": cutoff,
                "target_at": target["target_at"],
                "hour_sin": math.sin(2 * math.pi * local_target.hour / 24),
                "hour_cos": math.cos(2 * math.pi * local_target.hour / 24),
                "dow_sin": math.sin(2 * math.pi * local_target.dayofweek / 7),
                "dow_cos": math.cos(2 * math.pi * local_target.dayofweek / 7),
                "lag_1": values[-1],
                "lag_4": values[-4],
                "lag_96": values[-96],
                "lag_672": values[-672],
                "rolling_4": float(values[-4:].mean()),
                "rolling_96": float(values[-96:].mean()),
            }
        )
    if len(seen) != len(targets):
        raise RuntimeError("No se pudo construir una fila para cada target.")
    return pd.DataFrame(rows)


def persist_cycle(client: Client, cycle: dict) -> None:
    client.table("forecast_cycles").upsert(
        {
            "cycle_id": cycle["cycle_id"],
            "data_cutoff": cycle["data_cutoff"],
            "closes_at": cycle["closes_at"],
            "expected_predictions": cycle["expected_predictions"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "payload": cycle,
        },
        on_conflict="cycle_id",
        returning="minimal",
        default_to_null=False,
    ).execute()
    rows = [
        {
            "cycle_id": cycle["cycle_id"],
            "station_id": target["station_id"],
            "target_at": target["target_at"],
            "horizon_minutes": target["horizon_minutes"],
        }
        for target in cycle["targets"]
    ]
    for start in range(0, len(rows), 500):
        client.table("cycle_targets").upsert(
            rows[start : start + 500],
            on_conflict="cycle_id,station_id,target_at",
            returning="minimal",
        ).execute()


def run_forecast(client: Client, send: bool) -> None:
    sync_stream(client)
    response = api_get("/v1/forecast-cycles/current")
    if response.status_code == 404:
        try:
            error_body = response.json()
            detail = error_body.get("detail")
            code = detail.get("code") if isinstance(detail, dict) else detail
        except (ValueError, AttributeError):
            code = None
        if code in {"no_open_cycle", "no_open_forecast_cycle"}:
            print("No hay ciclo abierto; sincronización completada sin envío.")
            return
    if response.status_code >= 400:
        raise RuntimeError(f"Consulta del ciclo respondió HTTP {response.status_code}.")
    cycle = response.json()
    if cycle.get("state") != "open":
        print("La API no reporta un ciclo abierto; no se envía.")
        return
    persist_cycle(client, cycle)

    existing = (
        client.table("submissions")
        .select("submission_id")
        .eq("cycle_id", cycle["cycle_id"])
        .eq("is_official", True)
        .limit(1)
        .execute()
    )
    if existing.data:
        print(f"El ciclo {cycle['cycle_id']} ya tiene entrega oficial registrada; se omite.")
        return

    cutoff = pd.Timestamp(cycle["data_cutoff"])
    if cutoff.tzinfo is None:
        raise RuntimeError("El cutoff del ciclo no incluye zona horaria.")
    cutoff = cutoff.tz_convert("UTC")
    champion = load_champion(client)
    training_end = pd.Timestamp(champion["training_data_end"])
    if training_end.tzinfo is None:
        training_end = training_end.tz_localize("UTC")
    if training_end.tz_convert("UTC") > cutoff:
        raise RuntimeError("El champion fue entrenado con datos posteriores al cutoff.")
    history = load_observations(client, cutoff)
    station_latest = history.groupby("station_id")["observed_at"].max()
    too_stale = station_latest[station_latest < cutoff - pd.Timedelta(minutes=30)]
    if not too_stale.empty:
        raise RuntimeError(
            "Datos desactualizados al cutoff para estaciones: "
            + ", ".join(map(str, too_stale.index.tolist()))
        )
    features = build_target_features(history, cycle)
    estimator = champion["pipeline"]
    raw_values = estimator.predict(features[FEATURES])
    predictions = [
        {
            "station_id": str(row.station_id),
            "target_at": str(row.target_at),
            "value": round(max(0.0, float(value)), 3),
        }
        for row, value in zip(features.itertuples(index=False), raw_values, strict=True)
    ]
    expected = {
        (str(row["station_id"]), pd.Timestamp(row["target_at"]).isoformat())
        for row in cycle["targets"]
    }
    actual = {
        (row["station_id"], pd.Timestamp(row["target_at"]).isoformat())
        for row in predictions
    }
    if actual != expected or len(predictions) != cycle["expected_predictions"]:
        raise RuntimeError("Predicciones no coinciden exactamente con los targets de la API.")
    if any(not math.isfinite(row["value"]) or row["value"] > 100000 for row in predictions):
        raise RuntimeError("El modelo produjo una predicción fuera del contrato de la API.")

    run_hash = hashlib.sha256(
        f"{cycle['cycle_id']}|{champion['model_version']}".encode("utf-8")
    ).hexdigest()[:32]
    run_id = f"pulso-{run_hash}"
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": run_id,
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": champion["model_version"],
            "trained_at": champion["trained_at"],
            "training_data_end": champion["training_data_end"],
        },
        "predictions": predictions,
    }
    if champion.get("code_commit"):
        payload["model"]["git_commit"] = champion["code_commit"]
    if not send:
        print(
            f"DRY RUN: ciclo {cycle['cycle_id']}, modelo {champion['model_version']}, "
            f"{len(predictions)}/{cycle['expected_predictions']} targets listos; no se envió."
        )
        print(json.dumps(predictions[:3], ensure_ascii=False))
        return

    try:
        submission = requests.post(
            f"{os.environ.get('PULSO_API_URL', BASE_URL).rstrip('/')}/v1/submissions",
            headers={
                "Authorization": f"Bearer {required_env('PULSO_API_KEY')}",
                "Idempotency-Key": run_id,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Envío a Pulso API falló ({type(exc).__name__}).") from None
    if submission.status_code >= 400:
        try:
            error = submission.json()
            detail = error.get("detail", {})
            code = detail.get("code") if isinstance(detail, dict) else None
        except ValueError:
            code = None
        raise RuntimeError(
            f"Pulso API rechazó la entrega (HTTP {submission.status_code}, código {code or 'no disponible'})."
        )
    receipt = submission.json()
    status = receipt.get("status")
    submission_id = receipt.get("submission_id")
    if status not in {"accepted", "duplicate"} or not submission_id:
        raise RuntimeError("La API no devolvió un recibo aceptado válido.")
    predictions_hash = hashlib.sha256(
        json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    received_at = receipt.get("received_at") or receipt.get("accepted_at")
    if not received_at:
        raise RuntimeError("El recibo aceptado no incluye timestamp de recepción.")
    client.table("submissions").upsert(
        {
            "submission_id": submission_id,
            "client_run_id": run_id,
            "cycle_id": cycle["cycle_id"],
            "model_version": champion["model_version"],
            "attempt": receipt.get("attempt", 1),
            "status": status,
            "accepted_at": received_at,
            "data_cutoff": cycle["data_cutoff"],
            "training_data_end": champion["training_data_end"],
            "code_commit": champion.get("code_commit"),
            "batch_sha256": predictions_hash,
            "is_official": bool(receipt.get("is_official", True)),
            "receipt": receipt,
        },
        on_conflict="submission_id",
        returning="minimal",
    ).execute()
    client.table("predictions").upsert(
        [
            {
                "submission_id": submission_id,
                "station_id": row["station_id"],
                "target_at": row["target_at"],
                "predicted_value": row["value"],
            }
            for row in predictions
        ],
        on_conflict="submission_id,station_id,target_at",
        returning="minimal",
    ).execute()
    client.table("forecast_cycles").update({"status": "submitted"}).eq(
        "cycle_id", cycle["cycle_id"]
    ).execute()
    print(
        f"Entrega {status}: {submission_id}; "
        f"predicciones {receipt.get('predictions_received')}/{receipt.get('expected_predictions')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Opera la solución Pulso TransMi.")
    parser.add_argument("command", choices=("sync", "train", "forecast"))
    parser.add_argument(
        "--send",
        action="store_true",
        help="Enviar la predicción. Sin esta opción forecast solo valida e imprime un preview.",
    )
    args = parser.parse_args()
    load_local_env()
    client = supabase_client()
    if args.command == "sync":
        sync_stream(client)
    elif args.command == "train":
        sync_stream(client)
        cycle_response = api_get("/v1/forecast-cycles/current")
        if cycle_response.status_code == 200:
            cycle = cycle_response.json()
            cutoff = pd.Timestamp(cycle["data_cutoff"])
            if cutoff.tzinfo is None:
                raise SystemExit("El cutoff del ciclo no incluye zona horaria.")
            cutoff = cutoff.tz_convert("UTC")
        elif cycle_response.status_code == 404:
            history = load_observations(client)
            cutoff = history["observed_at"].max()
        else:
            raise SystemExit(f"No se pudo consultar el ciclo (HTTP {cycle_response.status_code}).")
        train_and_promote(client, cutoff)
    else:
        run_forecast(client, send=args.send)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        message = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        print(f"Pipeline detenido: {message}", file=sys.stderr)
        raise SystemExit(1) from None
