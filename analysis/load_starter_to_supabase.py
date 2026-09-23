"""Verify and idempotently load the public starter dataset into Supabase."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from supabase import Client, create_client


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "starter"
BATCH_SIZE = 500


def load_local_env() -> None:
    """Load local variables without ever printing their values."""
    env_path = ROOT / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value:
            os.environ.setdefault(name, value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(frame: pd.DataFrame) -> list[dict]:
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def upsert_in_batches(
    client: Client,
    table: str,
    rows: list[dict],
    conflict_columns: str,
) -> None:
    total = len(rows)
    for start in range(0, total, BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        (
            client.table(table)
            .upsert(
                batch,
                on_conflict=conflict_columns,
                returning="minimal",
                default_to_null=False,
            )
            .execute()
        )
        end = min(start + len(batch), total)
        print(f"{table}: {end:,}/{total:,}")


def main() -> None:
    load_local_env()
    supabase_url = os.environ.get("SUPABASE_URL")
    secret_key = os.environ.get("SUPABASE_SECRET_KEY")
    if not supabase_url or not secret_key:
        raise SystemExit(
            "Faltan SUPABASE_URL o SUPABASE_SECRET_KEY en .env.local; valores ocultos."
        )

    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    for filename, details in metadata["files"].items():
        actual = file_sha256(DATA / filename)
        if actual != details["sha256"]:
            raise SystemExit(f"Integridad fallida para {filename}; no se cargaron datos.")

    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        DATA / "observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])

    if observations.duplicated(["station_id", "observed_at"]).any():
        raise SystemExit("Hay claves station_id/observed_at duplicadas; no se cargaron datos.")
    if observations["demand"].isna().any() or observations["demand"].lt(0).any():
        raise SystemExit("La demanda contiene valores inválidos; no se cargaron datos.")
    if stations["station_id"].isna().any() or stations["station_id"].duplicated().any():
        raise SystemExit("El catálogo de estaciones contiene IDs inválidos o repetidos.")

    station_ids = set(stations["station_id"].astype(str))
    if set(observations["station_id"].astype(str)) != station_ids:
        raise SystemExit("Las estaciones de observaciones no coinciden con el catálogo.")
    if len(observations) != metadata["observation_rows"]:
        raise SystemExit("El total de observaciones no coincide con metadata.json.")
    if len(context) != metadata["context_rows"]:
        raise SystemExit("El total de contexto no coincide con metadata.json.")

    for frame in (observations, context):
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True).map(
            lambda value: value.isoformat().replace("+00:00", "Z")
        )

    client = create_client(supabase_url, secret_key)
    try:
        upsert_in_batches(client, "stations", records(stations), "station_id")
        upsert_in_batches(
            client,
            "context_observations",
            records(context),
            "observed_at",
        )
        upsert_in_batches(
            client,
            "observations",
            records(observations),
            "station_id,observed_at",
        )
    except Exception as exc:  # Avoid SDK exception text, which may contain request details.
        raise SystemExit(
            f"Carga interrumpida ({type(exc).__name__}); puedes repetirla sin duplicar filas."
        ) from None

    print(
        "Carga completada: "
        f"{len(stations)} estaciones, {len(context):,} filas de contexto y "
        f"{len(observations):,} observaciones."
    )


if __name__ == "__main__":
    main()
