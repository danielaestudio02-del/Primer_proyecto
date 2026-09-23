"""Exploratory analysis for the Pulso TransMi starter dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "starter"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TZ = "America/Bogota"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(frame.index.name or "")] + [str(column) for column in frame.columns]
    rows = [headers]
    for index, row in frame.iterrows():
        rows.append([str(index), *[str(value) for value in row.tolist()]])
    widths = [max(len(row[column]) for row in rows) for column in range(len(headers))]
    lines = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(rows[0])) + " |"]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    observations = pd.read_csv(
        DATA / "observations.csv",
        parse_dates=["observed_at"],
        dtype={"station_id": "string"},
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": "string"})

    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    observations["local_time"] = observations["observed_at"].dt.tz_convert(TZ)
    observations["hour"] = observations["local_time"].dt.hour
    observations["day_of_week"] = observations["local_time"].dt.day_name()

    expected_times = pd.date_range(
        observations["observed_at"].min(),
        observations["observed_at"].max(),
        freq="15min",
    )
    per_station = observations.groupby("station_id", observed=True)
    counts = per_station.size()
    duplicate_keys = int(observations.duplicated(["station_id", "observed_at"]).sum())
    missing_values = int(observations[["observed_at", "station_id", "demand"]].isna().sum().sum())
    missing_intervals = {
        station: len(expected_times.difference(group["observed_at"]))
        for station, group in per_station
    }
    station_stats = per_station["demand"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    overall = observations["demand"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    joined = observations.merge(context, on="observed_at", validate="many_to_one")
    context_columns = [
        "rain_mm",
        "rain_forecast",
        "temperature_c",
        "temperature_forecast",
        "event_intensity",
    ]
    context_corr = joined[["demand", *context_columns]].corr()["demand"].drop("demand")

    sns.set_theme(style="whitegrid", context="notebook")
    daily = (
        observations.set_index("local_time")
        .groupby("station_id")["demand"]
        .resample("D")
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=daily, x="local_time", y="demand", hue="station_id", ax=ax, linewidth=1)
    ax.set(title="Demanda diaria por estación", xlabel="Fecha (Bogotá)", ylabel="Demanda sintética")
    ax.legend(title="Estación", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "demanda_diaria_estacion.png", dpi=150)
    plt.close(fig)

    hourly = observations.groupby(["station_id", "hour"], observed=True)["demand"].mean().reset_index()
    heatmap_data = hourly.pivot(index="station_id", columns="hour", values="demand")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="mako", ax=ax, cbar_kws={"label": "Demanda media"})
    ax.set(title="Perfil horario medio por estación", xlabel="Hora local (Bogotá)", ylabel="ID estación")
    fig.tight_layout()
    fig.savefig(FIGURES / "perfil_horario_estacion.png", dpi=150)
    plt.close(fig)

    hour_profile = observations.groupby("hour", observed=True)["demand"].mean()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_profile = observations.groupby("day_of_week", observed=True)["demand"].mean().reindex(dow_order)
    station_names = stations.set_index("station_id")["station_name"]
    station_stats.index = [f"{sid} — {station_names.get(sid, 'sin nombre')}" for sid in station_stats.index]
    station_table = markdown_table(station_stats.round(2))
    dow_table = markdown_table(dow_profile.rename_axis("Dia").rename("Demanda media").round(2).to_frame())
    station_counts = counts.to_dict()
    actual_hashes = {name: sha256(DATA / name) for name in metadata["files"]}
    hashes_ok = all(
        actual_hashes[name] == metadata["files"][name]["sha256"] for name in actual_hashes
    )

    report = f"""# Análisis exploratorio — Pulso TransMi starter v1

## Alcance

El starter contiene demanda **sintética** y contexto generado para el reto. Los nombres, corredores y coordenadas de estaciones corresponden a metadatos; la demanda no representa afluencia histórica real. La zona horaria analítica es `{TZ}`.

- Filas de demanda: {len(observations):,}; filas de contexto: {len(context):,}; estaciones: {observations['station_id'].nunique()}.
- Cobertura: {observations['observed_at'].min().tz_convert(TZ)} a {observations['observed_at'].max().tz_convert(TZ)}, en intervalos de 15 minutos.
- Filas por estación: mínimo {counts.min():,}, máximo {counts.max():,}; intervalos esperados por estación: {len(expected_times):,}.
- Claves estación-tiempo duplicadas: {duplicate_keys}; valores ausentes en observaciones: {missing_values}; valores negativos: {int(observations['demand'].lt(0).sum())}.
- Intervalos faltantes por estación: {sum(missing_intervals.values())} en total.
- Verificación SHA-256 del manifiesto: {'correcta' if hashes_ok else 'NO coincide'}.

## Distribución de demanda

| Estadístico | Valor |
|---|---:|
| Media | {overall['mean']:.2f} |
| Desviación estándar | {overall['std']:.2f} |
| Mínimo / mediana / máximo | {overall['min']:.0f} / {overall['50%']:.0f} / {overall['max']:.0f} |
| Percentil 90 / 95 / 99 | {overall['90%']:.0f} / {overall['95%']:.0f} / {overall['99%']:.0f} |
| Filas con demanda cero | {int(observations['demand'].eq(0).sum())} ({observations['demand'].eq(0).mean():.2%}) |

### Resumen por estación

{station_table}

## Patrones temporales iniciales

La media agregada por hora alcanza su máximo a las **{int(hour_profile.idxmax()):02d}:00** ({hour_profile.max():.1f}) y su mínimo a las **{int(hour_profile.idxmin()):02d}:00** ({hour_profile.min():.1f}). La mayor media por día de semana es **{dow_profile.idxmax()}** ({dow_profile.max():.1f}); la menor es **{dow_profile.idxmin()}** ({dow_profile.min():.1f}). Estas cifras agregan estaciones con escalas distintas, por lo que se debe revisar también el perfil por estación.

{dow_table}

![Demanda diaria por estación](figures/demanda_diaria_estacion.png)

![Perfil horario por estación](figures/perfil_horario_estacion.png)

## Contexto

El contexto se une por timestamp, con validación muchos-a-uno. Correlaciones lineales descriptivas con demanda:

{markdown_table(context_corr.rename("correlation").round(3).to_frame())}

Estas correlaciones no establecen causalidad. Además, para pronosticar cada target solo se pueden usar variables de contexto conocidas en el momento de emitir la predicción; las columnas observadas futuras provocarían fuga de información.

## Decisiones para la siguiente fase

1. Mantener `station_id` como texto para preservar identificadores como `02300`.
2. Usar validación temporal: entrenar en los primeros 38 días y evaluar los últimos 7 días, replicando los cuatro horizontes de 15 a 60 minutos.
3. Comparar persistencia y naïve estacional (96 intervalos diarios; 672 semanales) antes del Random Forest global sugerido como ejemplo por el curso.
4. Reportar métricas por estación y horizonte junto con la métrica oficial promedio por estación.
5. No usar una partición aleatoria ni rellenar rezagos con valores futuros.

## Integridad

| Archivo | SHA-256 local | Coincide con metadata |
|---|---|---|
""" + "\n".join(
        f"| `{name}` | `{actual_hashes[name]}` | {'sí' if actual_hashes[name] == metadata['files'][name]['sha256'] else 'NO'} |"
        for name in actual_hashes
    ) + "\n"
    (REPORTS / "eda_starter_v1.md").write_text(report, encoding="utf-8")
    print(f"Reporte: {REPORTS / 'eda_starter_v1.md'}")
    print(f"Hash del manifiesto: {'OK' if hashes_ok else 'FALLÓ'}")
    print(f"Claves duplicadas: {duplicate_keys}; ausentes: {missing_values}; negativos: {int(observations['demand'].lt(0).sum())}")
    print(f"Demanda media: {overall['mean']:.2f}; mediana: {overall['50%']:.0f}; máximo: {overall['max']:.0f}")


if __name__ == "__main__":
    main()
