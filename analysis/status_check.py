"""Estado operativo (solo lectura): últimas entregas, últimas sincronizaciones y
frescura de las observaciones."""
import pandas as pd

import pulso_pipeline as pp

TZ = "America/Bogota"


def main() -> None:
    client = pp.supabase_client()
    subs = client.table("submissions").select(
        "cycle_id,status,is_official,accepted_at,model_version,data_cutoff"
    ).order("accepted_at", desc=True).limit(8).execute().data
    print("Últimas entregas:")
    for s in subs:
        print(" ", s["accepted_at"], s["status"], s["cycle_id"], "cutoff", s["data_cutoff"])
    runs = client.table("collector_runs").select(
        "started_at,status,rows_received,metadata,error_summary"
    ).order("started_at", desc=True).limit(6).execute().data
    print("Últimas sincronizaciones:")
    for r in runs:
        print(" ", r["started_at"], r["status"], r["rows_received"], r.get("metadata"), r.get("error_summary") or "")
    last = client.table("observations").select("observed_at").order("observed_at", desc=True).limit(1).execute().data
    print("Última observación guardada:", last[0]["observed_at"] if last else None)
    cycle = pp.api_get("/v1/forecast-cycles/current")
    print("Ciclo actual:", cycle.status_code, {k: v for k, v in cycle.json().items() if k != "targets"} if cycle.status_code == 200 else cycle.text[:300])
    me = pp.api_get("/v1/me")
    print("Yo:", me.status_code, me.text[:600])
    lb = pp.api_get("/v1/leaderboard")
    print("Leaderboard:", lb.status_code)
    try:
        body = lb.json()
        rows = body if isinstance(body, list) else next((v for v in body.values() if isinstance(v, list)), [])
        print("claves:", list(body.keys()) if isinstance(body, dict) else "lista")
        for r in rows[:40]:
            print("  ", str(r)[:400])
    except Exception as exc:  # noqa: BLE001
        print("  no JSON:", lb.text[:800], exc)
    clock = pp.api_get("/v1/clock")
    print("Reloj:", clock.status_code, clock.text[:400])


if __name__ == "__main__":
    main()
