#!/usr/bin/env bash
# Run a command and retry it only when it exits with 75 (transient network error),
# up to RETRY_ATTEMPTS times with RETRY_DELAY seconds between attempts.
set -u
attempts="${RETRY_ATTEMPTS:-3}"
delay="${RETRY_DELAY:-30}"
for attempt in $(seq 1 "$attempts"); do
  "$@"
  code=$?
  if [ "$code" -ne 75 ]; then
    exit "$code"
  fi
  if [ "$attempt" -lt "$attempts" ]; then
    echo "Error de red transitorio (intento $attempt/$attempts); reintento en ${delay} s."
    sleep "$delay"
  fi
done
echo "Error de red transitorio persistente tras $attempts intentos."
exit 75
