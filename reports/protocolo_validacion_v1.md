# Protocolo de validacion — Pulso TransMi v1

## Separar los dos calendarios

El starter historico contiene observaciones desde el 26 de julio hasta el 8 de septiembre de 2026, en hora de Bogota. Por eso el benchmark local actual entrena hasta el 1 de septiembre y valida sobre los ultimos siete dias disponibles, del 2 al 8 de septiembre. Es una evaluacion offline anterior a la activacion de la competencia.

La competencia se activo el 21 de septiembre. La API separa:

- `observed_at`: timestamp virtual/sintetico al que corresponde la observacion;
- `released_at`: momento real en que la API publico esa observacion.

En la consulta de estado del 22 de septiembre (hora de Bogota), el stream ya mostraba observaciones con `observed_at` desde el 9 de septiembre y `released_at` el 21 de septiembre. Por tanto, “datos publicados desde el 21” no significa que su `observed_at` empiece el 21. El reloj de la API y el `data_cutoff` de cada ciclo son la autoridad.

## Evaluacion offline

1. Usar el starter como conjunto historico inicial.
2. Comparar modelos con cortes temporales antes de la competencia, sin particiones aleatorias.
3. Mantener el periodo del 2 al 8 de septiembre como holdout inicial ya medido; usarlo para comparar baselines, no como score oficial.
4. Repetir en otros origenes temporales dentro del historial antes de elegir un modelo promovido.

## Prueba oficial

1. Consultar `/v1/forecast-cycles/current` en el momento de ejecutar; no construir ciclos ni timestamps por cuenta propia.
2. Entrenar solo con observaciones disponibles hasta el `data_cutoff` devuelto.
3. Predecir exactamente las parejas `station_id` + `target_at` entregadas por la API; el ciclo normal pide 48 valores.
4. Enviar con la API key personal y conservar el recibo `accepted`.
5. Evaluar cuando el sistema revele los targets reales. No incorporar datos liberados despues del cutoff a ese pronostico.

El 18 de septiembre existio una ronda de practica de 12 targets para probar la conexion; no iniciaba el reloj ni aportaba al ranking. Esa ronda ya termino. La primera prediccion oficial es un flujo distinto y su guia vigente pide un recibo con todos los targets del ciclo.

## Estado consultado

La API respondio `state=running` en `/v1/clock`. `/v1/forecast-cycles/current` respondio `404 no_open_cycle` durante la consulta; esto significa que no habia ventana de entrega abierta en ese instante. No se envio ninguna submission.

## Referencias

- [Repositorio del curso y contrato actual](https://github.com/uexternadojz/pulso-transmi)
- [Primera prediccion oficial](https://github.com/uexternadojz/pulso-transmi/blob/main/docs/primera-prediccion.md)
- [Commit del 18 de septiembre: portal y ronda de practica](https://github.com/uexternadojz/pulso-transmi/commit/c5bc212867bcaddb273aca18d221c6da9793d162)
- [API Pulso TransMi](https://pulso-transmi.72-60-245-2.sslip.io/)
