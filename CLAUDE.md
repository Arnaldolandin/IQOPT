# CLAUDE.md

Guia para Claude Code al trabajar en este repositorio.

## Que es esto

Bot de **opciones binarias en IQ Option**. Estrategia unica: **`seq`**, un modelo
secuencial (LSTM) que mira las ultimas 64 velas de 5m y devuelve `P(sube)` a horizonte
2 velas (expiracion 10m, instrumento `binary`, payout ~87%). Sin indicadores, sin
primario: el modelo decide solo. Todo el codigo y los logs estan en espanol.

> **Historia (importante, leer antes de proponer nada):** este proyecto probo
> exhaustivamente (2026-05 a 2026-07) MACD, filtros EMA/pendiente, ATR, reversion
> (RSI/Bollinger), multi-timeframe, por par, por hora, walk-forward, analisis de
> payout, meta-labeling y modelos secuenciales. **Ninguna configuracion tiene ventaja
> demostrada.** El muro es estructural: break-even 53.48% con payout 87%.
> **Mantener en DEMO.** Los OTC son un feed RNG de la casa (~50%).
> Ver memorias `estrategia-sin-edge-iq-2026-07-15`,
> `fuga-cross-seccional-y-rollover-2026-07-21`, `modelo-directo-en-curso-2026-07-21`.

## Estado del modelo actual (2026-07-21)

Held-out con corte temporal estricto, EURUSD, break-even 53.48%:

| umbral | n | WR sin rollover | EV/op |
|---|---|---|---|
| 0.53 | 2353 | 52.15% | -0.025 |
| **0.54** | **1231** | **53.64%** | **+0.003** |
| 0.56 | 377 | 55.17% | +0.032 |

`val_loss 0.6918` contra `ln(2) = 0.6931`: el modelo apenas se despega del azar.
**No hay edge establecido.** El margen a umbral 0.54 son 0.16 puntos, muy dentro del
error estadistico.

Dos rarezas conocidas del modelo actual:
- **Solo abre PUT.** Su `P` rara vez supera 0.54 por arriba pero si baja de 0.46, por
  el sesgo de la tasa base del entrenamiento (49.15% de subidas).
- **Sin control de correlacion.** Con `max_train: 10` puede abrir varias posiciones
  sobre el mismo movimiento; en la practica son una sola apuesta con stake multiple.

## Trampas medidas (NO repetir)

1. **Fuga in-sample.** No evaluar un modelo sobre los datos con que se entreno. El
   `meta_train_iq.py` historico hacia `fit()` sobre todo el cache y medir ahi daba
   WR 63% falso.
2. **Fuga cross-seccional.** Con 49 pares hay ~49 senales simultaneas correlacionadas.
   Un `TimeSeriesSplit` corta dentro de esos bloques y entrena con EURUSD para testear
   con EURGBP del mismo minuto. Inflaba el OOF a 62-70% cuando lo real era 53-55%.
   **Siempre embargo temporal alrededor del corte.**
3. **Ventana de rollover 20-22 UTC.** Ahi los backtests encuentran 61-69% de WR, pero
   es artefacto de spread y **IQ rechaza las ordenes**: el 48% de los `not available`
   cae en esa franja. Separar siempre los resultados dentro/fuera.
4. **Continuidad.** Exigir que la vela de liquidacion este a `H*300s` exactos de la de
   decision. Sin eso se cuentan gaps de fin de semana como opciones de 10 minutos.
5. **BTCUSD nunca ejecuta.** 76 senales historicas, 0 entradas (`active is suspended`
   a todas horas). Excluirlo de todo backtest o se calibra sobre operaciones
   imposibles.
6. **Hora del log != hora del filtro.** `main.py` loguea en hora local de Chile
   (UTC-4); `filtro_hora` compara contra UTC. Sumar 4 h antes de cruzar datos.

## Archivos

- **`main.py`** — el bot. Estrategia `seq`, multi-hilo, reconexion, hot-reload,
  filtro ATR. Flags: `--dry` (solo senales), sin flag = demo, `--real` (CUIDADO).
- **`seq_model.py`** — **`ventana_features()` es la funcion critica**: la usan TANTO el
  entrenamiento COMO el bot en vivo. Si divergen, el bot alimenta al modelo con
  features distintas y falla **en silencio**, devolviendo probabilidades de aspecto
  razonable pero sin significado. Cambiarla obliga a reentrenar.
- **`train_seq_save.py`** — entrena y guarda. Lee `config.json -> entrenamiento`;
  los flags de CLI mandan sobre el config.
- **`eurusd_seq.py`** — compara LSTM vs Transformer vs baseline (HistGradientBoosting)
  **y contra un control de etiquetas barajadas**, que es el suelo de ruido: si el
  modelo real no lo supera, no hay senal.
- **`telegram_commands.py`** — control por Telegram.
- **`download_ohlc_5m.py`** / **`actualizar_cache_5m.py`** — velas 5m a `cache_ohlc_5m/`.
- **`config.json`** — credenciales (⚠️ texto plano, gitignored) + `operacion` +
  `entrenamiento`. **`config.example.json`** es la plantilla sin secretos.
- **`DEPLOY.md`** / **`requirements.txt`** — instalacion en servidor.
- **`models/*.pt`** + **`.pt.json`** — pesos y la receta (`arq`, `L`, `hp`) para
  reconstruir la red. Gitignored: no viajan con `git clone`.

## Como correr

```powershell
.venv314\Scripts\python.exe watchdog.py        # FORMA NORMAL: supervisa main.py en DEMO
.venv314\Scripts\python.exe main.py            # DEMO a pelo (muere con la consola)
.venv314\Scripts\python.exe main.py --dry      # solo loguea senales
.venv314\Scripts\python.exe train_seq_save.py  # reentrena (~1 min por par)
```

Usar **`.venv314`** (con `iqoptionapi` y `torch`). **El nombre miente: es Python
3.10.11**, no 3.14. Forzar `PYTHONIOENCODING=utf-8` o los logs en espanol petan en
consolas heredadas.

**Arrancar siempre por `watchdog.py`, no por `main.py`.** Reinicia el bot si el proceso
muere o si el *bucle de trading* se congela (heartbeat viejo; el hilo de Telegram sigue
escribiendo al log aunque el bucle este muerto, asi que el log NO sirve de senal de vida).

Disponibilidad (montado el 2026-07-24, despues de que el PC se reiniciara solo y el bot
pasara 27 min caido sin que nadie lo viera):
- Tarea programada de Windows **`IQOPT-watchdog`**, al iniciar sesion. Verla con
  `Get-ScheduledTask IQOPT-watchdog`; quitarla con `Unregister-ScheduledTask`.
- `watchdog.py` toma un **cerrojo exclusivo** (`watchdog.lock`). Dos watchdogs serian dos
  bots comprando la misma senal con el stake al doble: el `_lock` de `main.py` es
  intra-proceso y no los cruzaria. El segundo se rinde y lo dice en `watchdog.log`.
- El bot **avisa por Telegram en cada arranque**. Un aviso que no pediste = algo reinicio
  la maquina.

## Mecanica de la API de IQ (clave, facil de olvidar)

- **Velas** -> activo subyacente `"EURUSD"` (get_candles).
- **Comprar** -> `api.buy(monto, "EURUSD-op", "call"|"put", minutos)` -> `(status, order_id)`.
  Requiere `get_ALL_Binary_ACTIVES_OPCODE()` tras conectar (puede colgar -> usar timeout).
- Tras `connect()`: `api.change_balance("PRACTICE")` (demo) o `"REAL"`.
- `api.check_win_v4(order_id)` **bloquea** hasta el cierre del contrato.
- expiry<=5 -> `turbo` (~83%, break-even 54.64%); >5 -> `binary` (~87%, 53.48%).
  **El horizonte 1 (5 min) esta descartado por esto:** el payout se come la ventaja.
- `get_all_profit()` da payouts sin `update_ACTIVES_OPCODE()`. Keys OTC = `"AIG-OTC"`,
  reales = `"EURUSD-op"`.
- **`get_all_open_time()` esta desactivado por defecto** (`usar_open_time: false`): en
  esta cuenta el endpoint de digitales devuelve `None` y la libreria revienta en un
  hilo propio (traceback que no alcanza nuestro `try/except` + 10-20s de arranque).
  El payout NO sirve de proxy: IQ lo devuelve tambien con el activo cerrado.
- `websocket-client` DEBE ser **0.56.0**; versiones nuevas rompen `iqoptionapi`.

## Config: lo que muerde

- Hot-reload cada 30s de `operacion`, `max_trades`, `filtro_hora` y `riesgo`.
  `pares_binarios` y `entrenamiento` **no**: requieren reiniciar.
- **`filtro_hora` es lista blanca en UTC.** `horas_por_par[par]` son las horas
  *permitidas*, y un par ausente queda bloqueado 24h. Activarlo con `horas_por_par`
  vacio **apaga el bot en silencio**. `timezone_offset` solo se usa en un log.
- **`reintento_max_seg: 60`** (era 240). IQ solo acepta ordenes en ventanas de ~3-4 min
  que abren pasado cada cuarto de hora, y el escaneo del bot esta desalineado con ellas
  (ver el comentario largo en `main.py`), asi que algo de reintento hace falta: 60s deja
  que una senal de `:05` alcance la apertura de `:06`. Pero **la senal envejece mientras
  se reintenta** y el modelo predice a 10 min DESDE la vela de decision. Medido sobre 262
  operaciones cerradas (2026-07-24, fuera de rollover): inmediatas 60.8% (n=158),
  demoradas por reintentos 50.0% (n=78). z=1.57 -> indicio, no prueba; el argumento de
  fondo es que un fill tardio no es la apuesta que el modelo senalo.
- **BTCUSD fuera de `pares_binarios`** (trampa #5): 86 senales y 0 entradas. Sigue en
  `entrenamiento.pares`, o sea que se entrena pero no se opera.

## Git y entorno

- Copia de trabajo: **`D:\Proyects\IQOPT`**, que pushea a
  `github.com/Arnaldolandin/IQOPT`. (Hasta 2026-07-21 la doc hablaba de `D:\GIT\IQOPT` y
  `Y:\IQOPT`; **ninguna de las dos existe ya** en esta maquina.)
- El venv vive en `D:\Proyects\IQOPT\.venv314`.
- Gitignored: `config.json`, **`models/*` entero** (`.pt`, `.npz`, `.pt.json`),
  `cache_ohlc_5m*/`, `heartbeat.json`, `estado_velas.json`. Los modelos se regeneran con
  `train_seq_save.py`; no viajan con `git clone`. Pendiente: rotar password IQ + token
  Telegram (estuvieron en commits viejos).

## Convenciones

- Logging con `print()` + archivo (`rsi_iq.log`). Multi-hilo con `_lock`.
  Telegram en daemon thread.
- Al medir cualquier cosa: corte temporal estricto, embargo, continuidad, sin BTCUSD,
  y **separar rollover**. Sin eso los numeros mienten, y ya mintieron varias veces.
