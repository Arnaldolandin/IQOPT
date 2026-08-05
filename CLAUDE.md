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

**En produccion el modelo es un BLEND LSTM+HGB, no un LSTM puro.** `predecir_seq()`
(main.py:712) promedia `P(LSTM)` con `P(HGB)` (HistGradientBoosting sobre la misma
ventana aplanada) si al lado del `.pt` existe el `_hgb.pkl`, ponderando por
`hgb_peso` (`0.75` en produccion, default 0.5). Sin `.pkl` queda el LSTM solo
(retrocompat). Los 50 pares tienen su `_hgb.pkl` (2026-08-03). Umbral activo en
produccion: `seq_threshold: 0.54` + `umbrales_por_par` 0.54 para los 50 pares
(`umbral_de()`, main.py:701).

Dos rarezas conocidas del modelo actual:
- ~~**Solo abre PUT.**~~ **YA NO** (comprobado 2026-07-28). Era cierto de los modelos de
  una epoca anterior, sesgados por la tasa base del entrenamiento (49.15% de subidas). Los
  modelos actuales abren los dos lados casi por igual: de 527 cierres del log, 267 PUT y
  260 CALL. Ese log daba PUT 54.31% vs CALL 45.77%, 8.5 puntos, pero **eso NO es una
  asimetria real**: medido OOS con n~50k se invierte segun el umbral (a 0.56 gana CALL
  54.77 vs 53.89; a 0.54 gana PUT 53.78 vs 53.35). Es ruido de n pequena. No filtrar por
  lado. Ver `reversion_condicionada_test.py`.
- **Control de correlacion por exposicion** (`max_por_divisa: 2`).
  `correlacion_excedida()` (main.py:91) descompone cada par en exposiciones con signo
  (+EUR/-USD) y bloquea abrir una posicion si ya hay `max_por_divisa` vivas apostando la
  MISMA divisa (CALL +EUR y PUT -EUR no cuentan). No es total: aun pueden quedar varias
  abiertas, y como todas vencen en la MISMA marca de 15 min (ver "Mecanica de la API")
  liquidan sobre el mismo tick (se llegaron a ver 14 operaciones cerrando en el mismo
  minuto de reloj). `max_por_divisa: 0` desactiva el control.

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

### Estado en disco (dónde guarda el bot su estado)

El bot persiste estado en 4 archivos del directorio raiz, todos gitignored y
reescritos en cada ciclo. Son la unica fuente de verdad si el proceso muere:

- **`heartbeat.json`** — `{"ts": <epoch>}` del ultimo ciclo del bucle de trading.
  Escritura ATOMICA a proposito (`os.replace`): con `open(..., "w")` el watchdog
  podia leer el archivo a 0 bytes entre el truncado y la escritura y reiniciar un
  bot sano (paso 2 veces la noche del 2026-07-28 al 29). Es la senal de vida que
  lee `watchdog.py` (MAX_SILENCIO 600s -> reinicia).
- **`operaciones_pendientes.json`** — oids de operaciones abiertas sin cierre.
  `persistir_apertura()` (main.py:187) lo escribe tras cada `[ENTRADA]`;
  `quitar_pendiente()` (main.py:197) lo borra tras el `[CIERRE]`. Al arrancar,
  `recuperar_pendientes()` (main.py:236) consulta cada oid por el canal crudo
  (`api.api.get_betinfo`, sin el reconnect interno de la libreria, que rompia el
  WS) y loguea `[CIERRE-RECUP]` si IQ lo responde; si no responde en ~12s, la
  descarta (orden vieja: el balance lo refleja). Escritura atomica con `.tmp`.
- **`estado_velas.json`** — ultimo id de vela ya procesado por activo. Evita que un
  reinicio duplique operaciones sobre la misma vela (el 2026-07-22, con el
  vigilante reiniciando cada 3-5 min, 3 senales se convirtieron en 7 posiciones y
  EURJPY se compro 3 veces sobre la misma vela).
- **`rsi_iq.log`** — el log completo. NO sirve de senal de vida: el hilo de
  Telegram sigue escribiendo aunque el bucle de trading este muerto; el watchdog
  usa `heartbeat.json`, no este archivo.

## Como correr

```powershell
.venv314\Scripts\python.exe watchdog.py        # FORMA NORMAL: supervisa main.py en DEMO
.venv314\Scripts\python.exe main.py            # DEMO a pelo (muere con la consola)
.venv314\Scripts\python.exe main.py --dry      # solo loguea senales
.venv314\Scripts\python.exe train_seq_save.py  # reentrena (~1 min por par)
```

Usar **`.venv314`** (con `iqoptionapi` y `torch`). Forzar `PYTHONIOENCODING=utf-8` o los
logs en espanol petan en consolas heredadas.

**Hay DOS `.venv314` y no son iguales** (ver "Git y entorno"): el del PC de trabajo es
Python 3.14.4 con `torch` 2.13 operativo, y ahi se entrena. El del servidor **miente en
el nombre: es Python 3.10.11**, y `torch` no carga (`c10.dll`), asi que ahi solo se
infiere, en numpy puro, leyendo los `.npz` que exporta `exportar_npz.py`.

**Arrancar siempre por `watchdog.py`, no por `main.py`.** Reinicia el bot si el proceso
muere o si el *bucle de trading* se congela (heartbeat viejo; el hilo de Telegram sigue
escribiendo al log aunque el bucle este muerto, asi que el log NO sirve de senal de vida).

**AGUJERO CERCADO 2026-08-01: el bucle FATAL enmascaraba la congelacion.** La reconexion
in-process de iqoptionapi esta ROTA: tras una caida del WS, `connect()` cuelga >45s en
TODOS los intentos (medido: 8 seguidos). Antes, el bucle hacia `sleep(30); continue`
hasta el infinito: como el heartbeat se escribe al inicio de CADA ciclo (main.py:819),
quedaba fresco y el watchdog jamas reiniciaba -> bot caido HORAS con proceso "vivo".
Fix (main.py, bucle de trading): si `verificar_conexion()` falla tras sus 5 intentos,
`os._exit(3)` para que el watchdog relance un proceso limpio (un arranque conecta en
~2s). El connect() inicial tambien lleva timeout (45s). Watchdog: backoff anti-spam, si
el bot muere en <GRACIA segundos no relanza rapido (evita golpear IQ si esta caido y el
arranque falla en cadena).

**SEGUNDO AGUJERO 2026-08-01: el WS puede morir A MITAD de ciclo y el `for` de pares
martillea 20s por activo.** No hace falta que el WS este caido al inicio del ciclo: el
2026-08-01 a las 12:15 el WS cayo ~12:14:45 (se nota en que `_open_time_ciclo` expiro su
join de 35s y se auto-desactivo). Como `_mercado_abierto` devuelve True con datos vacios,
los 49 pares entraron a `get_candles`, y cada uno se colgaba 20s hasta el timeout ->
~16 min dentro del `for` con el heartbeat congelado -> watchdog reinicia a los 613s.
Peor: el hilo `ejecutar_trade` estaba colgado en `check_win_v2` (bloquea hasta el cierre;
con WS muerto no devuelve) -> la operacion de las 12:15 quedo sin `[CIERRE]` para siempre
(IQ si la resolvio: balance 10013.21->10012.21, -1.00).
Fix 1 (main.py:938-947): si `get_candles` falla -> `break` (no `continue`): un solo
timeout sale del `for`, el siguiente ciclo llama `verificar_conexion` -> `os._exit(3)`.
Fix 2 (main.py): persistencia de operaciones en vuelo. `persistir_apertura()` guarda el
oid en `operaciones_pendientes.json` tras cada `[ENTRADA]`; `quitar_pendiente()` lo borra
tras el `[CIERRE]`; al arrancar, `recuperar_pendientes()` consulta cada oid pendiente por
el canal crudo `api.api.get_betinfo` (sin el reconnect interno de la libreria) y loguea
`[CIERRE-RECUP]` si IQ lo responde; si no responde en ~12s, lo descarta (orden vieja: el
balance lo refleja). Asi una caida ya no destruye el resultado de una operacion en vuelo.

Disponibilidad (montado el 2026-07-24, despues de que el PC se reiniciara solo y el bot
pasara 27 min caido sin que nadie lo viera):
- Tarea programada de Windows **`IQOPT-watchdog`**, al iniciar sesion. Verla con
  `Get-ScheduledTask IQOPT-watchdog`; quitarla con `Unregister-ScheduledTask`.
- **Todo corre bajo `pythonw.exe`, y eso NO es cosmetico.** El mismo dia, ya con
  watchdog, el bot volvio a caer sin reinicio del PC: `LastTaskResult` de la tarea era
  `3221225786` = `0xC000013A` = **STATUS_CONTROL_C_EXIT**, o sea un evento de consola de
  un proceso vecino. `DETACHED_PROCESS` solo no bastaba: el `python.exe` del venv es un
  **stub que relanza el interprete base como hijo**, y ese hijo se abria consola propia.
  `pythonw.exe` es del subsistema GUI y no asigna consola nunca. Comprobacion: los 4
  procesos de la cadena deben dar `MainWindowHandle = 0`. Cuidado si se vuelve a
  `python.exe`: sin stdout redirigido, `sys.stdout` es `None` y `print()` peta.
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
- **Resolucion de resultado -> PUSH `option-closed`, NO `check_win_v2/v4`.** `main.py`
  usa `_esperar_cierre()` que vigila `api.api.order_async[oid]["option-closed"]`
  (client.py lo rellena solo con la conexion viva) y devuelve profit neto
  (`profit_amount - amount`: +0.87 / -1.00 sobre stake 1), lo que `_parse_result`
  espera. OJO (medido 2026-08-01): `check_win_v2` -> `get_betinfo` -> `self.connect()`
  interno (sin timeout) MATABA el WebSocket ~10-12s tras cada `[ENTRADA]` y el bot
  entraba en bucle de muerte con el watchdog relanzandolo cada ~6 min. IQ dejo de
  responder `api_game_betinfo`; un buy por si solo NO mata el WS (medido: 83s+ vivo
  tras entrar), y el push llega solo en la marca de cierre. `_esperar_cierre` tiene
  timeout 1000s (duracion real maxima 15 min); si no llega, loguea `[SIN-CIERRE]` y
  la persistencia lo reclama al arrancar.
- expiry<=5 -> `turbo` (~83%, break-even 54.64%); >5 -> `binary` (~87%, 53.48%). Eso es lo
  que se PIDE, y no determina cuanto vive la opcion (ver el punto siguiente).
- **La opcion NO dura `expiry_min`: vence en la siguiente marca de reloj de 15 min**
  (:00, :15, :30, :45), y el payout sigue siendo el de `binary`. Medido el 2026-07-28
  sobre 527 cierres del log: 434 caen en minuto multiplo de 15 y 496 en los primeros 10 s
  del minuto. Como el bot decide al cerrar cada vela de 5m, **la duracion real es 0-15 min
  segun el minuto en que dispare**: media 8.97 min sobre 325 pares entrada-cierre
  inequivocos, y se llego a ver una entrada a las 13:45:55 liquidando a las 13:46.
  Solo el 1.2% de las operaciones duro los 10 min que el modelo predice. El minuto de
  cierre de vela fija el horizonte: `mod 15 == 10` -> H=1, `== 5` -> H=2, `== 0` -> H=3.
  `expiry_alineado()` (main.py:458) filtra por esto y esta ACTIVA (`alinear_expiry: true`,
  tolerancia 1.5 min): solo opera cuando la marca siguiente queda a `horizonte_modelo_min`
  +/- tol (con H=2 y grilla de 15, dispara en el cierre de vela de `:05`, duracion real
  ~10 min). La formula se corrigio el 2026-07-31 (`minutos_al_vencimiento`, main.py:400):
  **IQ asigna la marca siguiente y punto, sin respetar el minimo pedido.** Con la formula
  vieja, pedir 10 a las :20:05 saltaba a :30 (24.9 min reales); hoy se pide 7 para que la
  marca de :30 absorba la latencia y la duracion real quede en ~10. Medido: mod 15 == 5
  -> mediana 9.01 min (el unico bucket que coincide con el horizonte entrenado).
  **Corolario que invalida una conclusion vieja:** aqui ponia que "el horizonte 1 (5 min)
  esta descartado porque el payout se lo come" (turbo 83%). Es falso: pidiendo 10 min se
  compra un `binary` que puede vivir 3 minutos y **se cobra al 87% igual** (profit +0.870
  sobre stake 1.00 en las ganadas de 0-2.5 min). O sea que H=1 vive a break-even 53.48%,
  no 54.64%, y era medible. Se midio el 2026-07-28 (`horizonte_corto_test.py`) y **NO
  aporta**: AUC OOS 0.5253 a H=1 vs 0.5259 a H=2. El horizonte corto no tiene mas senal;
  lo que mata el edge es el RETARDO de la entrada, que es otra cosa. Ver la memoria
  `horizonte-corto-y-brusquedad-2026-07-28`.
- `get_all_profit()` da payouts sin `update_ACTIVES_OPCODE()`. Keys OTC = `"AIG-OTC"`,
  reales = `"EURUSD-op"`.
- **Disponibilidad de activos: `usar_open_time: true`, pero NO via `get_all_open_time()`**
  (que lanza 3 hilos y 2 estan rotos en esta cuenta; el tercero era un endpoint de
  margen retirado). `_open_time_binario()` (main.py:296) lee SOLO `get_all_init_v2()`,
  lo unico que necesitamos (binarias). Cache con `open_time_ttl_seg: 300` (de ~1200 a
  12 consultas/h). Medido el 2026-07-30: sin el filtro el bot pedia ordenes sobre los 49
  activos y solo ejecutaba el 28%, y los 1.675 intentos contra activos cerrados colaban
  la cola haciendo caducar las buenas. Si la consulta falla o devuelve 0 abiertos, el
  filtro se desactiva PARA LA SESION (`_open_time_ciclo()`, main.py:340) y deja que IQ
  decida en el buy. El payout NO sirve de proxy: IQ lo devuelve tambien con el activo
  cerrado.
- `websocket-client` DEBE ser **0.56.0**; versiones nuevas rompen `iqoptionapi`.

## Config: lo que muerde

- Hot-reload cada 30s de `operacion`, `max_trades`, `filtro_hora` y `riesgo`.
  `pares_binarios` y `entrenamiento` **no**: requieren reiniciar.
- **`filtro_hora` es lista blanca en UTC.** `horas_por_par[par]` son las horas
  *permitidas*, y un par ausente queda bloqueado 24h. Activarlo con `horas_por_par`
  vacio **apaga el bot en silencio**. `timezone_offset` solo se usa en un log.
- **`reintento_max_seg: 30`** (era 60, y antes 240). IQ solo acepta ordenes en ventanas de ~3-4 min
  que abren pasado cada cuarto de hora, y el escaneo del bot esta desalineado con ellas
  (ver el comentario largo en `main.py`), asi que algo de reintento hace falta. Se bajo a
  30s el 2026-08-05 (medido en el log): con `alinear_expiry` activo TODOS los fills caen a
  0-25s del cierre de vela, y el chase de 60s no producia NI UN fill tardio -- solo
  martilleaba IQ ~58s sobre senales que nunca aceptaba (27 SKIPs de 6 intentos en agosto).
  Pero **la senal envejece mientras se reintenta** y el modelo predice a 10 min DESDE la
  vela de decision. Medido sobre 262
  operaciones cerradas (2026-07-24, fuera de rollover): inmediatas 60.8% (n=158),
  demoradas por reintentos 50.0% (n=78). z=1.57 -> indicio, no prueba; el argumento de
  fondo es que un fill tardio no es la apuesta que el modelo senalo.
- **BTCUSD fuera de `pares_binarios`** (trampa #5): 86 senales y 0 entradas. Sigue en
  `entrenamiento.pares`, o sea que se entrena pero no se opera.

## Git y entorno

- **Hay dos copias, y una ruta que aparenta ser una tercera no lo es.** Antes de editar,
  mirar desde donde se esta leyendo esto:

  | copia | desde el servidor | desde el PC de trabajo | que es |
  |---|---|---|---|
  | **produccion** | `D:\Proyects\IQOPT` | `Y:\IQOPT` | donde **corre** el bot |
  | **trabajo** | no visible | `D:\GIT\IQOPT` | donde se desarrolla y se entrena |

  `Y:` es el recurso de red `\\10.11.50.163\Proyects`, o sea que **`Y:\IQOPT` y
  `D:\Proyects\IQOPT` son la MISMA carpeta**, no dos. De ahi que la doc se contradiga
  segun quien la escribio: una nota anterior decia que `Y:\IQOPT` "ya no existe", lo cual
  es cierto en el servidor (no tiene ese mapeo) y falso en el PC de trabajo.
  **Consecuencia practica: editar `Y:\` desde el PC toca produccion en caliente**, con el
  bot corriendo. Las dos pueden pushear a `github.com/Arnaldolandin/IQOPT`.
- Cada copia tiene **su propio `.venv314`**, con distinto Python (ver "Como correr").
- Lo gitignored no viaja entre copias y **diverge en silencio**: el 2026-07-24 el PC de
  trabajo tenia un ensemble de 5 semillas por par (`seq_lstm_AIG_s1..s5`) y 29 pares en
  `modelos_por_par`, mientras produccion ya iba con **una sola semilla sin sufijo** y 50
  pares. Al sincronizar hay que traer `config.json` y `models/` a mano; solo con `git`
  el codigo queda al dia apuntando a modelos que no estan.
- Gitignored: `config.json`, **`models/*` entero** (`.pt`, `.npz`, `.pt.json`),
  `cache_ohlc_5m*/`, `heartbeat.json`, `estado_velas.json`. Los modelos se regeneran con
  `train_seq_save.py`; no viajan con `git clone`. Pendiente: rotar password IQ + token
  Telegram (estuvieron en commits viejos).

## Convenciones

- Logging con `print()` + archivo (`rsi_iq.log`). Multi-hilo con `_lock`.
  Telegram en daemon thread.
- Al medir cualquier cosa: corte temporal estricto, embargo, continuidad, sin BTCUSD,
  y **separar rollover**. Sin eso los numeros mienten, y ya mintieron varias veces.
