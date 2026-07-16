# CLAUDE.md

Guia para Claude Code al trabajar en este repositorio.

## Que es esto

Bot de **opciones binarias en IQ Option** que opera **231 activos** (50 real + 181 OTC).
Timeframe y expiracion configurables (por defecto velas 1m / expiracion 2m turbo).
Todo el codigo y los logs estan en espanol.

> **Historia (importante):** este proyecto probo exhaustivamente (2026-05 a 2026-07) muchas vias
> — MACD momentum, filtros EMA/pendiente, ATR, reversion (RSI/Bollinger), multi-timeframe,
> por par, por hora, walk-forward y analisis de payout. **Conclusion probada con rigor:
> ninguna config es rentable en IQ.** El momentum pierde (~47-49% WR). La reversion tiene un
> edge direccional real (~52%) pero el payout tope de IQ (max 88%, break-even 53.5%) se lo come.
> El muro es estructural (el payout), no la estrategia. **Mantener en DEMO.**
> Ver memoria `estrategia-sin-edge-iq-2026-07-15`. Los OTC son un feed RNG de la casa (~50%).

## Estrategias (seleccionables en `config.json` -> `operacion.estrategia`)

**`macd`** — cruce MACD puro sobre velas cerradas:
- CALL cuando MACD cruza signal de abajo-arriba; PUT al reves. MACD(`macd_fast`,`macd_slow`,`macd_signal`).
- **Filtro EMA opcional** (`macd_ema`, 0=off): solo opera A FAVOR de la EMA
  (CALL si precio>EMA, PUT si precio<EMA).

**`bb_rev`** — reversion con Bandas de Bollinger:
- CALL cuando el precio cruza por DEBAJO de la banda inferior (rebote); PUT sobre la superior.
- Bollinger(`bb_period`, `bb_k`).

Ambas comparten:
- **Filtro ATR opcional** (`min_atr`, fraccion del precio; 0=off): evita rangos de baja volatilidad.
- **Multi-hilo**: cada trade en su hilo, hasta `max_trades` simultaneos (bloquea en `check_win_v4`).
- Verifica payout vivo, reconexion automatica, hot-reload de config, control por Telegram.
- **Limites de riesgo** aplicados: `max_perdida_diaria` y `max_operaciones_hora`.
- El bot baja **~5x el periodo del indicador** para que la EMA CONVERJA (bug historico: con pocas
  velas la EMA arrastra el seed y decide mal ~4% de los cruces). Log `[ENTRADA]` audita cada trade.

## Archivos

- **`main.py`** — el bot. Dispatcher de estrategia (macd / bb_rev) + filtro ATR, multi-hilo,
  reconexion, hot-reload. Flags: `--dry` (solo senales), sin flag = demo, `--real` (CUIDADO).
- **`config.json`** — credenciales IQ (⚠️ texto plano, gitignored/untracked), `operacion`
  (timeframe_seg, expiry_min, stake, min_payout, estrategia, bb_period/bb_k, macd_fast/slow/signal,
  macd_ema, atr_period/min_atr), max_trades, riesgo, filtro_hora, 231 pares, Telegram.
- **`config.example.json`** — plantilla sin secretos.
- **`telegram_commands.py`** — comandos: /status, /balance, /modo, /activas, /pares, /estrategia,
  /pnl, /config, /ayuda, /setestrategia, /bb, /setbb, /macd, /setmacd, /setmacdema, /atr, /setatr,
  /setstake, /setmaxtrades, /setexpiry, /setpayout, /filtrar, /setpoll, /pausar, /reanudar, /reiniciar.
- **Backtests (framework riguroso, train/test + OOS + control de azar):**
  - `backtest_horas_ema_slope.py` — sweep EMA×pendiente + par×hora, OOS.
  - `backtest_multi_tf.py` — barrido multi-timeframe (5/10/15/30m).
  - `backtest_30m_valida.py` — validacion 30m con OOS de 2 meses.
  - `backtest_atr_sweep.py` — efecto del umbral ATR.
  - `backtest_reversion.py` — fade-MACD / RSI / Bollinger.
  - `backtest_walkforward_rev.py` — walk-forward mes a mes de la reversion.
  - `backtest_combo.py` — RSI vs BB vs union vs confluencia.
- **Descargadores** (a `cache_*/`, por span de tiempo, resumibles):
  `download_ohlc.py` / `download_ohlc_1m.py` (1m) / `download_ohlc_3m.py` (5m) / `download_ohlc_30m.py`.
- **`listar_otc.py`** — descubre OTC con payout.
- **`cache_ohlc/`** (5m ~1mes, 231), **`cache_ohlc_1m/`** (1m ~59d, 50 real),
  **`cache_ohlc_30m/`** (30m ~209d, 50 real), **`cache_closes/`**. IQ da ~10 meses de 5m, ~59d de 1m.
- **`rsi_iq.log`** — log del bot (gitignored).

## Como correr

```powershell
.venv314\Scripts\python.exe main.py            # DEMO (usa la estrategia de config.json)
.venv314\Scripts\python.exe main.py --dry      # solo loguea senales
```

Usar **`.venv314`** (Python 3.14, con `iqoptionapi`; el `.venv` esta roto).
Cambiar de estrategia sin reiniciar NO es posible para cambios de *codigo*, pero SI para
valores (config.json se hot-reloadea). Cambiar de `estrategia` en config entra por hot-reload.

## Mecanica de la API de IQ (clave, facil de olvidar)

- **Velas** -> activo subyacente `"USDJPY"` (get_candles).
- **Comprar** -> `api.buy(monto, "USDJPY-op", "call"|"put", minutos)` -> `(status, order_id)`.
  Requiere `get_ALL_Binary_ACTIVES_OPCODE()` tras conectar (puede colgar -> se usa con timeout).
- Tras `connect()`: `api.change_balance("PRACTICE")` (demo) o `"REAL"`.
- `api.check_win_v4(order_id)` **bloquea** hasta el cierre del contrato.
- expiry<=5 -> instrumento `turbo`; >5 -> `binary`. Binary paga mas que turbo (~87% vs ~83%).
- `get_all_profit()` da payouts sin `update_ACTIVES_OPCODE()`. Keys OTC = `"AIG-OTC"`, reales = `"EURUSD-op"`.

## Git y entorno

- **Dos copias locales**: `D:\GIT\IQOPT` (donde se corre) y `Y:\IQOPT` (canonico, pushea a
  `github.com/Arnaldolandin/IQOPT`). Historias divergentes: **D NO puede pushear**; editar en D,
  copiar a Y, commitear en ambos, `push` solo Y. Ver memoria `estado-github`.
- `config.json` gitignored/untracked (tiene credenciales). Pendiente: rotar password IQ + token Telegram.

## Convenciones

- Logging con `print()` + archivo (`rsi_iq.log`). Multi-hilo con `_lock`. Telegram en daemon thread.
  Filtro hora en UTC (OTC exentos). Escanea todos los activos del config cada ciclo.
