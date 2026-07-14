# CLAUDE.md

Guia para Claude Code al trabajar en este repositorio.

## Que es esto

Bot de **opciones binarias en IQ Option** que opera **231 activos** (50 real + 181 OTC) con
estrategia **MACD crossover** en velas de 5 minutos. Todo el codigo y los logs estan en espanol.

> **Historia (importante):** este proyecto tenia antes un bot de ~2000 lineas que operaba pares **OTC**
> con 8 estrategias de momentum/reversion. Se elimino porque se demostro que **pierde por diseno**:
> los OTC de IQ son un feed RNG de la casa, ~49.5% WR sobre 2568 ops reales, sin edge.
> No reintroducir esa via. Tambien hubo una fase RSI-reversion sobre USDJPY que no alcanzo
> significancia (OOS ~53.5%, bajo break-even con payout 87%). Abandonada.

## La estrategia actual

**MACD crossover** en velas de 5 minutos **cerradas**:
- **CALL** cuando MACD cruza signal de abajo-arriba.
- **PUT** cuando MACD cruza signal de arriba-abajo.
- MACD(12,26,9) por defecto (configurable en `config.json`).
- Binary 10 minutos de expiracion.
- Payout actual: 87-88%. Break-even ~53.5% WR.
- **231 activos** (50 real + 181 OTC). Sin OTC en historial.
- **Filtro EMA(100)**: solo opera A FAVOR de la tendencia.
- **Multi-hilo**: cada trade en su propio thread, hasta 10 simultaneos.
- **Hot-reload**: stake/expiry/payout/max_trades/MACD se actualizan sin reiniciar.
- **Telegram**: control remoto via comandos.

Backtest (50 real, 16k velas, train/test 70/30): top OOS GBPCAD 5m MACD(8,17,9) exp15m -> 67.2% WR p=0.022.
Backtest sweep MACD x EMA: **ninguna config supera break-even** (53.5%). Mejor: 6,13,5 base WR 47.9%.
Backtest riguroso 57 activos: **todos EV negativo**, mejor NZDJPY WR 52.5%.
⚠️ **No demostrado rentable en vivo todavia.** Mantener en DEMO hasta acumular muestra propia.

## Archivos

- **`main.py`** — el bot. MACD-crossover multi-activo MULTI-HILO, hasta 10 posiciones simultaneas
  (bloquea en `check_win_v4`), verifica payout vivo, filtro EMA, reconexion automatica,
  hot-reload de config. Lee todo de `config.json`. Flags: `--dry` (solo senales),
  sin flag = demo, `--real` (CUIDADO).
- **`config.json`** — credenciales IQ, parametros MACD (fast/slow/signal), operacion (timeframe,
  expiry, stake, min_payout, ema_trend), max_trades, filtro_hora, 231 pares binarios, Telegram.
  ⚠️ Password en texto plano en git.
- **`telegram_commands.py`** — comandos Telegram: /status, /balance, /modo, /activas, /pares,
  /estrategia, /pnl, /config, /ayuda, /setmacd, /setstake, /setmaxtrades, /setexpiry,
  /setpayout, /filtrar on|off, /pausar, /reanudar, /reiniciar.
- **`backtest_riguroso.py`** — backtester MACD-crossover sobre 57 activos con par×hora×dia.
- **`backtest_macd_ema_sweep.py`** — sweep 5 configs MACD x 3 EMAs (50/100/200).
- **`backtest_ema_trend.py`** — test filtro EMA trend.
- **`analisis_par_hora.py`** — analisis par×hora con velas 5m.
- **`listar_otc.py`** — descubre 182 OTC pares con payout.
- **`rsi_iq.log`** — log del bot.

## Como correr

```powershell
.venv314\Scripts\python.exe main.py            # DEMO (231 activos, binary 5m, EMA 100)
.venv314\Scripts\python.exe main.py --dry      # solo loguea senales
```

Usar **`.venv314`** (Python 3.14, con `iqoptionapi`).

## Mecanica de la API de IQ (clave, facil de olvidar)

- **Velas** -> activo subyacente `"USDJPY"` (get_candles).
- **Comprar** -> activo `"USDJPY-op"` (requiere `api.update_ACTIVES_OPCODE()` tras conectar,
  pero **HANGUEA** — se usa workaround con `get_all_profit()` + lista hardcodeada).
- Tras `connect()`: `api.change_balance("PRACTICE")` (demo) o `"REAL"`.
- `api.buy(monto, "USDJPY-op", "call"|"put", minutos)` -> `(status, order_id)`.
- `api.check_win_v4(order_id)` **bloquea** hasta el cierre del contrato.
- El binary de IQ usa expiraciones a horas fijas -> el tiempo real a expiracion varia (~10-15m).
- `get_all_profit()` devuelve payouts sin necesidad de `update_ACTIVES_OPCODE()`.
- OTC pairs: `get_all_profit()` keys son `"AIG-OTC"` (sin `-op`), reales son `"EURUSD-op"`.

## Convenciones

- Logging con `print()` + emojis y archivo (`rsi_iq.log`). Multi-hilo con `_lock` protege
  contadores compartidos. Telegram en daemon thread. Scanea todos los activos del config en
  cada ciclo. Filtro hora en UTC (OTC exentos).
