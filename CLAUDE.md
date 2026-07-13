# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio.

## Qué es esto

Bot de **opciones binarias en IQ Option** que opera **51 activos binarios reales** (no OTC) con
estrategia **MACD crossover** en velas de 5 minutos. Todo el código y los logs están en español.

> **Historia (importante):** este proyecto tenía antes un bot de ~2000 líneas que operaba pares **OTC**
> con 8 estrategias de momentum/reversión. Se eliminó porque se demostró que **pierde por diseño**:
> los OTC de IQ son un feed RNG de la casa, ~49.5% WR sobre 2568 ops reales (−$20.837), sin edge.
> No reintroducir esa vía. También hubo una fase RSI-reversión sobre USDJPY que no alcanzó
> significancia (OOS ~53.5%, bajo break-even con payout 87%). Abandonada.

## La estrategia actual

**MACD crossover** en velas de 5 minutos **cerradas**:
- **CALL** cuando MACD cruza signal de abajo-arriba.
- **PUT** cuando MACD cruza signal de arriba-abajo.
- MACD(12,26,9) por defecto (configurable en `config.json`).
- Binary 10 minutos de expiración.
- Payout actual: 87-88%. Break-even ~53.5% WR.
- **51 activos reales** (forex, commodities, crypto, acciones). Sin OTC.

Backtest (16k velas, train/test 70/30): top OOS GBPCAD 5m MACD(8,17,9) exp15m → 67.2% WR p=0.022.
⚠️ **No demostrado rentable en vivo todavía.** Mantener en DEMO hasta acumular muestra propia.

## Archivos

- **`main.py`** — el bot. MACD-crossover multi-activo, una posición a la vez (bloquea en
  `check_win_v4`), verifica payout vivo (salta si < min_payout). Lee todo de `config.json`.
  Flags: `--dry` (solo señales), sin flag = demo, `--real` (CUIDADO).
- **`config.json`** — credenciales IQ, parámetros MACD (fast/slow/signal), operación (timeframe,
  expiry, stake, min_payout), 51 pares binarios, Telegram. ⚠️ Password en texto plano en git.
- **`backtest_macd.py`** — backtester MACD-crossover sobre todos los activos binarios reales.
  Train/test 70/30, p-valor, walk-forward integrado.
- **`backtest_iq.py`** — backtester RSI-reversión (legacy, para referencia).
- **`walkforward.py`** — validación out-of-sample RSI (legacy).
- **`ranking.py`** — ranking multi-estrategia (legacy).
- **`listar_payouts.py`** — lista activos binarios y payouts.
- **`rsi_iq.log`** — log del bot.

## Cómo correr

```powershell
.venv314\Scripts\python.exe main.py            # DEMO (51 activos, binary 10m)
.venv314\Scripts\python.exe main.py --dry      # solo loguea señales
```

Usar **`.venv314`** (Python 3.14, con `iqoptionapi`).

## Mecánica de la API de IQ (clave, fácil de olvidar)

- **Velas** → activo subyacente `"USDJPY"` (get_candles).
- **Comprar** → activo `"USDJPY-op"` (requiere `api.update_ACTIVES_OPCODE()` tras conectar,
  pero **HANGUEA** — se usa workaround con `get_all_profit()` + lista hardcodeada).
- Tras `connect()`: `api.change_balance("PRACTICE")` (demo) o `"REAL"`.
- `api.buy(monto, "USDJPY-op", "call"|"put", minutos)` → `(status, order_id)`.
- `api.check_win_v4(order_id)` **bloquea** hasta el cierre del contrato.
- El binary de IQ usa expiraciones a horas fijas → el tiempo real a expiración varía (~10-15m).
- `get_all_profit()` devuelve payouts sin necesidad de `update_ACTIVES_OPCODE()`.

## Convenciones

- Logging con `print()` + emojis y archivo (`rsi_iq.log`). Una posición a la vez (el bot bloquea en
  `check_win_v4`). Scanea todos los activos del config en cada ciclo.
