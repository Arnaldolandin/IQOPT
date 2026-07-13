# CLAUDE.md

Guía para Claude Code al trabajar en este repositorio.

## Qué es esto

Bot de **opciones binarias en IQ Option** enfocado en el **único edge validado**: RSI-reversión
sobre **USDJPY real** (no OTC). Todo el código y los logs están en español.

> **Historia (importante):** este proyecto tenía antes un bot de ~2000 líneas (`main.py`) que operaba
> pares **OTC** con 8 estrategias de momentum/reversión. Se eliminó porque se demostró que **pierde por
> diseño**: los OTC de IQ son un feed RNG de la casa, ~49.5% WR sobre 2568 ops reales (−$20.837), sin
> edge en ningún segmento (par/hora/día). No reintroducir esa vía.

## El edge (lo único que gana)

RSI(14) sobre velas de 1 minuto **cerradas**: **CALL si RSI < 35, PUT si RSI > 65**. Reversión.
- **Solo USDJPY.** EURUSD y GBPJPY no tienen edge (backtest negativo), igual que en el bot de Deriv.
- Validado en dos feeds independientes (Deriv frxUSDJPY 57.1% WR n=869 p=0.003; IQ USDJPY ~57-58%).
- ⚠️ **No demostrado rentable en IQ todavía:** el payout de IQ (83-84%) sube el break-even a ~54.3-54.6%,
  y la muestra de IQ (7.6 días) no alcanza significancia (10m p=0.062, 5m p=0.30). Es reversión →
  riesgo de régimen (pierde en tendencia fuerte). **DEMO hasta acumular muestra propia. No pasar a real.**

## Archivos

- **`rsi_iq_bot.py`** — el bot. RSI-reversión sobre USDJPY, una posición a la vez, verifica payout vivo
  (salta si < `MIN_PAYOUT`). Autónomo (no depende de otros módulos). Default **binary 10m** (mejor margen
  del backtest: 58.8% WR). Flags: `--dry` (solo señales), sin flag = demo, `--real` (CUIDADO), `--stake`,
  `--expiry` (≤5 turbo, >5 binary).
- **`backtest_iq.py`** — RSI-reversión des-solapado sobre velas históricas reales; WR vs break-even por horizonte.
- **`backtest_iq_robustez.py`** — WR de USDJPY por ventanas (estabilidad del edge).
- **`walkforward.py`** — validación out-of-sample: elige umbral/horizonte in-sample y lo mide fuera de
  muestra. El 2026-07-13 dio 7.5pt de sobreajuste (IS 65% → OOS 57.5%, p=0.130) y la config del bot
  (30/70@10m) bajo break-even fuera de muestra. Correr antes de fiarse de cualquier WR de backtest.
- **`listar_payouts.py`** — lista activos binarios abiertos y payouts (real vs OTC).
- **`config.json`** — credenciales IQ (`email`/`password`) + `telegram`. ⚠️ Está en git con la password
  en texto plano — pendiente de rotar y sacar a variables de entorno.

## Cómo correr

```powershell
.venv314\Scripts\python.exe rsi_iq_bot.py            # DEMO (binary 10m)
.venv314\Scripts\python.exe rsi_iq_bot.py --dry      # solo loguea señales
.venv314\Scripts\python.exe backtest_iq.py           # revalidar el edge
```

Usar **`.venv314`** (Python 3.14, con `iqoptionapi`). El `.venv` viejo (Python 3.13) se eliminó: estaba roto.

## Mecánica de la API de IQ (clave, fácil de olvidar)

- **Velas / RSI** → activo `"USDJPY"` (subyacente real, `get_candles`).
- **Comprar** → activo `"USDJPY-op"` y **requiere `api.update_ACTIVES_OPCODE()` tras conectar** (si no:
  "asset not found on consts"). Turbo si expiración ≤5 min, binary si >5.
- Tras `connect()`: `api.change_balance("PRACTICE")` (demo) o `"REAL"`.
- `api.buy(monto, "USDJPY-op", "call"|"put", minutos)` → `(status, order_id)`.
- `api.check_win_v4(order_id)` **bloquea** hasta el cierre del contrato.
- El binary de IQ usa expiraciones a horas fijas → el tiempo real a expiración puede variar (~10-15m).

## Convenciones

- Logging con `print()` + emojis y archivo (`rsi_iq.log`). Una posición a la vez (el bot bloquea en
  `check_win_v4`). Solo opera en horario de mercado de USD/JPY (fuera de él, la compra falla y se reintenta).
