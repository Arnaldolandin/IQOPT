# Despliegue del bot IQOPT en un servidor

Bot de opciones binarias en IQ Option. Estrategia **`seq`**: un modelo secuencial
(LSTM) mira las últimas 64 velas de 5m y devuelve `P(sube)` a horizonte 2 velas
(expiración 10 min, instrumento `binary`, payout ~87%).

> **ANTES DE EMPEZAR — el modelo no tiene ventaja demostrada.**
> Medido en held-out con corte temporal estricto: **53.64% de aciertos a umbral 0.54**
> contra un **break-even de 53.48%**. El margen (0.16 puntos) está muy por dentro del
> error estadístico. La pérdida de validación es `0.6918` cuando el azar puro da
> `ln(2) = 0.6931`: el modelo apenas se despega de tirar una moneda.
> **Desplegar solo en cuenta DEMO.** Con dinero real la expectativa es perder.

---

## 1. Requisitos

- **Python 3.11 - 3.14** (se probó en 3.14.4).
- Salida a internet (websockets hacia `iqoption.com`).
- ~1 GB de disco; ~500 MB de RAM. Sin GPU: el modelo corre en CPU de sobra.
- Cuenta de IQ Option.

## 2. Archivos a copiar

**Imprescindibles para operar:**

```
main.py                        el bot
seq_model.py                   ventana de features + carga del modelo
telegram_commands.py           control por Telegram
config.json                    credenciales y parámetros  (NO está en git)
models/seq_lstm_EURUSD.pt      pesos del modelo           (NO está en git)
models/seq_lstm_EURUSD.pt.json receta para reconstruir la red
requirements.txt
```

**Solo si vas a reentrenar en el servidor:**

```
train_seq_save.py
eurusd_seq.py                  compara LSTM vs Transformer vs baseline
download_ohlc_5m.py            baja las velas
actualizar_cache_5m.py         las mantiene al día
cache_ohlc_5m/                 73 MB de velas (se puede regenerar; ver §6)
```

> `models/*.pt` y `config.json` están en `.gitignore`. Un `git clone` **no** los trae:
> hay que copiarlos a mano (scp / AnyDesk / lo que uses).

## 3. Instalación

```bash
git clone https://github.com/Arnaldolandin/IQOPT.git
cd IQOPT
python -m venv .venv314
.venv314/bin/python -m pip install --upgrade pip
.venv314/bin/python -m pip install -r requirements.txt
```

En Windows: `.venv314\Scripts\python.exe` en lugar de `.venv314/bin/python`.

Después copiá `config.json` y la carpeta `models/`.

**Detalle que rompe la instalación si se ignora:** `iqoptionapi` no está en PyPI, se
instala desde git y **exige `websocket-client==0.56.0`**. Si algo actualiza esa
dependencia, el bot deja de conectar. Está fijada en `requirements.txt`.

## 4. Configuración (`config.json`)

```jsonc
{
  "email": "...", "password": "...",        // credenciales IQ, en texto plano
  "operacion": {
    "timeframe_seg": 300,                   // velas de 5m (el modelo se entrenó así)
    "expiry_min": 10,                       // 10m -> binary (~87%). <=5 iría a turbo (~83%)
    "stake": 100.0,
    "min_payout": 0.87,
    "estrategia": "seq",
    "seq_model": "models/seq_lstm_EURUSD.pt",
    "seq_threshold": 0.54,                  // CALL si P>=0.54, PUT si P<=0.46
    "solo_par": "EURUSD",                   // el modelo se entrenó SOLO con este par
    "excluir_otc": true,
    "atr_period": 14, "min_atr": 0.0
  },
  "entrenamiento": { ... },                 // solo lo lee train_seq_save.py
  "max_trades": 10,
  "riesgo": { "max_perdida_diaria": 10000, "max_operaciones_hora": 10000 },
  "filtro_hora": { "habilitado": false, "timezone_offset": -4, "horas_por_par": {} },
  "pares_binarios": [ ... ],
  "telegram": { "habilitado": true, "token": "...", "chat_id": "..." }
}
```

Se hot-reloadea cada 30 s: `operacion`, `max_trades`, `filtro_hora` y `riesgo` se
pueden cambiar **sin reiniciar**. `pares_binarios` y `entrenamiento`, no.

**`filtro_hora` es lista blanca**: `horas_por_par[par]` son las horas *permitidas* en
**UTC**, y un par ausente del diccionario queda bloqueado las 24 h. Si activás
`habilitado: true` con `horas_por_par` vacío, **el bot deja de operar en silencio**.
El campo `timezone_offset` solo se usa en un mensaje de log, no para filtrar.

## 5. Ejecución

```bash
.venv314/bin/python main.py            # DEMO
.venv314/bin/python main.py --dry      # solo loguea señales, no compra
.venv314/bin/python main.py --real     # CUIDADO: dinero real
```

Como servicio systemd (Linux):

```ini
[Unit]
Description=Bot IQOPT (seq)
After=network-online.target

[Service]
WorkingDirectory=/opt/IQOPT
ExecStart=/opt/IQOPT/.venv314/bin/python main.py
Restart=always
RestartSec=30
User=iqopt
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=multi-user.target
```

`PYTHONIOENCODING=utf-8` no es opcional: los logs están en español y sin eso
petan en consolas con codificación heredada.

## 6. Reentrenar en el servidor

```bash
.venv314/bin/python download_ohlc_5m.py     # ~6 meses de velas 5m (reanudable)
.venv314/bin/python train_seq_save.py       # lee config.json -> "entrenamiento"
```

Toma ~1 minuto por par. Ajustá épocas, arquitectura, tamaño de ventana e
hiperparámetros en el bloque `entrenamiento` del config; los flags de CLI
(`--arq transformer`, `--epocas 100`) mandan sobre el config.

Cada modelo guarda su propia receta (`arq`, `L`, `hp`) en el `.pt.json`. Por eso
podés cambiar `hidden` o `dropout` sin que los modelos viejos dejen de cargarse.

**Si cambiás `seq_model.ventana_features()`, hay que reentrenar.** El modelo guardado
está atado a esa versión del vector de features. Si el bot y el entrenamiento
construyen ventanas distintas, no falla con un error: sigue devolviendo
probabilidades de aspecto razonable pero sin ningún significado.

## 7. Control por Telegram

`/status`, `/balance`, `/activas`, `/pnl`, `/config`, `/estrategia`, `/ayuda`,
`/setstake`, `/setumbral`, `/setmaxtrades`, `/setexpiry`, `/setpayout`, `/setatr`,
`/filtrar on|off`, `/pausar`, `/reanudar`, `/reiniciar`.

## 8. Qué vas a ver al operar

- **Pocas operaciones.** Con `solo_par: EURUSD` se escanea un activo y el umbral
  dispara en ~5 de cada 100 velas: unas pocas señales por día.
- **Casi todas PUT.** El modelo quedó sesgado a la baja (la tasa base del período de
  entrenamiento fue 49.15% de subidas), así que su `P` rara vez supera 0.54 por
  arriba pero sí baja de 0.46.
- **Rachas.** Con `max_trades: 10` y un solo par, puede abrir varias posiciones sobre
  el mismo movimiento. No hay control de correlación: 10 stakes simultáneos pueden
  ser en la práctica una sola apuesta.

## 9. Seguridad

- `config.json` guarda **email, password de IQ y token de Telegram en texto plano**.
  Está en `.gitignore`, pero en el servidor conviene `chmod 600`.
- Hay credenciales en commits antiguos del repositorio: **rotá la contraseña de IQ y
  el token de Telegram**.
- No expongas el directorio del bot en ningún servicio web.
