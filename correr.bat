@echo off
REM correr.bat - Lanza el bot IQOPT. Doble clic o desde cmd.
REM
REM   correr.bat          DEMO
REM   correr.bat --dry    solo loguea senales, no compra
REM   correr.bat --real   CUIDADO: dinero real
REM
REM No usa PowerShell a proposito: evita el lio de ExecutionPolicy en el servidor.

cd /d "%~dp0"

REM Consola en UTF-8. Los logs estan en espanol y sin esto el bot muere con
REM UnicodeEncodeError en consolas cp1252, que es lo habitual en Windows Server.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Detectar el venv: en algunas maquinas se llama .venv y en otras .venv314.
REM Activar el que no tiene torch da "err seq: ModuleNotFoundError" en cada vela
REM sin que el bot se caiga, asi que el sintoma no apunta al venv equivocado.
set VPY=
if exist ".venv314\Scripts\python.exe" set VPY=.venv314\Scripts\python.exe
if not defined VPY if exist ".venv\Scripts\python.exe" set VPY=.venv\Scripts\python.exe

if not defined VPY (
    echo [ERROR] No se encontro ningun entorno virtual ^(.venv314 ni .venv^)
    echo Corre primero: instalar.bat
    pause
    exit /b 1
)

REM Verificar que el venv elegido tenga torch, no solo que exista.
"%VPY%" -c "import torch" 2>nul
if errorlevel 1 (
    echo [ERROR] El entorno %VPY% no tiene torch instalado.
    echo Corre: instalar.bat   ^(o^)   "%VPY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Entorno: %VPY%

if not exist "config.json" (
    echo [ERROR] Falta config.json ^(tiene las credenciales, no viaja con git^)
    echo Copialo desde la maquina de desarrollo.
    pause
    exit /b 1
)

if not exist "models\seq_lstm_EURUSD.pt" (
    echo [ERROR] Falta models\seq_lstm_EURUSD.pt
    echo Sin el modelo el bot arranca pero NUNCA opera.
    echo Copia models\*.pt y models\*.pt.json desde la maquina de desarrollo.
    pause
    exit /b 1
)

echo.
echo === Bot IQOPT (estrategia seq / LSTM) ===
echo El modelo NO tiene ventaja demostrada: 53.64%% contra un break-even de 53.48%%.
echo Usar en cuenta DEMO. Ctrl+C para detener.
echo.

"%VPY%" main.py %*

echo.
echo El bot se detuvo.
pause
