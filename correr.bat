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

set VPY=.venv314\Scripts\python.exe

if not exist "%VPY%" (
    echo [ERROR] No existe el entorno virtual .venv314
    echo Corre primero: instalar.bat
    pause
    exit /b 1
)

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
