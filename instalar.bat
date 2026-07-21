@echo off
REM instalar.bat - Instalacion del bot IQOPT en Windows. Doble clic o desde cmd.
REM
REM Crea el entorno virtual, instala dependencias y verifica que esten los dos
REM archivos que git NO trae (config.json y models\*.pt). Su ausencia es la causa
REM numero uno de "arranca pero nunca opera".

cd /d "%~dp0"
chcp 65001 >nul
setlocal

echo == 1/4  Python ==
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta en el PATH.
    echo Instalalo desde python.org ^(3.11 a 3.14^) y marca "Add to PATH".
    pause
    exit /b 1
)
REM Sin '%' en la linea de Python: en un .bat cmd lo interpreta como variable
REM y se come el formato ('%d.%d' llegaba a Python como 'd').
python -c "import sys;assert sys.version_info>=(3,11);print('   Python',sys.version.split()[0])"
if errorlevel 1 (
    echo [ERROR] Se requiere Python 3.11 o superior.
    pause
    exit /b 1
)

echo == 2/4  Entorno virtual ==
if exist ".venv314\Scripts\python.exe" (
    echo    .venv314 ya existe, se reutiliza
) else (
    python -m venv .venv314
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo    .venv314 creado
)
set VPY=.venv314\Scripts\python.exe

echo == 3/4  Dependencias ^(torch pesa ~200 MB, esto tarda^) ==
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo == 4/4  Archivos que git NO trae ==
set FALTA=0

if exist "config.json" (
    echo    config.json OK
) else (
    echo    [FALTA] config.json  ^<- copialo desde la maquina de desarrollo
    set FALTA=1
)

if exist "models\seq_lstm_EURUSD.pt" (
    if exist "models\seq_lstm_EURUSD.pt.json" (
        echo    modelo OK
    ) else (
        echo    [FALTA] models\seq_lstm_EURUSD.pt.json  ^(receta de la red^)
        set FALTA=1
    )
) else (
    echo    [FALTA] models\seq_lstm_EURUSD.pt
    set FALTA=1
)

echo.
if "%FALTA%"=="1" (
    echo Instalacion INCOMPLETA. Copia los archivos marcados y volve a correr este script.
    echo Sin ellos el bot arranca pero nunca opera.
    pause
    exit /b 1
)

echo Listo. Para probar sin comprar nada:
echo    correr.bat --dry
echo.
echo RECORDATORIO: el modelo NO tiene ventaja demostrada
echo ^(53.64%% contra un break-even de 53.48%%^). Usar SOLO en demo.
pause
