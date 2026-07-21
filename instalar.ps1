# instalar.ps1 - Instalacion del bot IQOPT en Windows.
#   powershell -ExecutionPolicy Bypass -File instalar.ps1
#
# Crea el venv, instala dependencias y verifica que esten los dos archivos que git
# NO trae (config.json y models\*.pt), que son la causa numero uno de "arranca pero
# nunca opera".

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== 1/4  Python ==" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "Python no esta en el PATH. Instalalo desde python.org (3.11-3.14)." }
$ver = & python -c "import sys;print('%d.%d' % sys.version_info[:2])"
Write-Host "   Python $ver"
if ([version]$ver -lt [version]"3.11") { throw "Se requiere Python 3.11 o superior (tenes $ver)." }

Write-Host "== 2/4  Entorno virtual ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv314\Scripts\python.exe")) {
    & python -m venv .venv314
    Write-Host "   .venv314 creado"
} else {
    Write-Host "   .venv314 ya existe, se reutiliza"
}
$vpy = ".\.venv314\Scripts\python.exe"

Write-Host "== 3/4  Dependencias (torch pesa ~200 MB, paciencia) ==" -ForegroundColor Cyan
& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Fallo la instalacion de dependencias." }

Write-Host "== 4/4  Archivos que git NO trae ==" -ForegroundColor Cyan
$falta = $false

if (-not (Test-Path "config.json")) {
    Write-Host "   [FALTA] config.json  <- copialo desde la maquina de desarrollo" -ForegroundColor Red
    $falta = $true
} else {
    $cfg = Get-Content config.json -Raw | ConvertFrom-Json
    $modelo = $cfg.operacion.seq_model
    Write-Host "   config.json OK  (estrategia=$($cfg.operacion.estrategia), par=$($cfg.operacion.solo_par), thr=$($cfg.operacion.seq_threshold))"
    if (-not (Test-Path $modelo)) {
        Write-Host "   [FALTA] $modelo  <- copiar models\*.pt y *.pt.json" -ForegroundColor Red
        $falta = $true
    } elseif (-not (Test-Path "$modelo.json")) {
        Write-Host "   [FALTA] $modelo.json  (la receta de la red; sin ella no carga)" -ForegroundColor Red
        $falta = $true
    } else {
        Write-Host "   modelo OK  ($modelo)"
    }
}

if ($falta) {
    Write-Host "`nInstalacion INCOMPLETA. Copia los archivos marcados y volve a correr este script." -ForegroundColor Yellow
    Write-Host "Sin ellos el bot arranca pero nunca opera (err seq: FileNotFoundError en cada vela)."
    exit 1
}

Write-Host "`nListo. Probalo sin comprar nada:" -ForegroundColor Green
Write-Host "   powershell -ExecutionPolicy Bypass -File correr.ps1 -Dry"
Write-Host "`nRECORDATORIO: el modelo NO tiene ventaja demostrada (53.64% vs break-even 53.48%)." -ForegroundColor Yellow
Write-Host "Usar SOLO en cuenta demo."
