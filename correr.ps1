# correr.ps1 - Lanza el bot con la codificacion correcta.
#   powershell -ExecutionPolicy Bypass -File correr.ps1        # DEMO
#   powershell -ExecutionPolicy Bypass -File correr.ps1 -Dry   # solo loguea senales
#   powershell -ExecutionPolicy Bypass -File correr.ps1 -Real  # CUIDADO: dinero real
#
# PYTHONIOENCODING=utf-8 no es opcional: los logs estan en espanol y sin eso el bot
# muere con UnicodeEncodeError en consolas con codificacion heredada (cp1252), que es
# lo habitual en un servidor Windows.
param(
    [switch]$Dry,
    [switch]$Real
)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$vpy = ".\.venv314\Scripts\python.exe"

if (-not (Test-Path $vpy)) {
    Write-Host "No existe el venv. Corre primero: instalar.ps1" -ForegroundColor Red
    exit 1
}

$args = @()
if ($Dry)  { $args += "--dry" }
if ($Real) {
    Write-Host "MODO REAL: se opera con dinero de verdad." -ForegroundColor Red
    Write-Host "El modelo mide 53.64% contra un break-even de 53.48%: la expectativa es perder."
    $r = Read-Host "Escribi SI para continuar"
    if ($r -ne "SI") { Write-Host "Cancelado."; exit 0 }
    $args += "--real"
}

& $vpy main.py @args
