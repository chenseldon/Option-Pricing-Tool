# ============================================================
#  Option Pricing Tool - Web App launcher (PowerShell)
#
#  This project runs on the conda env "Quant1.0" (Python 3.12),
#  the only env on this machine with py_vollib installed.
#  Plain `python app.py` falls back to Anaconda base (3.13) and
#  dies with: ModuleNotFoundError: No module named 'py_vollib'
#
#  Usage:  .\run_app.ps1
# ============================================================

# NOTE: deliberately NOT 'Stop'. In Windows PowerShell 5.1, a native exe's
# stderr becomes a terminating NativeCommandError under 'Stop' -- and
# py_vollib emits a DeprecationWarning on import, which would abort us.
# For the same reason, never redirect the probe's stderr below.
$ErrorActionPreference = 'Continue'

$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot

$EnvPython = 'E:\anaconda\envs\Quant1.0\python.exe'

if (-not (Test-Path -LiteralPath $EnvPython)) {
    Write-Host "[ERROR] Interpreter not found: $EnvPython" -ForegroundColor Red
    Write-Host "        Rebuild: conda create -n Quant1.0 python=3.12; conda activate Quant1.0; pip install -r requirements.txt"
    exit 1
}

# -W ignore keeps the py_vollib DeprecationWarning off stderr entirely.
$probe = & $EnvPython -W ignore -c "import py_vollib, flask, numpy, scipy, openpyxl; print('deps-ok')"
if ($LASTEXITCODE -ne 0 -or $probe -ne 'deps-ok') {
    Write-Host "[ERROR] Dependencies missing or broken in Quant1.0." -ForegroundColor Red
    Write-Host "        Fix: & '$EnvPython' -m pip install -r requirements.txt"
    exit 1
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  Starting on http://127.0.0.1:5000     (Ctrl+C to stop)' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

& $EnvPython app.py
