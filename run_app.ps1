# ============================================================
#  Option Pricing Tool - launcher (PowerShell)
#
#  Starts the production server (waitress, debugger off) via serve.py.
#
#  Usage:  .\run_app.ps1
#          .\run_app.ps1 -Port 5050
#          $env:OTCPRICER_PYTHON = 'C:\path\to\python.exe'; .\run_app.ps1
#
#  Three things this used to get wrong
#  -----------------------------------
#  It required the conda env "Quant1.0" and aborted unless py_vollib imported
#  there. py_vollib was removed in Phase 1 -- the pricing moved to the
#  project's own otcpricer.models.bsm -- so the hard gate was on a package the
#  code does not use, and the header claimed Python 3.13 "dies" without it.
#  Both 3.12 and 3.13 work and both are tested.
#
#  It pointed at a different interpreter from the one the project is tested
#  on, so the documented launch path was not the verified path.
#
#  It inlined the dependency probe as a here-string, and PowerShell stripped
#  the quotes out of it. The probe died with a SyntaxError and the launcher
#  blamed the interpreter. The probe is now scripts/check_deps.py -- a file,
#  with no quoting layer to get wrong, shared with run_app.bat.
#
#  It now tries each candidate interpreter and uses the first that can
#  actually run the app, rather than the first that merely exists.
# ============================================================

param(
    [int]$Port = 0,
    [string]$BindHost = ''
)

# Deliberately NOT 'Stop'. In Windows PowerShell 5.1 a native exe's stderr
# becomes a terminating NativeCommandError under 'Stop', so any library that
# writes a warning on import would abort the launcher.
$ErrorActionPreference = 'Continue'

$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot

if ($Port -ne 0)      { $env:OTCPRICER_PORT = "$Port" }
if ($BindHost -ne '') { $env:OTCPRICER_HOST = $BindHost }

# --- Candidate interpreters, in order of preference --------------------
$Candidates = @()
if ($env:OTCPRICER_PYTHON) { $Candidates += $env:OTCPRICER_PYTHON }
$resolved = Get-Command python -ErrorAction SilentlyContinue
if ($resolved) { $Candidates += $resolved.Source }
$Candidates += 'E:\anaconda\envs\Quant1.0\python.exe'   # developed against; a convenience now

# --- Use the first that passes the probe -------------------------------
$EnvPython = $null
$Reports = @()
foreach ($candidate in $Candidates) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $report = & $candidate scripts/check_deps.py 2>&1
    if ($LASTEXITCODE -eq 0) {
        $EnvPython = $candidate
        Write-Host ($report -join "`n") -ForegroundColor DarkGray
        break
    }
    $Reports += "--- $candidate"
    $Reports += ($report -join "`n")
}

if (-not $EnvPython) {
    Write-Host "[ERROR] No interpreter here can run the application." -ForegroundColor Red
    if ($Reports.Count -gt 0) {
        Write-Host ($Reports -join "`n")
    } else {
        Write-Host "        No Python interpreter was found at all."
        Write-Host "        Set one:  `$env:OTCPRICER_PYTHON = 'C:\path\to\python.exe'"
    }
    exit 1
}

# -u so the banner reaches the console before the first request rather than
# sitting in a pipe buffer.
& $EnvPython -u serve.py
