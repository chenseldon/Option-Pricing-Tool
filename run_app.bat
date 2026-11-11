@echo off
rem ============================================================
rem  Option Pricing Tool - launcher
rem
rem  Starts the production server (waitress, debugger off) via serve.py.
rem
rem  Usage:  run_app.bat
rem          set OTCPRICER_PORT=5050 && run_app.bat
rem          set OTCPRICER_PYTHON=C:\path\to\python.exe && run_app.bat
rem
rem  This used to activate the conda env "Quant1.0" and abort unless
rem  py_vollib imported there. Both are stale: py_vollib was removed in
rem  Phase 1 when the pricing moved to the project's own
rem  otcpricer.models.bsm, so the gate was on a package the code does not
rem  use -- and the env it insisted on is not the one the project is tested
rem  against. Python 3.12 and 3.13 both work and both are tested.
rem
rem  It now tries each candidate interpreter against
rem  scripts/check_deps.py and uses the first that can actually run the
rem  app, rather than the first that merely exists.
rem ============================================================
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if defined OTCPRICER_PYTHON (
    call :try "%OTCPRICER_PYTHON%"
    if not errorlevel 1 goto :run
)

where python >nul 2>nul
if not errorlevel 1 (
    call :try python
    if not errorlevel 1 goto :run
)

if exist "E:\anaconda\envs\Quant1.0\python.exe" (
    call :try "E:\anaconda\envs\Quant1.0\python.exe"
    if not errorlevel 1 goto :run
)

echo.
echo [ERROR] No interpreter here can run the application.
echo         Set one explicitly:  set OTCPRICER_PYTHON=C:\path\to\python.exe
echo         Or install the dependencies:  python -m pip install -r requirements.txt
pause
exit /b 1

:try
rem Probe one candidate. Leaves PY set and errorlevel 0 when it passes.
"%~1" scripts\check_deps.py
if errorlevel 1 exit /b 1
set "PY=%~1"
exit /b 0

:run
rem -u so the banner appears immediately rather than sitting in a buffer.
"%PY%" -u serve.py
pause
