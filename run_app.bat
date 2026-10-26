@echo off
rem ============================================================
rem  Option Pricing Tool - Web App launcher
rem
rem  This project runs on the conda env "Quant1.0" (Python 3.12),
rem  which is the ONLY env on this machine with py_vollib installed.
rem  Plain `python app.py` falls back to Anaconda base (3.13) and
rem  dies with: ModuleNotFoundError: No module named 'py_vollib'
rem ============================================================
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

call D:\Anaconda3\condabin\conda.bat activate Quant1.0
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "Quant1.0".
    echo         Fallback: D:\Anaconda3\condabin\conda.bat run --no-capture-output -n Quant1.0 python app.py
    pause
    exit /b 1
)

python -c "import py_vollib" 2>nul
if errorlevel 1 (
    echo [ERROR] py_vollib not importable in Quant1.0.
    echo         Fix with: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Starting on http://127.0.0.1:5000     ^(Ctrl+C to stop^)
echo ============================================================
echo.
python app.py
pause
