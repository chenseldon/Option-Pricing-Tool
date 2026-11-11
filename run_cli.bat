@echo off
rem ============================================================
rem  Option Pricing Tool - CLI mode launcher (main.py)
rem
rem  Same env pinning as run_app.bat. The extra PYTHONIOENCODING
rem  matters here: main.py prints CJK text and box-drawing chars,
rem  which crash with UnicodeEncodeError under the GBK codepage
rem  whenever stdout is redirected to a file or pipe.
rem ============================================================
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

call D:\Anaconda3\condabin\conda.bat activate Quant1.0
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env "Quant1.0".
    pause
    exit /b 1
)

python main.py
pause
