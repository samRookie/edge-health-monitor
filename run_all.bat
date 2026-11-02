@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   EDGE HEALTH MONITOR: END-TO-END REPRODUCTION SCRIPT
echo ========================================================

set QUICK_FLAG=--quick
if "%1"=="--full" (
    set QUICK_FLAG=
    echo Running FULL training mode...
) else (
    echo Running QUICK training mode (pass --full for full training)...
)

:: Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo Error: .venv\Scripts\python.exe not found!
    echo Please create the virtual environment: uv venv .venv --python 3.11
    exit /b 1
)

set PYTHON=.venv\Scripts\python.exe
set PYTEST=.venv\Scripts\pytest.exe

echo.
echo [1/6] Training Baseline Models and Features...
%PYTHON% src/train.py %QUICK_FLAG%
if %ERRORLEVEL% neq 0 (
    echo Error: Phase 2 baseline training failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/6] Running Compression, Pruning, Distillation and Benchmarking...
%PYTHON% scripts/run_compression_and_benchmarks.py
if %ERRORLEVEL% neq 0 (
    echo Error: Compression and benchmarking failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/6] Running Federated Learning (FedAvg) Simulation...
%PYTHON% src/federated.py
if %ERRORLEVEL% neq 0 (
    echo Error: Federated learning simulation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [4/6] Running Adversarial Robustness and Noise Analysis...
%PYTHON% src/adversarial.py
if %ERRORLEVEL% neq 0 (
    echo Error: Adversarial evaluation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [5/6] Running Automated PyTest Unit and Integration Test Suite...
%PYTEST% tests/
if %ERRORLEVEL% neq 0 (
    echo Error: PyTest test suite encountered failures!
    exit /b %ERRORLEVEL%
)

echo.
echo [6/6] Running 30-Item UI Audit and Generating Report PDF...
%PYTHON% scripts/run_ui_audit.py
%PYTHON% scripts/generate_report_pdf.py
%PYTHON% scripts/verify_metrics.py

echo.
echo ========================================================
echo   ALL STAGES COMPLETED SUCCESSFULLY
echo ========================================================
echo Launch the interactive system:
echo   %PYTHON% src/server/app.py
echo.
echo Local dashboard: https://localhost:8000/dashboard
echo Mobile phone UI: https://^<LAN-IP^>:8000/phone
echo ========================================================
