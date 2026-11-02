#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  EDGE HEALTH MONITOR: END-TO-END REPRODUCTION SCRIPT"
echo "========================================================"

QUICK_FLAG="--quick"
if [ "$1" == "--full" ]; then
    QUICK_FLAG=""
    echo "Running FULL training mode..."
else
    echo "Running QUICK training mode (pass --full for full training)..."
fi

# Detect python executable
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
    PYTEST=".venv/Scripts/pytest.exe"
else
    PYTHON="python3"
    PYTEST="pytest"
fi

echo ""
echo "[1/6] Training Baseline Models and Features..."
$PYTHON src/train.py $QUICK_FLAG

echo ""
echo "[2/6] Running Compression, Pruning, Distillation and Benchmarking..."
$PYTHON scripts/run_compression_and_benchmarks.py

echo ""
echo "[3/6] Running Federated Learning (FedAvg) Simulation..."
$PYTHON src/federated.py

echo ""
echo "[4/6] Running Adversarial Robustness and Noise Analysis..."
$PYTHON src/adversarial.py

echo ""
echo "[5/6] Running Automated PyTest Unit and Integration Test Suite..."
$PYTEST tests/

echo ""
echo "[6/6] Running 30-Item UI Audit and Generating Report PDF..."
$PYTHON scripts/run_ui_audit.py
$PYTHON scripts/generate_report_pdf.py
$PYTHON scripts/verify_metrics.py

echo ""
echo "========================================================"
echo "  ALL STAGES COMPLETED SUCCESSFULLY"
echo "========================================================"
echo "Launch the interactive system:"
echo "  $PYTHON src/server/app.py"
echo ""
echo "Local dashboard: https://localhost:8000/dashboard"
echo "Mobile phone UI: https://<LAN-IP>:8000/phone"
echo "========================================================"
