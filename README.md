# Edge Health Monitor

An end-to-end TinyML healthcare wearable embedded system for the course **Introduction to AI and Embedded Systems**.

The system provides real-time dual-modality patient monitoring:
1. **Human Activity Recognition (HAR):** 1D-CNN (<17k parameters) processing 6-axis IMU signals (50 Hz, 128 samples, 2.56-second windows) on the UCI HAR dataset with dual-threshold heuristic fall detection.
2. **Emergency Keyword Spotting (KWS):** Depthwise Separable CNN (DS-CNN, <14k parameters) processing 40-band audio MFCC spectrograms (16 kHz, 1.0-second clips) on Google Speech Commands v0.02, spotting emergency keyword "stop".

---

## 1. Quick Start

### Prerequisites
- Python 3.11 (installed via `uv` or system Python 3.11)
- Windows (PowerShell / cmd) or Linux / macOS

### Installation
```bash
# 1. Create virtual environment using Python 3.11
uv venv .venv --python 3.11

# 2. Install pinned dependencies
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 3. Install Playwright browser binary
.venv/Scripts/playwright.exe install chromium
```

---

## 2. Running the Interactive System

### Launch Command
```bash
.venv\Scripts\python.exe src/server/app.py
```
*(On Linux/macOS: `.venv/bin/python src/server/app.py`)*

The server automatically:
- Detects the primary local LAN IP address.
- Generates a self-signed TLS/SSL certificate with Subject Alternative Names (SAN) for `localhost` and the LAN IP in `data/cert.pem` and `data/key.pem`.
- Launches a secure HTTPS and WSS (WebSocket Secure) server on port 8000.

### Accessing the System
1. **Desktop Engineering Dashboard:**
   Open Chrome or Firefox to:
   ```
   https://localhost:8000/dashboard
   ```
   *(Accept the self-signed certificate warning: click "Advanced" -> "Proceed to localhost (unsafe)").*

2. **Mobile Phone Sensor Capture UI:**
   Navigate to the **Live classification** tab (item 10 on left navigation) to view the dynamically generated QR code and your LAN IP.
   Scan the QR code with your mobile device or navigate to:
   ```
   https://<YOUR-LAN-IP>:8000/phone
   ```

### 4-Step Phone Connection Procedure
1. Ensure both your computer and phone are connected to the same Wi-Fi network (or use your phone's personal hotspot).
2. Scan the QR code displayed on the Live classification dashboard page with your mobile phone camera.
3. Accept the self-signed SSL certificate on your phone: tap "Advanced" -> "Proceed to `<LAN-IP>` (unsafe)".
4. On the `/phone` page, tap **"Start sensors"** and allow microphone access. Accelerometer and orientation telemetry will stream live to the dashboard.

---

## 3. Operating Modes

### Input Source Switch
On the Live Classification page, use the **Input source** dropdown:
- **Phone live (WebSocket):** Ingests live real-time IMU motion and microphone audio from connected smartphones over encrypted WebSockets.
- **Simulated replay (offline):** Executes a deterministic pre-recorded physical emergency trajectory (normal walking -> sudden high-g impact spike -> stillness -> horizontal laying posture -> keyword audio clip) for presentation and verification without a physical phone.

### Alert Mode Switch
Choose how emergency events are evaluated:
- **Both (fall rule or keyword):** Triggers an emergency alert on either physical fall detection or emergency keyword "stop".
- **Rule-based fall only:** Evaluates kinetic impact spike (>2.5g), post-impact stillness variance (<0.05), and posture confirmation (`LAYING` confidence >0.70).
- **Keyword plus activity only:** Triggers emergency activation when the keyword "stop" is uttered while the patient is immobilized in an abnormal posture (`LAYING`).

All alerts feature stateful debouncing with an immutable **5.0-second cooldown timer** to suppress event storms.

---

## 4. End-to-End Reproduction

To run the complete pipeline from scratch:
```bash
# Windows
run_all.bat

# Linux / macOS
chmod +x run_all.sh
./run_all.sh
```

This sequentially executes:
1. `src/train.py --quick`: Baseline HAR 1D-CNN and KWS DS-CNN training, classical ML comparisons, and overfit/underfit experiments.
2. `scripts/run_compression_and_benchmarks.py`: Magnitude pruning (50%, 80%), FP16, full INT8 integer PTQ, knowledge distillation, C-array header export (`.h`), and Cortex-M4 feasibility analysis.
3. `src/federated.py`: FedAvg distributed simulation on UCI HAR across IID and Non-IID partitions with bandwidth accounting.
4. `src/adversarial.py`: Fast Gradient Sign Method (FGSM) attacks on MFCCs, environmental acoustic noise SNR analysis, and noise augmentation defense.
5. `pytest tests/`: 9 unit and integration tests covering alert logic, debouncing, data shapes, and live WebSocket streaming.
6. `scripts/run_ui_audit.py`: 30-item UI and design token audit using headless Playwright with automated screenshots of all 14 pages.
7. `scripts/generate_report_pdf.py` & `scripts/verify_metrics.py`: Compiles `docs/report.pdf` and verifies metric honesty against `results/*.json`.

---

## 5. Headline Results

*Every metric cited below is read directly from generated files in `results/`.*

| Model & Task | Variant | Accuracy | Flash Size | Tensor Arena RAM [estimate] | MACs | Cortex-M4 Est. Latency [estimate] |
|---|---|---|---|---|---|---|
| **HAR 1D-CNN** | FP32 Baseline | 90.02% | 69.53 KB | 172.75 KB | 782,720 | 48.92 ms |
| (6 activities) | Pruned (50%) | 74.86% | 69.53 KB | 172.75 KB | 782,720 | 48.92 ms |
| | Pruned (80%) | 45.78% | 69.53 KB | 172.75 KB | 782,720 | 48.92 ms |
| | FP16 | 90.02% | 39.53 KB | 191.55 KB | 782,720 | 48.92 ms |
| | **INT8 PTQ** | **89.55%** | **26.17 KB** | **48.31 KB** | **782,720** | **15.29 ms** |
| **KWS DS-CNN** | FP32 Baseline | 69.50% | 57.35 KB | 1,936.64 KB | 4,544,128 | 284.01 ms |
| (8 classes) | FP16 | 69.50% | 35.10 KB | 1,951.57 KB | 4,544,128 | 284.01 ms |
| | **INT8 PTQ** | **68.33%** | **28.45 KB** | **504.77 KB** | **4,544,128** | **88.75 ms** |
| | Distilled Student | 12.50% | 13.94 KB | 233.94 KB | 2,070,016 | 40.43 ms |

*Cortex-M4 latency estimates assume 64 MHz clock, 1.25 cycles/MAC for INT8 CMSIS-NN, and 4.0 cycles/MAC for FP32. Not measured on physical hardware.*

- **Federated Learning (FedAvg):** Centralized baseline 90.02%, IID FedAvg 81.03%, Non-IID by subject 85.27% (4.75% accuracy gap). INT8 quantization reduces communication overhead from 760.78 KB/round to 190.20 KB/round (75% savings).
- **Adversarial Robustness:** Clean baseline accuracy (69.50%) drops to 17.67% at FGSM $\epsilon=0.05$ and 6.17% at $\epsilon=0.20$. Acoustic noise drops accuracy to 18.50% at 5 dB SNR.

---

## 6. Project Artifacts and Deliverables

- **Academic Report:** [`docs/report.pdf`](docs/report.pdf) (compiled via ReportLab) and [`docs/report.md`](docs/report.md).
- **Architecture Diagram:** [`docs/architecture.png`](docs/architecture.png).
- **Simulation and Limitation Boundary:** [`docs/WHAT_WAS_NOT_VERIFIED.md`](docs/WHAT_WAS_NOT_VERIFIED.md).
- **UI Design Token Audit:** [`results/ui_audit.md`](results/ui_audit.md) (All 30 items verified PASS).
- **Automated UI Screenshots:** [`results/ui_screenshots/`](results/ui_screenshots/) (all 14 desktop views and mobile phone view).
- **Exported C Headers:** [`models/har_int8.h`](models/har_int8.h) and [`models/kws_int8.h`](models/kws_int8.h).
