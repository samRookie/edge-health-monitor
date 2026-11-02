# EDGE HEALTH MONITOR — FULL EXECUTION PROMPT (single instruction, run end to end)

You are a senior embedded-AI / TinyML engineer working autonomously. Execute this entire document from planning to final build without asking me questions. Make reasonable decisions, log them, and keep going. I will not be available until you finish.

---

## 0. CONTEXT AND GOAL

I am an undergraduate (ECE, Embedded & IoT) submitting a project for the course "Introduction to AI and Embedded Systems". The professor's rule: **AI must be present in the project.** The course covers:

- Unit 1: AI and ML basics, embedded constraints (memory, power, compute), edge vs cloud AI, AI lifecycle for embedded apps, accelerators (MCU, NPU, FPGA), frameworks (TensorFlow Lite, Edge Impulse, CMSIS-NN).
- Unit 2: supervised/unsupervised learning, neural networks, training/evaluation, overfitting/underfitting, TinyML, quantization (INT8, FP16), pruning, knowledge distillation.
- Unit 3: TFLite architecture and interpreter, conversion to TFLite, deploying on Cortex-M, Edge Impulse workflow, CMSIS-NN, benchmarking inference.
- Unit 4: keyword spotting, CNN image/audio models, anomaly detection, gesture/IMU recognition, real-time constraints and optimization.
- Unit 5: healthcare wearables with AI inference, federated learning on edge devices, AI security / adversarial attacks, future trends (neuromorphic, in-sensor computing).

**Project:** "Edge Health Monitor" — a simulated wearable that
1. classifies human activity from IMU data (1D-CNN) and raises a fall alert, and
2. spots an emergency keyword from audio (DS-CNN on MFCC),
both compressed as TinyML models (pruning, FP16, INT8, distillation), with federated learning and adversarial-robustness studies, a live dashboard, and a report mapped to all five units.

**Hard constraints**
- Software only. **No physical MCU, no gem5, no QEMU.**
- CPU only (assume no GPU). Everything must train in minutes, not hours.
- I have only a few hours total. Prioritize in the order in section 12 if time runs short.
- The phone (Android, Chrome) must be usable as the live sensor source (accelerometer, gyroscope, microphone). A simulated-replay mode must always work as a fallback.

---

## 1. NON-NEGOTIABLE HONESTY RULES

1. Every number in the report, dashboard, and README must be read from a saved results file (JSON/CSV) produced by your code. Never invent or hand-type metrics.
2. Label figures precisely:
   - Desktop inference time = "relative, desktop CPU, not MCU latency".
   - Flash = .tflite file size. RAM = tensor arena estimate (state how you computed it). MACs = computed by your code (state the method).
   - Any Cortex-M latency figure = "analytic estimate" with the formula and assumptions, or omit it.
3. Do not claim CMSIS-NN speedups, Edge Impulse results, or on-device results you did not produce. CMSIS-NN and Edge Impulse are covered by a written section plus an optional appendix of manual steps.
4. Live phone accuracy will be lower than test-set accuracy (different sensor placement, noise, sampling). Say so in the report. Benchmark numbers come only from held-out test sets.
5. If a step fails twice, log the failure to `logs/failures.md`, apply the documented fallback, and continue. Never stall.
6. Finish with `docs/WHAT_WAS_NOT_VERIFIED.md` listing everything unverified or approximate.

---

## 2. ENVIRONMENT AND REPO LAYOUT

- Python 3.10 or 3.11, virtualenv at `.venv`. Pin versions in `requirements.txt` (TensorFlow 2.x with Keras, `tensorflow-model-optimization` for pruning, numpy, scipy, pandas, scikit-learn, librosa or tf.signal for MFCC, matplotlib, fastapi, uvicorn[standard], websockets, jinja2, qrcode[pil], cryptography, reportlab or a Markdown-to-PDF route, pytest).
- Check Python and TensorFlow compatibility first. If `tensorflow-model-optimization` fails to install, use the fallback in section 6.
- Fix random seeds everywhere (numpy, tf, python).
- Layout:

```
edge-health-monitor/
  README.md
  requirements.txt
  run_all.sh            # one command reruns everything (also run_all.bat for Windows)
  config.yaml           # classes, thresholds, paths, alert mode, caps
  data/                 # downloaded datasets (git-ignored)
  src/
    data_har.py  data_kws.py  features.py
    models.py  train.py  evaluate.py
    compress.py  convert.py  benchmark.py
    federated.py  adversarial.py
    alerts.py  fall_rule.py
    server/ (app.py, static/, templates/)   # dashboard + phone page
    demo_replay.py
  results/              # JSON/CSV/PNG outputs (source of truth for the report)
  models/               # .keras and .tflite files
  docs/                 # report.md, report.pdf, architecture.png, WHAT_WAS_NOT_VERIFIED.md
  logs/
  tests/
```

---

## 3. PHASE 1 — DATA (download automatically, no manual steps)

### 3.1 IMU: UCI HAR Dataset
- Download "UCI HAR Dataset" zip from the UCI repository (official source). If that URL fails, search for an official mirror.
- Use the raw **Inertial Signals**: `total_acc_x/y/z` and `body_gyro_x/y/z` (6 channels, 128 samples per window, 50 Hz, windows of 2.56 s). Use the dataset's own subject-wise train/test split.
- Classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
- Keep the subject id for each window (needed for federated clients). Hold out a validation set by subject from the training subjects.
- Units: UCI total_acc is in g, gyro in rad/s. Phone `accelerationIncludingGravity` is m/s², so divide by 9.81 in preprocessing. Document the phone axis-mapping assumption.

### 3.2 Audio: Google Speech Commands v0.02
- Download `speech_commands_v0.02.tar.gz` from `download.tensorflow.org/data/` (about 2 GB) and extract. The official validation_list.txt and testing_list.txt define the splits.
- **Important:** the dataset has no word "help". Use these 6 target words: `yes, no, stop, go, up, down`, plus `unknown` (sampled from the other words) and `silence` (from `_background_noise_`). Total 8 classes.
- **The emergency keyword is configurable (`config.yaml: emergency_keyword`), default `stop`.** Document this clearly in the report as a dataset limitation.
- Cap samples per class (about 2500 train, proportional val/test) and cap `unknown` so classes are balanced. Add noise mixing and time-shift augmentation on the training set only.
- 1-second clips at 16 kHz, MFCC features (about 40 coefficients, 25 ms window, 10 ms hop, giving around 98x40 input). Document the exact parameters.

---

## 4. PHASE 2 — BASELINE MODELS AND TRAINING

- **HAR model:** small 1D-CNN (for example Conv1D 32 -> Conv1D 64 -> GlobalAvgPool -> Dense). Target under about 50k parameters.
- **KWS model:** DS-CNN (depthwise-separable convolution stack), target under about 100k parameters.
- Also train, for the Unit 2 supervised-vs-classical discussion: a Random Forest or SVM on simple features for HAR, and **one unsupervised method** (k-means or an autoencoder on HAR features) with a short comparison.
- For each model produce: training and validation loss/accuracy curves (PNG), test accuracy, per-class precision/recall/F1, confusion matrix (PNG), parameter count.
- **Overfitting / underfitting demonstration:** train a deliberately tiny (underfit) and an over-large/no-regularization (overfit) variant of the HAR model, plot curves, and explain. Keep this cheap (few epochs).
- Use early stopping and a CPU-friendly batch size. Save all metrics to `results/*.json`.
- Acceptance targets (report whatever you honestly get): HAR test accuracy about 90% or higher, KWS test accuracy about 85% or higher. If lower, tune briefly and report the actual values.

---

## 5. PHASE 3 — COMPRESSION (Unit 2)

For both models, produce and evaluate:
1. **FP32 baseline.**
2. **Pruned** (magnitude pruning, target 50% and 80% sparsity if time allows, with fine-tuning), then strip pruning wrappers. Note that sparsity only shrinks the file after compression (zip); report both raw and gzip-compressed sizes.
3. **FP16** TFLite.
4. **INT8** full-integer TFLite with a representative dataset (a few hundred training samples), int8 input and output. Report accuracy on the full test set via the TFLite interpreter (not the Keras model).
5. **Knowledge distillation (KWS only):** train a larger teacher DS-CNN, then a small student with temperature-scaled soft targets plus hard-label loss. Compare student-with-distillation vs the same student trained without it.

Fallback if pruning tooling fails: implement simple magnitude-weight masking manually, and note it in the report.

---

## 6. PHASE 4 — TFLITE CONVERSION AND BENCHMARK (Unit 3)

- Convert every variant to `.tflite` and validate with the TFLite interpreter: run the whole test set and confirm accuracy.
- Benchmark table (`results/benchmark.csv` and a PNG/markdown table) with columns: model, variant, test accuracy, size (KB), gzip size (KB), tensor arena estimate (KB, state the method, for example from interpreter tensor details), MACs, desktop median inference time (ms over at least 200 runs, labelled relative), parameter count.
- Success criteria to check and report as pass/fail: INT8 at least 3x smaller than FP32, and INT8 accuracy drop at most 3 percentage points. If missed, report the actual result without adjusting the criteria.
- Include a feasibility check: does each variant fit a stated MCU budget (for example Cortex-M4 with 256 KB RAM and 1 MB flash)? Make this a computed table, labelled as an estimate.
- Write a short explanation of the TFLite Micro architecture (interpreter, op resolver, tensor arena, flatbuffer) and how the `.tflite` would be converted to a C array (`xxd -i`) for an MCU. Generate the C array file for the INT8 models as a demonstration (do not claim it was run on hardware).

---

## 7. PHASE 5 — FEDERATED LEARNING (Unit 5)

- Simulate federated learning on the HAR model with plain TensorFlow (FedAvg implemented manually, no extra framework needed). Treat each UCI HAR training subject (or groups of subjects) as a client, with 5 to 10 clients, local epochs 1 to 3, about 20 rounds.
- Compare: centralized training vs FedAvg (IID-ish groups) vs FedAvg (non-IID by subject). Plot test accuracy per round and the final comparison. Report the gap honestly.
- Note communication cost: parameters transferred per round (bytes) for FP32 vs INT8-quantized updates (simple quantization of the weight deltas is an optional extra).
- State clearly that this is a single-machine simulation.

---

## 8. PHASE 6 — ADVERSARIAL ROBUSTNESS (Unit 5)

- On the KWS model: apply FGSM on the MFCC features (or the waveform if simpler) at several epsilon values. Plot accuracy vs epsilon for clean vs attacked.
- Defense: adversarial training or noise augmentation (pick one, implement, and measure). Report clean accuracy, attacked accuracy, and defended accuracy in one chart and table.
- Also test robustness to plain background noise at several SNR levels, since the phone microphone will be noisy.
- Keep the scope honest: white-box FGSM against the float model is only a proxy for real attacks on a deployed INT8 model. State that.

---

## 9. PHASE 7 — ALERT LOGIC (BOTH FALL OPTIONS)

Implement `src/alerts.py` with a configurable `alert_mode` in `config.yaml` and a switch on the dashboard:

- **Option 1 — "fall_rule":** a fall is flagged when a sharp acceleration-magnitude spike (threshold configurable, for example above 2.5 g) is followed within a few seconds by stillness (low variance) and the HAR model predicts LAYING with confidence above a threshold. Label it in the UI and report as "rule-assisted fall alert; ML provides the posture confirmation". Do NOT claim the CNN detects falls, because UCI HAR has no fall class. Test the rule on synthetic spike-then-still sequences and report that these are synthetic tests, not real fall data.
- **Option 2 — "keyword_plus_activity":** an alert fires when the emergency keyword is detected (confidence above threshold) AND the activity state is unusual (for example LAYING, or no movement for more than N seconds).
- **Both (default):** an alert fires if either condition is satisfied. Display which rule triggered it.
- Add debouncing and a cooldown so alerts do not spam. Unit-test the alert logic with pytest.

---

## 10. PHASE 8 — LIVE SYSTEM: SERVER, DASHBOARD, PHONE (ANDROID)

Build one FastAPI app (`src/server/app.py`) that serves both a laptop dashboard and a phone sensor page, over HTTPS, with WebSockets.

### 10.1 Server
- Generate a self-signed certificate at startup (cryptography library) including the laptop's LAN IP and `localhost` in the SAN. Run uvicorn with the cert.
- Detect the laptop's LAN IP automatically. Print the URL to the terminal.
- Endpoints: `/dashboard` (laptop), `/phone` (phone), `/ws/sensors` (phone sends data), `/ws/dashboard` (server pushes predictions), `/api/results` (serves saved results JSON), `/api/mode` (switch input mode and alert mode).
- Load the INT8 TFLite models with the TFLite interpreter for live inference. Windowing: buffer 128 IMU samples at 50 Hz with 50% overlap; buffer 1 s audio with a 0.25 s stride.

### 10.2 Phone page (`/phone`)
- Big "Start sensors" button. On tap: request microphone via `getUserMedia`, and start `DeviceMotionEvent` (call `requestPermission` if the function exists, to also cover iOS).
- Send accelerometer (`accelerationIncludingGravity`), gyroscope (`rotationRate`, convert deg/s to rad/s), and timestamps over the WebSocket. Resample client-side or server-side to a uniform 50 Hz.
- Capture audio with an AudioWorklet (or ScriptProcessor fallback), downsample from 44.1/48 kHz to 16 kHz in the browser or server, and send PCM16 chunks.
- Show connection status, sensor sample rate, and a live level meter. Request a Screen Wake Lock so the phone stays awake.

### 10.3 Dashboard (`/dashboard`)
- **QR code** at the top encoding `https://<LAN-IP>:<port>/phone`. Show the URL as text as well, with a short "how to connect" panel:
  1. Same Wi-Fi (or the phone's hotspot).
  2. Scan the QR with the Android camera, open in Chrome.
  3. Tap Advanced -> Proceed (self-signed certificate, one time).
  4. Tap Start sensors and allow the microphone.
- Mode switch: **Phone live / Simulated replay**, and an alert-mode switch (Option 1 / Option 2 / Both).
- Live panels: current activity and confidence, live accelerometer magnitude trace, keyword prediction and confidence, alert indicator and alert history with the triggering rule, connection status.
- Tabs for saved results: benchmark table, training curves, confusion matrices, federated chart, adversarial chart, architecture diagram.
- Use Chart.js (or similar) vendored locally or from a CDN; if the CDN is unreachable, bundle it locally.

### 10.4 Phone-connection fallbacks (implement and document in the README)
1. **Chrome flag:** `chrome://flags/#unsafe-treat-insecure-origin-as-secure`, add the plain-http origin of the laptop, enable, relaunch. Provide a `--http` server option that serves plain HTTP for this case.
2. **USB reverse:** `adb reverse tcp:<port> tcp:<port>`, then open `https://localhost:<port>/phone` (or `http://localhost:<port>/phone`) on the phone, which browsers treat as a secure context. Print these commands at startup.
3. **Hotspot note** in case the college Wi-Fi blocks client-to-client traffic.
4. **Simulated replay mode** must always work with no phone: `src/demo_replay.py` streams real test-set IMU windows and audio clips (including a scripted "walking -> stillness -> LAYING" sequence and a keyword clip) through the same pipeline and dashboard, so the whole demo works offline.

### 10.5 Verification you can do without a phone
- Write a **fake phone client** script (Python websocket client) that streams recorded test windows to `/ws/sensors`, and use it in an automated test to prove the live pipeline end to end.
- Test the page's JavaScript for syntax errors. State clearly in `WHAT_WAS_NOT_VERIFIED.md` that real-phone behaviour was not tested by you.

---

## 11. PHASE 9 — REPORT AND ARCHITECTURE (Units 1 to 5)

Create `docs/architecture.png` (generated with matplotlib or graphviz) showing: phone/simulator -> preprocessing (windowing, MFCC) -> HAR 1D-CNN and KWS DS-CNN (INT8 TFLite) -> alert logic -> dashboard, plus the offline pipeline (train -> compress -> convert -> benchmark) and the extras (FedAvg, FGSM, distillation).

Write `docs/report.md` and convert it to `docs/report.pdf` (Pandoc if available, otherwise a Python route). Structure:

1. Abstract and problem statement
2. Introduction: AI/ML basics, embedded constraints, edge vs cloud (include a table comparing latency, privacy, bandwidth, power, accuracy, cost), the AI lifecycle for embedded applications
3. System architecture (the diagram) and requirements, with measurable success criteria and pass/fail results
4. Hardware and accelerators survey: MCU vs NPU vs FPGA (analytic comparison table), and a framework survey (TFLite/TFLM, Edge Impulse, CMSIS-NN), with the honest scope note
5. Data and preprocessing (both datasets, the "help" keyword limitation, the HAR no-fall-class limitation)
6. Models and training results (curves, confusion matrices, overfit/underfit demo, classical and unsupervised comparison)
7. Compression results (pruning, FP16, INT8, distillation) with the benchmark table and feasibility table
8. TFLite deployment notes (conversion, interpreter, C array, arena)
9. Applications: keyword spotting, IMU activity recognition, the alert logic, real-time constraints and optimizations
10. Case study: healthcare wearable, federated learning results
11. Security: adversarial results and defense; also privacy benefits of edge inference
12. Future trends: neuromorphic computing, in-sensor computing (written section, short)
13. Live phone integration and how to run it; troubleshooting
14. Limitations (a complete honest list) and future work
15. Appendix A: Edge Impulse manual steps (upload the same HAR and KWS data, design the impulse, train, use the EON profiler for Cortex-M estimates, and what to screenshot). Mark them as not executed.
16. Appendix B: syllabus-to-project mapping table (unit topic -> section/file)
17. References (the textbooks listed in the course plus dataset and tool citations; do not fabricate citations)

All figures and tables must be embedded from `results/`.

---

## 12. PRIORITY ORDER IF TIME IS SHORT

1. Data + both baseline models + test accuracy
2. TFLite conversion (FP32, FP16, INT8) + benchmark table
3. Alert logic + simulated-replay dashboard
4. Phone live mode + QR code
5. Report + architecture diagram
6. Pruning and distillation
7. Federated learning
8. Adversarial attack and defense
9. Overfit/underfit demo, classical and unsupervised comparisons

Implement in this order. If a lower-priority item is cut, record it in `WHAT_WAS_NOT_VERIFIED.md` and mark the matching report section as "not completed" rather than leaving it out silently.

---

## 13. FINAL ACCEPTANCE CHECKLIST (run it yourself at the end)

- [ ] `run_all.sh` reproduces results from a clean clone (training can use a `--quick` flag for fewer epochs, but document which numbers came from which run).
- [ ] pytest passes (alert logic, preprocessing shapes, TFLite output sanity, fake phone client test).
- [ ] Every number in the report matches a file in `results/` (write a script that cross-checks key metrics).
- [ ] Dashboard starts, shows the QR code, and simulated replay works end to end with predictions and alerts.
- [ ] Phone mode runs with the fake client; real phone behaviour is marked unverified.
- [ ] `docs/report.pdf`, `docs/architecture.png`, all `.tflite` files, and README with exact run commands exist.
- [ ] `docs/WHAT_WAS_NOT_VERIFIED.md` is complete.

## 14. FINAL RESPONSE FORMAT

When finished, reply with: (1) the exact commands to launch the dashboard, (2) where the QR code and phone steps are, (3) a table of headline results read from `results/`, (4) the list of cut or unverified items, and (5) the three most important things I should check manually before presenting. Keep it brief.

BEGIN NOW. Start with Phase 0: write a short plan to `docs/PLAN.md`, then execute the phases in order.
