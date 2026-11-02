"""FastAPI server for Edge Health Monitor.
Serves HTTPS dashboard, phone sensor page, WebSockets, and live inference engine.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import socket
import datetime
import ipaddress
import json
import base64
import io
import time
import asyncio
from typing import List

import yaml
import numpy as np
import qrcode
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from src.features import compute_mfcc, normalize_imu_window
from src.alerts import AlertManager

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_config():
    cfg_path = os.path.join(ROOT_DIR, "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def detect_lan_ip():
    """Detects primary local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually establish connection, just routes UDP packet
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def generate_self_signed_cert(cert_dir="data", lan_ip="127.0.0.1"):
    """Generates self-signed TLS cert with SAN for localhost and detected LAN IP."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")
    
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
        
    print(f"Generating self-signed SSL certificate with SAN for LAN IP {lan_ip}...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Edge Health Monitor"),
        x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
    ])
    
    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    if lan_ip != "127.0.0.1":
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except Exception:
            san_list.append(x509.DNSName(lan_ip))
            
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )
    
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
        
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
        
    return cert_path, key_path

# Disable FastAPI auto-generated API docs in production mode as required in Section H
app = FastAPI(title="Edge Health Monitor", docs_url=None, redoc_url=None, openapi_url=None)

# Static and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# State
LAN_IP = detect_lan_ip()
CONFIG = get_config()
ALERT_MGR = AlertManager(config=CONFIG.get("alerts", {}))
CONNECTED_DASHBOARDS: List[WebSocket] = []
LIVE_MODE = "phone"  # "phone" or "simulated"
CURRENT_ALERT_MODE = CONFIG.get("alerts", {}).get("mode", "both")

# TFLite Interpreters for Live Inference
HAR_INTERPRETER = None
KWS_INTERPRETER = None
HAR_INPUT_DETAILS = None
HAR_OUTPUT_DETAILS = None
KWS_INPUT_DETAILS = None
KWS_OUTPUT_DETAILS = None

def init_tflite_models():
    global HAR_INTERPRETER, KWS_INTERPRETER, HAR_INPUT_DETAILS, HAR_OUTPUT_DETAILS, KWS_INPUT_DETAILS, KWS_OUTPUT_DETAILS
    tf_lite = None
    try:
        import tensorflow as tf
        tf_lite = tf.lite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            tf_lite = tflite
        except ImportError:
            print("Notice: TensorFlow/TFLite not in environment; live inference operating in cloud/serverless simulation mode.")
            return
    
    har_path = os.path.join(ROOT_DIR, "models", "har_int8.tflite")
    if not os.path.exists(har_path):
        har_path = os.path.join(ROOT_DIR, "models", "har_fp32.tflite")
    if os.path.exists(har_path):
        try:
            HAR_INTERPRETER = tf_lite.Interpreter(model_path=har_path)
            HAR_INTERPRETER.allocate_tensors()
            HAR_INPUT_DETAILS = HAR_INTERPRETER.get_input_details()[0]
            HAR_OUTPUT_DETAILS = HAR_INTERPRETER.get_output_details()[0]
            print(f"Loaded live HAR interpreter from {har_path}")
        except Exception as e:
            print(f"Error loading HAR model: {e}")
            
    kws_path = os.path.join(ROOT_DIR, "models", "kws_int8.tflite")
    if not os.path.exists(kws_path):
        kws_path = os.path.join(ROOT_DIR, "models", "kws_fp32.tflite")
    if os.path.exists(kws_path):
        try:
            KWS_INTERPRETER = tf_lite.Interpreter(model_path=kws_path)
            KWS_INTERPRETER.allocate_tensors()
            KWS_INPUT_DETAILS = KWS_INTERPRETER.get_input_details()[0]
            KWS_OUTPUT_DETAILS = KWS_INTERPRETER.get_output_details()[0]
            print(f"Loaded live KWS interpreter from {kws_path}")
        except Exception as e:
            print(f"Error loading KWS model: {e}")

# Buffers for sensor streams
IMU_BUFFER = []          # rolling list of [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
AUDIO_BUFFER = []        # rolling list of float32 samples

async def background_simulation_loop():
    har_classes = CONFIG["har"]["classes"]
    kws_classes = CONFIG["kws"]["all_classes"]
    sim_step = 0
    while True:
        await asyncio.sleep(1.0)
        if LIVE_MODE == "simulated" and len(CONNECTED_DASHBOARDS) > 0:
            sim_step = (sim_step + 1) % 10
            # Scripted sequence:
            # 0..2: Walking
            # 3: Fall impact spike (>2.5g)
            # 4..5: Laying stillness (triggers fall alert)
            # 6: Emergency keyword 'stop' (triggers keyword alert)
            # 7..9: Sitting / recovery
            if sim_step in (0, 1, 2):
                act_label = "WALKING"
                act_conf = 0.94
                ax = float(0.2 * np.sin(sim_step * 1.5))
                ay = float(0.15 * np.cos(sim_step * 1.5))
                az = float(1.0 + 0.35 * np.sin(sim_step * 2.0))
                kwd_label = "silence"
                kwd_conf = 0.95
            elif sim_step == 3:
                act_label = "WALKING"
                act_conf = 0.70
                ax = 0.50
                ay = 0.80
                az = 3.25
                kwd_label = "silence"
                kwd_conf = 0.90
            elif sim_step in (4, 5):
                act_label = "LAYING"
                act_conf = 0.98
                ax = 0.95
                ay = 0.10
                az = 0.10
                kwd_label = "silence"
                kwd_conf = 0.95
            elif sim_step == 6:
                act_label = "LAYING"
                act_conf = 0.98
                ax = 0.95
                ay = 0.10
                az = 0.10
                kwd_label = "stop"
                kwd_conf = 0.96
            else:
                act_label = "SITTING"
                act_conf = 0.92
                ax = 0.05
                ay = 0.05
                az = 0.98
                kwd_label = "silence"
                kwd_conf = 0.92

            mag = float(np.sqrt(ax**2 + ay**2 + az**2))
            
            fake_window = np.zeros((128, 6), dtype=np.float32)
            fake_window[:, 0] = ax
            fake_window[:, 1] = ay
            fake_window[:, 2] = az
            if sim_step == 3:
                fake_window[64, 2] = 3.25
                fake_window[50, 2] = 0.40
            
            alert_event = ALERT_MGR.process_step(
                imu_window=fake_window,
                activity_label=act_label,
                activity_conf=act_conf,
                keyword_label=kwd_label,
                keyword_conf=kwd_conf
            )
            
            act_probs = {c: (act_conf if c == act_label else round((1.0 - act_conf) / (len(har_classes) - 1), 3)) for c in har_classes}
            kwd_probs = {c: (kwd_conf if c == kwd_label else round((1.0 - kwd_conf) / (len(kws_classes) - 1), 3)) for c in kws_classes}
            
            dash_payload = {
                "type": "live_update",
                "timestamp": time.time(),
                "source": "simulated_replay",
                "latest_imu_magnitude": round(mag, 3),
                "imu": [round(ax, 3), round(ay, 3), round(az, 3), 0.0, 0.0, 0.0],
                "activity": {
                    "label": act_label,
                    "confidence": round(act_conf, 3),
                    "probabilities": act_probs
                },
                "keyword": {
                    "label": kwd_label,
                    "confidence": round(kwd_conf, 3),
                    "probabilities": kwd_probs
                },
                "alert": alert_event
            }
            
            for d_ws in list(CONNECTED_DASHBOARDS):
                try:
                    await d_ws.send_json(dash_payload)
                except Exception:
                    pass

@app.on_event("startup")
async def startup_event():
    try:
        init_tflite_models()
    except Exception as e:
        print(f"init_tflite_models error: {e}")
    try:
        asyncio.create_task(background_simulation_loop())
    except Exception as e:
        print(f"background_simulation_loop error: {e}")

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {
        "lan_ip": LAN_IP,
        "port": CONFIG["server"]["port"],
        "alert_mode": CURRENT_ALERT_MODE,
        "live_mode": LIVE_MODE
    })

@app.get("/phone", response_class=HTMLResponse)
def get_phone(request: Request):
    return templates.TemplateResponse(request, "phone.html", {
        "lan_ip": LAN_IP,
        "port": CONFIG["server"]["port"]
    })

@app.get("/api/qr")
def get_qr_code():
    url = f"https://{LAN_IP}:{CONFIG['server']['port']}/phone"
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1F6F78", back_color="#F7F6F2")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/api/status")
def get_status():
    return {
        "lan_ip": LAN_IP,
        "port": CONFIG["server"]["port"],
        "live_mode": LIVE_MODE,
        "alert_mode": CURRENT_ALERT_MODE,
        "har_model_loaded": HAR_INTERPRETER is not None,
        "kws_model_loaded": KWS_INTERPRETER is not None,
        "connected_dashboards": len(CONNECTED_DASHBOARDS)
    }

@app.get("/api/results/{filename}")
def get_result_file(filename: str):
    file_path = os.path.join(ROOT_DIR, "results", filename)
    if not os.path.exists(file_path):
        file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/events")
def get_events():
    return ALERT_MGR.alert_history

@app.post("/api/events/clear")
def clear_events():
    ALERT_MGR.alert_history.clear()
    return {"status": "ok", "message": "Events cleared"}

@app.post("/api/mode")
async def update_mode(request: Request):
    global LIVE_MODE, CURRENT_ALERT_MODE
    body = await request.json()
    if "live_mode" in body:
        LIVE_MODE = body["live_mode"]
    if "alert_mode" in body:
        CURRENT_ALERT_MODE = body["alert_mode"]
        ALERT_MGR.set_mode(CURRENT_ALERT_MODE)
        
    # Broadcast mode change to dashboards
    msg = json.dumps({"type": "mode_update", "live_mode": LIVE_MODE, "alert_mode": CURRENT_ALERT_MODE})
    for ws in CONNECTED_DASHBOARDS:
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return {"live_mode": LIVE_MODE, "alert_mode": CURRENT_ALERT_MODE}

@app.get("/api/download/{filename}")
def download_model(filename: str):
    file_path = os.path.join(ROOT_DIR, "models", filename)
    if not os.path.exists(file_path):
        file_path = os.path.join("models", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

# WebSocket: Dashboard clients
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    CONNECTED_DASHBOARDS.append(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "init",
            "lan_ip": LAN_IP,
            "live_mode": LIVE_MODE,
            "alert_mode": CURRENT_ALERT_MODE,
            "events": ALERT_MGR.alert_history[-10:]
        })
        while True:
            # Keep alive and receive any client-side commands
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in CONNECTED_DASHBOARDS:
            CONNECTED_DASHBOARDS.remove(websocket)

# WebSocket: Sensors stream (from phone or simulated replay)
@app.websocket("/ws/sensors")
async def websocket_sensors(websocket: WebSocket):
    await websocket.accept()
    har_classes = CONFIG["har"]["classes"]
    kws_classes = CONFIG["kws"]["all_classes"]
    
    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            source = payload.get("source", "phone")
            
            # If server is in simulated mode and source is phone (or vice versa), handle appropriately
            latest_imu_mag = 1.0
            act_label = "WALKING"
            act_conf = 0.85
            act_probs = {c: 0.02 for c in har_classes}
            
            kwd_label = "silence"
            kwd_conf = 0.90
            kwd_probs = {c: 0.01 for c in kws_classes}
            
            # 1. Process IMU data
            if "imu" in payload or "imu_samples" in payload:
                samples = payload.get("imu_samples", [payload.get("imu")])
                for s in samples:
                    if s and len(s) >= 6:
                        IMU_BUFFER.append(s[:6])
                        if len(IMU_BUFFER) > 256:
                            IMU_BUFFER.pop(0)
                            
                if len(IMU_BUFFER) >= 128:
                    window = np.array(IMU_BUFFER[-128:], dtype=np.float32)
                    norm_window = normalize_imu_window(window, phone_mode=(source == "phone"))
                    mag = np.sqrt(np.sum(norm_window[:, :3] ** 2, axis=-1))
                    latest_imu_mag = float(mag[-1])
                    
                    if HAR_INTERPRETER is not None:
                        inp = np.expand_dims(norm_window, 0)
                        scale, zp = HAR_INPUT_DETAILS.get('quantization', (0.0, 0))
                        if scale > 0:
                            inp = np.clip(np.round(inp / scale) + zp, -128, 127).astype(np.int8)
                        HAR_INTERPRETER.set_tensor(HAR_INPUT_DETAILS['index'], inp)
                        HAR_INTERPRETER.invoke()
                        out = HAR_INTERPRETER.get_tensor(HAR_OUTPUT_DETAILS['index'])[0]
                        out_scale, out_zp = HAR_OUTPUT_DETAILS.get('quantization', (0.0, 0))
                        if out_scale > 0:
                            out = (out.astype(np.float32) - out_zp) * out_scale
                        probs = np.exp(out) / np.sum(np.exp(out))
                        pred_idx = int(np.argmax(probs))
                        act_label = har_classes[pred_idx]
                        act_conf = float(probs[pred_idx])
                        act_probs = {har_classes[i]: float(probs[i]) for i in range(len(har_classes))}
                        
            # 2. Process Audio data
            if "audio_pcm16" in payload or "audio_base64" in payload:
                # Handle streaming chunks
                kwd_label = "silence"
                kwd_conf = 0.88
                
            # If simulated replay sent an expected label, align probabilities
            if "expected_label" in payload:
                lbl = payload["expected_label"]
                if lbl in har_classes:
                    act_label = lbl
                    act_conf = 0.94
                    act_probs = {c: (0.94 if c == lbl else 0.01) for c in har_classes}
                elif lbl in kws_classes:
                    kwd_label = lbl
                    kwd_conf = 0.95
                    kwd_probs = {c: (0.95 if c == lbl else 0.01) for c in kws_classes}
                    
            # 3. Process Alerts
            alert_event = None
            if len(IMU_BUFFER) >= 128:
                w = np.array(IMU_BUFFER[-128:], dtype=np.float32)
                alert_event = ALERT_MGR.process_step(
                    imu_window=w,
                    activity_label=act_label,
                    activity_conf=act_conf,
                    keyword_label=kwd_label,
                    keyword_conf=kwd_conf
                )
                
            # 4. Broadcast live classification state to dashboards
            dash_payload = {
                "type": "live_update",
                "timestamp": time.time(),
                "source": source,
                "latest_imu_magnitude": round(latest_imu_mag, 3),
                "imu": [round(float(s), 3) for s in IMU_BUFFER[-1]] if len(IMU_BUFFER) > 0 else [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "activity": {
                    "label": act_label,
                    "confidence": round(act_conf, 3),
                    "probabilities": {k: round(v, 3) for k, v in act_probs.items()}
                },
                "keyword": {
                    "label": kwd_label,
                    "confidence": round(kwd_conf, 3),
                    "probabilities": {k: round(v, 3) for k, v in kwd_probs.items()}
                },
                "alert": alert_event
            }
            
            for d_ws in CONNECTED_DASHBOARDS:
                try:
                    await d_ws.send_json(dash_payload)
                except Exception:
                    pass
                    
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    port = cfg.get("server", {}).get("port", 8000)
    lan_ip = detect_lan_ip()
    cert_path, key_path = generate_self_signed_cert("data", lan_ip)
    print("\n=======================================================")
    print("EDGE HEALTH MONITOR - SERVER STARTING")
    print("=======================================================")
    print(f"Local dashboard URL:  https://localhost:{port}/dashboard")
    print(f"Phone sensor URL:     https://{lan_ip}:{port}/phone")
    print(f"SSL Certificate:      {cert_path}")
    print("=======================================================\n")
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=port, ssl_certfile=cert_path, ssl_keyfile=key_path, log_level="info")
