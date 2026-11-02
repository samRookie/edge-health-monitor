"""Integration test for the live WebSocket sensor and dashboard pipeline."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pytest
from starlette.testclient import TestClient
from src.server.app import app

def test_api_status():
    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "lan_ip" in data
    assert "live_mode" in data
    assert "alert_mode" in data

def test_qr_code_endpoint():
    client = TestClient(app)
    resp = client.get("/api/qr")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

def test_live_websocket_pipeline():
    client = TestClient(app)
    
    # Connect dashboard WebSocket first
    with client.websocket_connect("/ws/dashboard") as dash_ws:
        init_msg = dash_ws.receive_json()
        assert init_msg["type"] == "init"
        
        # Connect sensor WebSocket and send simulated IMU packets
        with client.websocket_connect("/ws/sensors") as sensor_ws:
            # Send 128 samples of walking
            dummy_window = [[0.1, 0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(128)]
            payload = {
                "source": "simulated_test",
                "type": "imu",
                "imu_samples": dummy_window,
                "expected_label": "WALKING"
            }
            sensor_ws.send_json(payload)
            
            # Dashboard should receive live classification update
            update_msg = dash_ws.receive_json()
            assert update_msg["type"] == "live_update"
            assert "activity" in update_msg
            assert "keyword" in update_msg
            assert update_msg["activity"]["label"] == "WALKING"
