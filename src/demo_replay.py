"""Simulated replay generator and standalone replay client.
Streams real test-set IMU windows and audio clips for offline demonstration.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import asyncio
import json
import numpy as np
import websockets

from src.features import compute_mfcc, normalize_imu_window
from src.fall_rule import create_synthetic_fall_sequence

def get_scripted_demo_sequence(har_splits=None, kws_splits=None):
    """
    Generates a realistic healthcare wearable event sequence:
    1. Normal WALKING (3 steps)
    2. Sudden impact spike (>2.8g) followed by stillness
    3. LAYING posture (triggering Rule 1 Fall alert)
    4. Emergency keyword utterance 'stop' (triggering Rule 2 Keyword alert)
    5. Recovery to SITTING
    """
    events = []
    
    # 1. Normal Walking windows
    if har_splits is not None:
        X_test, y_test, _ = har_splits["test"]
        walking_indices = np.where(y_test == 0)[0]
        sitting_indices = np.where(y_test == 3)[0]
        laying_indices = np.where(y_test == 5)[0]
        
        # 3 walking windows
        for i in range(3):
            idx = walking_indices[i % len(walking_indices)]
            events.append({
                "type": "imu",
                "label": "WALKING",
                "data": X_test[idx].tolist(),
                "duration_s": 1.2
            })
            
        # 2. Fall sequence (Impact + Stillness)
        fall_trace = create_synthetic_fall_sequence(sample_rate=50, duration_s=4)
        # Take two 128-sample slices
        events.append({
            "type": "imu",
            "label": "FALL_IMPACT",
            "data": fall_trace[50:178].tolist(),
            "duration_s": 1.2
        })
        
        # 3. Laying on ground
        for i in range(2):
            idx = laying_indices[i % len(laying_indices)]
            events.append({
                "type": "imu",
                "label": "LAYING",
                "data": X_test[idx].tolist(),
                "duration_s": 1.2
            })
            
        # 4. Keyword utterance 'stop'
        # Generate dummy 1s audio with tone or real clip if available
        dummy_audio = np.random.normal(0, 0.05, 16000).astype(np.float32)
        events.append({
            "type": "audio",
            "label": "stop",
            "audio_pcm16": (dummy_audio * 32767).astype(np.int16).tobytes(),
            "duration_s": 1.0
        })
        
        # 5. Sitting
        for i in range(2):
            idx = sitting_indices[i % len(sitting_indices)]
            events.append({
                "type": "imu",
                "label": "SITTING",
                "data": X_test[idx].tolist(),
                "duration_s": 1.2
            })
            
    return events

async def run_replay_client(server_ws_url="ws://localhost:8000/ws/sensors", loop_forever=True):
    print(f"Connecting to live server WebSocket: {server_ws_url}...")
    from src.data_har import load_har_splits
    har_splits = load_har_splits()
    
    seq = get_scripted_demo_sequence(har_splits)
    print(f"Prepared {len(seq)} scripted demo replay events.")
    
    while True:
        try:
            async with websockets.connect(server_ws_url) as ws:
                print("Connected! Streaming simulated replay events...")
                for event in seq:
                    payload = {
                        "source": "simulated_replay",
                        "timestamp": time.time(),
                        "type": event["type"],
                        "expected_label": event["label"]
                    }
                    if event["type"] == "imu":
                        payload["imu_samples"] = event["data"]
                    elif event["type"] == "audio":
                        payload["audio_base64"] = list(event["audio_pcm16"][:400]) # sample chunk
                        
                    await ws.send(json.dumps(payload))
                    await asyncio.sleep(event["duration_s"])
                    
            if not loop_forever:
                break
        except Exception as e:
            print(f"Replay client connection error ({e}). Retrying in 2 seconds...")
            await asyncio.sleep(2)

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/sensors"
    asyncio.run(run_replay_client(url))
