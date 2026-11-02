"""Unit tests for dual fall alert logic, keyword triggering, and debouncing cooldown."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pytest

from src.fall_rule import FallRuleDetector, create_synthetic_fall_sequence
from src.alerts import AlertManager

def test_fall_rule_detection_on_synthetic_sequence():
    detector = FallRuleDetector(spike_thresh_g=2.5, stillness_thresh=0.05, posture_thresh=0.70)
    fall_data = create_synthetic_fall_sequence(sample_rate=50, duration_s=5)
    
    # 1. Normal walking (0 to 128 samples) -> should not trigger
    walking_window = fall_data[:128]
    triggered, details = detector.evaluate(walking_window, predicted_activity="WALKING", activity_confidence=0.92)
    assert not triggered, "Normal walking should not trigger fall alert"
    
    # 2. Impact + Stillness window (samples 100 to 228) -> should trigger when posture is LAYING
    impact_window = fall_data[100:228]
    triggered, details = detector.evaluate(impact_window, predicted_activity="LAYING", activity_confidence=0.88)
    assert triggered, "Impact spike followed by stillness and LAYING posture must trigger fall alert"

def test_keyword_plus_activity_alert():
    mgr = AlertManager(config={
        "mode": "keyword_plus_activity",
        "emergency_keyword": "stop",
        "cooldown_seconds": 5.0,
        "keyword_confidence_threshold": 0.80
    })
    
    dummy_imu = np.zeros((128, 6), dtype=np.float32)
    dummy_imu[:, 2] = 1.0
    
    # Keyword 'stop' with confidence 0.95 and abnormal posture 'LAYING'
    alert = mgr.process_step(
        imu_window=dummy_imu,
        activity_label="LAYING",
        activity_conf=0.90,
        keyword_label="stop",
        keyword_conf=0.95,
        timestamp=100.0
    )
    assert alert is not None, "Emergency keyword 'stop' with LAYING activity must trigger alert"
    assert "Keyword plus activity" in alert["rule"]

def test_alert_debouncing_and_cooldown():
    mgr = AlertManager(config={
        "mode": "both",
        "emergency_keyword": "stop",
        "cooldown_seconds": 5.0,
        "keyword_confidence_threshold": 0.80
    })
    
    dummy_imu = np.zeros((128, 6), dtype=np.float32)
    dummy_imu[:, 2] = 1.0
    
    # First alert at t = 100.0
    alert1 = mgr.process_step(dummy_imu, "LAYING", 0.9, "stop", 0.95, timestamp=100.0)
    assert alert1 is not None, "Initial event must fire"
    
    # Second identical event at t = 102.0 (within 5.0s cooldown) -> should be suppressed
    alert2 = mgr.process_step(dummy_imu, "LAYING", 0.9, "stop", 0.95, timestamp=102.0)
    assert alert2 is None, "Repeated event within cooldown period must be suppressed by debouncing"
    
    # Third event at t = 106.0 (after cooldown expiration) -> should fire
    alert3 = mgr.process_step(dummy_imu, "LAYING", 0.9, "stop", 0.95, timestamp=106.0)
    assert alert3 is not None, "Event after cooldown expiration must fire"
