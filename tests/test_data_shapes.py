"""Tests for data shapes, windowing, and normalization."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pytest

from src.features import compute_mfcc, normalize_imu_window
from src.data_har import load_har_splits
from src.data_kws import load_kws_splits

def test_mfcc_shape_and_norm():
    dummy_audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    feat = compute_mfcc(dummy_audio)
    assert feat.shape == (98, 40), f"Expected MFCC shape (98, 40), got {feat.shape}"
    assert np.abs(np.mean(feat)) < 0.1, "MFCC should be approximately zero-mean normalized"
    assert np.abs(np.std(feat) - 1.0) < 0.2, "MFCC should have unit variance"

def test_imu_normalization():
    # Accelerometer in m/s^2 (e.g. 9.81 m/s^2 on Z)
    raw_phone_imu = np.zeros((128, 6), dtype=np.float32)
    raw_phone_imu[:, 2] = 9.80665
    
    norm = normalize_imu_window(raw_phone_imu, phone_mode=True)
    assert np.isclose(norm[0, 2], 1.0, atol=1e-3), "Phone mode should normalize 9.81 m/s^2 to ~1.0 g"

def test_har_splits_shapes():
    splits = load_har_splits()
    X_train, y_train, sub_train = splits["train"]
    X_val, y_val, sub_val = splits["val"]
    X_test, y_test, sub_test = splits["test"]
    
    assert X_train.shape[1:] == (128, 6), f"Unexpected HAR train shape {X_train.shape}"
    assert X_val.shape[1:] == (128, 6), f"Unexpected HAR val shape {X_val.shape}"
    assert X_test.shape[1:] == (128, 6), f"Unexpected HAR test shape {X_test.shape}"
    
    assert len(splits["classes"]) == 6, "Expected 6 HAR classes"
    assert set(np.unique(y_test)) == set(range(6)), "All 6 classes must be present in test set"
