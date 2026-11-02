"""Orchestrator script for Phase 3 (Compression) and Phase 4 (Benchmarking)."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import tensorflow as tf
import tf_keras as keras

from src.data_har import load_har_splits
from src.data_kws import load_kws_splits
from src.convert import convert_to_tflite_fp32, convert_to_tflite_fp16, convert_to_tflite_int8, export_c_array
from src.compress import apply_magnitude_pruning, run_distillation
from src.benchmark import run_benchmark

def main():
    print("\n=======================================================")
    print("PHASE 3 & 4: COMPRESSION, TFLITE CONVERSION & BENCHMARK")
    print("=======================================================")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. HAR Pipeline
    print("\n--- Processing HAR 1D-CNN Compression ---")
    har_splits = load_har_splits()
    X_train_har, y_train_har, _ = har_splits["train"]
    har_model = keras.models.load_model("models/har_baseline.keras", compile=False)
    
    # FP32
    if not os.path.exists("models/har_fp32.tflite"):
        convert_to_tflite_fp32(har_model, "models/har_fp32.tflite")
    # FP16
    if not os.path.exists("models/har_fp16.tflite"):
        convert_to_tflite_fp16(har_model, "models/har_fp16.tflite")
    # INT8
    if not os.path.exists("models/har_int8.tflite"):
        convert_to_tflite_int8(har_model, X_train_har[:300], "models/har_int8.tflite")
    if not os.path.exists("models/har_int8.h"):
        export_c_array("models/har_int8.tflite", "models/har_int8.h", "har_model_int8")
    
    # Pruned variants
    if not os.path.exists("models/har_pruned_50.tflite"):
        har_p50 = apply_magnitude_pruning(har_model, X_train_har, y_train_har, target_sparsity=0.50, epochs=2)
        convert_to_tflite_fp32(har_p50, "models/har_pruned_50.tflite")
    
    if not os.path.exists("models/har_pruned_80.tflite"):
        har_p80 = apply_magnitude_pruning(har_model, X_train_har, y_train_har, target_sparsity=0.80, epochs=2)
        convert_to_tflite_fp32(har_p80, "models/har_pruned_80.tflite")
    
    # 2. KWS Pipeline
    print("\n--- Processing KWS DS-CNN Compression ---")
    kws_splits = load_kws_splits()
    X_train_kws, y_train_kws = kws_splits["train"]
    X_train_kws = np.expand_dims(X_train_kws, -1)
    kws_model = keras.models.load_model("models/kws_baseline.keras", compile=False)
    
    # FP32
    if not os.path.exists("models/kws_fp32.tflite"):
        convert_to_tflite_fp32(kws_model, "models/kws_fp32.tflite")
    # FP16
    if not os.path.exists("models/kws_fp16.tflite"):
        convert_to_tflite_fp16(kws_model, "models/kws_fp16.tflite")
    # INT8
    if not os.path.exists("models/kws_int8.tflite"):
        convert_to_tflite_int8(kws_model, X_train_kws[:300], "models/kws_int8.tflite", is_audio=True)
    if not os.path.exists("models/kws_int8.h"):
        export_c_array("models/kws_int8.tflite", "models/kws_int8.h", "kws_model_int8")
    
    # Pruned variants
    if not os.path.exists("models/kws_pruned_50.tflite"):
        kws_p50 = apply_magnitude_pruning(kws_model, X_train_kws, y_train_kws, target_sparsity=0.50, epochs=2)
        convert_to_tflite_fp32(kws_p50, "models/kws_pruned_50.tflite")
    
    if not os.path.exists("models/kws_pruned_80.tflite"):
        kws_p80 = apply_magnitude_pruning(kws_model, X_train_kws, y_train_kws, target_sparsity=0.80, epochs=2)
        convert_to_tflite_fp32(kws_p80, "models/kws_pruned_80.tflite")
    
    # Knowledge Distillation
    if not os.path.exists("models/kws_student_distilled.tflite"):
        distilled_student = run_distillation(kws_splits, quick=True)
        convert_to_tflite_int8(distilled_student, X_train_kws[:300], "models/kws_student_distilled.tflite", is_audio=True)
    
    # 3. Comprehensive Benchmark & Feasibility
    print("\n--- Running TFLite Benchmarks ---")
    run_benchmark(models_dir="models", results_dir="results")
    print("\n=== Compression and Benchmarking Complete ===")

if __name__ == "__main__":
    main()
