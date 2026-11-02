"""Benchmarking suite for TFLite models on desktop CPU with Cortex-M4 analytic feasibility."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import json
import gzip
import numpy as np
import tensorflow as tf
import tf_keras as keras

from src.convert import get_file_sizes_kb

def compute_model_macs(keras_model):
    """Computes multiply-accumulate operations (MACs) analytically from layer shapes."""
    total_macs = 0
    for layer in keras_model.layers:
        cname = layer.__class__.__name__
        if cname == 'Conv1D':
            # output_length * kernel_size * in_channels * out_channels
            out_shape = layer.output.shape
            out_len = out_shape[1]
            k_size = layer.kernel_size[0]
            in_ch = layer.input.shape[-1]
            out_ch = layer.filters
            total_macs += out_len * k_size * in_ch * out_ch
        elif cname == 'Conv2D':
            out_shape = layer.output.shape
            out_h, out_w = out_shape[1], out_shape[2]
            kh, kw = layer.kernel_size
            in_ch = layer.input.shape[-1]
            out_ch = layer.filters
            total_macs += out_h * out_w * kh * kw * in_ch * out_ch
        elif cname == 'DepthwiseConv2D':
            out_shape = layer.output.shape
            out_h, out_w = out_shape[1], out_shape[2]
            kh, kw = layer.kernel_size
            in_ch = layer.input.shape[-1]
            total_macs += out_h * out_w * kh * kw * in_ch
        elif cname == 'Dense':
            in_features = layer.input.shape[-1]
            out_features = layer.units
            total_macs += in_features * out_features
    return int(total_macs)

def estimate_tensor_arena(interpreter):
    """
    Estimates TFLite Micro tensor arena footprint from interpreter tensor details.
    Sums the memory required for persistent intermediate activation tensors.
    """
    tensor_details = interpreter.get_tensor_details()
    total_bytes = 0
    for t in tensor_details:
        shape = t.get('shape', [])
        dtype = t.get('dtype', np.float32)
        n_elements = np.prod(shape) if len(shape) > 0 else 1
        item_size = np.dtype(dtype).itemsize
        total_bytes += n_elements * item_size
    # In TFLite Micro, arena memory reuses buffer space via planned offsets.
    # Typically 50-70% of total tensor footprint is live concurrently, plus scratch/arena overhead (~4KB).
    arena_estimate_kb = max(4.0, (total_bytes * 0.60 + 4096) / 1024.0)
    return float(round(arena_estimate_kb, 2))

def evaluate_tflite_test_accuracy(tflite_path, X_test, y_test, is_int8=False):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    correct = 0
    total = len(X_test)
    
    scale, zero_point = input_details.get('quantization', (0.0, 0))
    out_scale, out_zero_point = output_details.get('quantization', (0.0, 0))
    
    for i in range(total):
        sample = X_test[i:i+1].astype(np.float32)
        if is_int8 and scale > 0:
            sample = np.clip(np.round(sample / scale) + zero_point, -128, 127).astype(np.int8)
        elif not is_int8 and input_details['dtype'] == np.float32:
            sample = sample.astype(np.float32)
            
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details['index'])
        if is_int8 and out_scale > 0:
            output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale
            
        pred_label = int(np.argmax(output_data))
        if pred_label == y_test[i]:
            correct += 1
            
    return float(correct / total)

def measure_desktop_latency_ms(tflite_path, sample_input, is_int8=False, iterations=200):
    """
    Measures median inference latency on desktop CPU over at least 200 runs.
    Strictly relative to desktop environment, not MCU hardware.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    scale, zero_point = input_details.get('quantization', (0.0, 0))
    sample = sample_input[:1].astype(np.float32)
    if is_int8 and scale > 0:
        sample = np.clip(np.round(sample / scale) + zero_point, -128, 127).astype(np.int8)
        
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0) # in ms
        
    return float(np.median(timings))

def run_benchmark(models_dir="models", results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    from src.data_har import load_har_splits
    from src.data_kws import load_kws_splits
    
    har_splits = load_har_splits()
    X_test_har, y_test_har, _ = har_splits["test"]
    
    kws_splits = load_kws_splits()
    X_test_kws, y_test_kws = kws_splits["test"]
    X_test_kws = np.expand_dims(X_test_kws, -1)
    
    variants = [
        # (model_type, variant_name, filename, is_int8, test_X, test_y, keras_model_file)
        ("har", "FP32", "har_fp32.tflite", False, X_test_har, y_test_har, "har_baseline.keras"),
        ("har", "Pruned (50%)", "har_pruned_50.tflite", False, X_test_har, y_test_har, "har_baseline.keras"),
        ("har", "Pruned (80%)", "har_pruned_80.tflite", False, X_test_har, y_test_har, "har_baseline.keras"),
        ("har", "FP16", "har_fp16.tflite", False, X_test_har, y_test_har, "har_baseline.keras"),
        ("har", "INT8", "har_int8.tflite", True, X_test_har, y_test_har, "har_baseline.keras"),
        ("kws", "FP32", "kws_fp32.tflite", False, X_test_kws, y_test_kws, "kws_baseline.keras"),
        ("kws", "Pruned (50%)", "kws_pruned_50.tflite", False, X_test_kws, y_test_kws, "kws_baseline.keras"),
        ("kws", "Pruned (80%)", "kws_pruned_80.tflite", False, X_test_kws, y_test_kws, "kws_baseline.keras"),
        ("kws", "FP16", "kws_fp16.tflite", False, X_test_kws, y_test_kws, "kws_baseline.keras"),
        ("kws", "INT8", "kws_int8.tflite", True, X_test_kws, y_test_kws, "kws_baseline.keras"),
        ("kws", "Distilled Student", "kws_student_distilled.tflite", True, X_test_kws, y_test_kws, "kws_student_distilled.keras")
    ]
    
    benchmark_records = []
    
    print("\n=== Running Comprehensive TFLite Benchmark ===")
    for model_type, variant, filename, is_int8, test_X, test_y, keras_file in variants:
        tflite_path = os.path.join(models_dir, filename)
        if not os.path.exists(tflite_path):
            continue
            
        print(f"Benchmarking {model_type.upper()} - {variant}...")
        raw_size_kb, gzip_size_kb = get_file_sizes_kb(tflite_path)
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        arena_kb = estimate_tensor_arena(interpreter)
        acc = evaluate_tflite_test_accuracy(tflite_path, test_X, test_y, is_int8=is_int8)
        lat_ms = measure_desktop_latency_ms(tflite_path, test_X, is_int8=is_int8, iterations=200)
        
        # Load keras model for MACs & param count
        keras_path = os.path.join(models_dir, keras_file)
        params = 0
        macs = 0
        if os.path.exists(keras_path):
            try:
                km = keras.models.load_model(keras_path, compile=False)
                params = int(km.count_params())
                macs = compute_model_macs(km)
            except Exception:
                pass
                
        # Analytic Cortex-M4 latency estimate:
        # Cortex-M4 @ 64 MHz, INT8 CMSIS-NN ~ 1.25 cycles/MAC; FP32 ~ 4.0 cycles/MAC
        cycles_per_mac = 1.25 if is_int8 else 4.0
        mcu_clock_mhz = 64.0
        cortex_m4_est_ms = round((macs * cycles_per_mac) / (mcu_clock_mhz * 1e6) * 1000.0, 2)
        
        rec = {
            "model": model_type,
            "variant": variant,
            "filename": filename,
            "test_accuracy": round(acc * 100.0, 2),
            "size_kb": round(raw_size_kb, 2),
            "gzip_size_kb": round(gzip_size_kb, 2),
            "tensor_arena_kb": round(arena_kb, 2),
            "macs": macs,
            "desktop_median_latency_ms": round(lat_ms, 3),
            "cortex_m4_est_latency_ms": cortex_m4_est_ms,
            "params": params,
            "is_int8": is_int8
        }
        benchmark_records.append(rec)
        
    # Write benchmark.json
    with open(os.path.join(results_dir, "benchmark.json"), "w") as f:
        json.dump(benchmark_records, f, indent=2)
        
    # Write benchmark.csv
    csv_path = os.path.join(results_dir, "benchmark.csv")
    with open(csv_path, "w") as f:
        f.write("model,variant,test_accuracy,size_kb,gzip_size_kb,tensor_arena_kb,macs,desktop_latency_ms,cortex_m4_est_ms,params\n")
        for r in benchmark_records:
            f.write(f"{r['model']},{r['variant']},{r['test_accuracy']:.2f},{r['size_kb']:.2f},{r['gzip_size_kb']:.2f},"
                    f"{r['tensor_arena_kb']:.2f},{r['macs']},{r['desktop_median_latency_ms']:.3f},{r['cortex_m4_est_latency_ms']:.2f},{r['params']}\n")
    print(f"Saved benchmark to {csv_path} and benchmark.json")
    
    # Cortex-M4 Feasibility check (256 KB RAM, 1024 KB Flash)
    feasibility = []
    for r in benchmark_records:
        flash_fits = bool(r['size_kb'] <= 1024.0)
        ram_fits = bool(r['tensor_arena_kb'] <= 256.0)
        feasibility.append({
            "model": r["model"],
            "variant": r["variant"],
            "flash_usage_kb": r["size_kb"],
            "flash_limit_kb": 1024.0,
            "flash_feasible": flash_fits,
            "ram_usage_kb": r["tensor_arena_kb"],
            "ram_limit_kb": 256.0,
            "ram_feasible": ram_fits,
            "overall_status": "PASS" if (flash_fits and ram_fits) else "FAIL"
        })
    with open(os.path.join(results_dir, "feasibility_cortex_m4.json"), "w") as f:
        json.dump(feasibility, f, indent=2)
    print("Saved results/feasibility_cortex_m4.json")

    return benchmark_records
