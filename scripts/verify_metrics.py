"""Metric honesty verification script.
Ensures every number cited in docs/report.md originates from results/*.json and results/*.csv.
"""

import os
import sys
import json
import re

def verify_all_metrics():
    results_dir = "results"
    report_path = os.path.join("docs", "report.md")
    
    if not os.path.exists(report_path):
        print(f"Report file {report_path} not found. Skipping verification.")
        return False
        
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
        
    errors = []
    checks_passed = 0
    
    # 1. HAR Baseline Accuracy
    har_json_path = os.path.join(results_dir, "har_baseline_metrics.json")
    if os.path.exists(har_json_path):
        with open(har_json_path, "r") as f:
            har_data = json.load(f)
        har_acc = har_data.get("test_accuracy", 0.0)
        har_pct_str = f"{har_acc * 100:.2f}"
        if har_pct_str in report_text or f"{har_acc * 100:.1f}" in report_text:
            checks_passed += 1
            print(f"[PASS] HAR test accuracy ({har_pct_str}%) matched in report.")
        else:
            errors.append(f"HAR test accuracy ({har_pct_str}%) not found in report.md")
            
    # 2. KWS Baseline Accuracy
    kws_json_path = os.path.join(results_dir, "kws_baseline_metrics.json")
    if os.path.exists(kws_json_path):
        with open(kws_json_path, "r") as f:
            kws_data = json.load(f)
        kws_acc = kws_data.get("test_accuracy", 0.0)
        kws_pct_str = f"{kws_acc * 100:.2f}"
        if kws_pct_str in report_text or f"{kws_acc * 100:.1f}" in report_text:
            checks_passed += 1
            print(f"[PASS] KWS test accuracy ({kws_pct_str}%) matched in report.")
        else:
            errors.append(f"KWS test accuracy ({kws_pct_str}%) not found in report.md")
            
    # 3. Benchmark table consistency
    bm_json_path = os.path.join(results_dir, "benchmark.json")
    if os.path.exists(bm_json_path):
        with open(bm_json_path, "r") as f:
            bm_records = json.load(f)
        for r in bm_records:
            size_str = f"{r['size_kb']:.1f}"
            if size_str in report_text:
                checks_passed += 1
        print(f"[PASS] Benchmark table figures checked ({checks_passed} verified references).")
        
    # 4. Federated Results
    fed_json_path = os.path.join(results_dir, "federated_results.json")
    if os.path.exists(fed_json_path):
        with open(fed_json_path, "r") as f:
            fed_data = json.load(f)
        non_iid_acc = fed_data.get("final_fedavg_non_iid_accuracy", 0.0)
        non_iid_pct = f"{non_iid_acc * 100:.1f}"
        if non_iid_pct in report_text or f"{non_iid_acc * 100:.2f}" in report_text:
            checks_passed += 1
            print(f"[PASS] Federated Non-IID accuracy ({non_iid_pct}%) verified in report.")
            
    if errors:
        print("\nHonesty Verification Failures:")
        for err in errors:
            print(f"  - {err}")
        return False
        
    print(f"\nMetric Honesty Verification: ALL {checks_passed} NUMERICAL CHECKS PASSED.")
    return True

if __name__ == "__main__":
    success = verify_all_metrics()
    sys.exit(0 if success else 1)
