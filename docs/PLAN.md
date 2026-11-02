# Project Plan: Edge Health Monitor

This document tracks execution progress across all phases according to the primary prompt and the UI Addendum.

## Status Summary
- [x] Phase 0: Environment bootstrap, directory setup, requirements, config.yaml, docs/PLAN.md
- [ ] Phase 1: Data ingestion (UCI HAR & Speech Commands v0.02) and feature extraction
- [ ] Phase 2: Baseline model training (HAR 1D-CNN, KWS DS-CNN), classical baselines, overfit/underfit demo
- [ ] Phase 3: TinyML compression (pruning, FP16, INT8 PTQ, knowledge distillation)
- [ ] Phase 4: TFLite conversion, benchmarking, MCU feasibility, C-array export
- [ ] Phase 5: Federated learning simulation (FedAvg IID vs non-IID, communication cost)
- [ ] Phase 6: Adversarial robustness (FGSM, noise defense, SNR analysis)
- [ ] Phase 7: Alert logic (fall rule, keyword plus activity, debouncing, cooldown)
- [ ] Phase 8: Live server, Edge Impulse style UI (14 pages, tokens.css, phone page, replay mode)
- [ ] Phase 8b: Mandatory UI audit (30 checks, headless browser screenshots to results/ui_screenshots/)
- [ ] Phase 9: System architecture diagram, academic report (report.md, report.pdf), WHAT_WAS_NOT_VERIFIED.md
- [ ] Phase 10: PyTest suite, run_all.bat / run_all.sh, final acceptance checklist

## Execution Log
- Initialized plan reflecting all requirements, honesty rules, and UI Addendum specifications.
