# Analysis configurations

`task_relevant_signal.yaml` is the fixed Stage-D comparison of the seven main-study
22-channel inputs. It reads completed Session-2 arrays and does not train or alter a
restoration model or TCFormer.

The analysis reports class-conditional mu/beta log-relative-power error, regularized
affine-invariant covariance distance, and CSP feature preservation. Subject-specific
CSP filters are fitted only on Session-1 True-22 train+validation trials, then frozen
and applied to every aligned Session-2 input. Direct MI-9 has no full-22 spatial
representation, so zero-padded MI-9 is used only as its signal-space reference.

Validate paths, then run the CPU analysis:

```bash
python scripts/run_signal_analysis.py --dry-run
python scripts/run_signal_analysis.py \
  2>&1 | tee artifacts/logs/analysis/task_relevant_signal.log
```

The output directory is immutable by default. Use `--overwrite` only when intentionally
rebuilding the same experiment after a code or figure correction. `--skip-figures`
omits MNE/Matplotlib figure generation while preserving numeric endpoints.

## Formal latency benchmark

`latency_benchmark.yaml` measures online batch-1 input construction/restoration plus
the matched seed-0 TCFormer for all eight retained inputs. It uses 20 warm-up trials
and 288 balanced held-out trials per method, synchronizes every CUDA stage, fixes CPU
threads to one, and includes the full fixed DDIM-100 loop. Model loading, file I/O,
and spline-matrix setup are excluded.

```bash
python scripts/run_latency_benchmark.py --device cpu --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_latency_benchmark.py --device cuda:0
```

Method results are restartable and signature-checked. Do not use `--overwrite` unless
intentionally replacing a completed formal measurement.
