# neural-rs-decoder

ML-augmented RS(255, 223) decoder benchmark for byte-level FEC over UDP-based application traffic.

[![CI](https://github.com/kayumov-24B81/neural-rs-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/kayumov-24B81/neural-rs-decoder/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements and benchmarks a Reed-Solomon decoder augmented with a neural network for error position prediction. A multi-layer perceptron predicts the positions of byte errors from the syndrome, and these predictions are passed to a classical Berlekamp-Massey decoder as erasures, exploiting the fact that BM corrects up to twice as many erasures as errors.

The benchmark stand compares three decoders side by side — classical BM (baseline), oracle (upper bound, knows true error positions), and the ML-augmented hybrid — across configurable burst-error channels modelling realistic UDP-based application traffic. The deliverable is the stand itself, designed for systematic evaluation rather than a single best model.

This is a coursework project; correctness, reproducibility, and clean code are prioritised over peak performance.

## Quickstart

### Installation

```bash
git  clone  https://github.com/kayumov-24B81/neural-rs-decoder.git
cd  neural-rs-decoder
pip  install  -e  .  # runtime dependencies only
pip  install  -e  ".[dev]"  # also installs pytest, black, flake8, pre-commit
```

## Run the benchmark

```bash
python  benchmark.py  --config  configs/default.yaml  --samples  100  --decoders  classic
```

Output: a CSV with per-decoder metrics and a YAML with the full run context (config, git commit, environment), both under `results/runs/`.

Output example (per-decoder metrics on GE moderate preset, 100 samples):

| decoder | fer | ber | dfr | precision | recall | mask_covers_all | num_erasures_mean | num_erasures_max | overflow_rate | per_frame_ms |
|---------|------|------|------|-----------|--------|-----------------|-------------------|------------------|---------------|--------------|
| classic | 0.46 | 0.46 | 0.46 | — | — | — | — | — | — | 3.51 |
| oracle | 0.02 | 0.02 | 0.02 | — | — | — | — | — | — | 3.78 |

## Repository structure

```
neural-rs-decoder/
├── benchmark.py # CLI benchmark entry point
├── configs/
│ └── default.yaml # default benchmark config
├── src/
│ ├── channel.py # GE and erasure channel models
│ ├── codec.py # RS encode + Classic / Oracle / Hybrid decoders
│ ├── dataset.py # training dataset (channel-parameterised)
│ ├── evaluate.py # metrics evaluation loop
│ ├── metrics.py # benchmark metrics accumulation
│ ├── model.py # PositionPredictor MLP
│ ├── train.py # training loop
│ └── utils.py # syndrome computation, I/O helpers
├── tests/ # pytest suite
├── notebooks/ # interactive analysis (kept out of CI)
└── results/runs/ # benchmark outputs (csv + yaml per run)
```

## Benchmark stand

### CLI

| Flag | Type | Description |
|------|------|-------------|
| `--config` | path | **Required.** Path to YAML config. |
| `--model` | path | Override `model.path`. |
| `--channel` | preset name | Override `channel.preset` (`light`/`moderate`/`heavy`/`custom`). |
| `--samples` | int | Override `benchmark.num_samples`. |
| `--device` | `cpu`/`cuda`/`auto` | Override device. `auto` uses CUDA if available. |
| `--decoders` | comma-separated | Enable specific decoders (e.g. `classic,oracle`). Overrides `decoders.*`. |
| `--seed` | int | Override `seed`. |
| `--threshold` | float | Override `model.threshold` (neural mask cutoff). |
| `--tag` | string | Override `benchmark.tag` (used in output filename). |
| `--output` | path | Override `output.dir`. |
| `--verbose` / `--no-verbose` | flag | Toggle progress display. |

Example: run only the classical decoder for a quick sanity check:

```bash
python benchmark.py --config configs/default.yaml --decoders classic --samples 100
```

### Configuration

The config controls every detail of a run — channel, model, decoders, sampling, output. CLI flags override individual config fields for the current run; see the table above for the full list.

Base config: [`configs/default.yaml`](configs/default.yaml).

### Output format

Each run produces two files in `results/runs/`, sharing a common `{timestamp}_{tag}` prefix:

-  **`{run_id}.csv`** — one row per decoder, columns: `decoder`, `fer`, `ber`, `dfr`, `precision`, `recall`, `mask_covers_all`, `num_erasures_mean`, `num_erasures_max`, `overflow_rate`, `per_frame_ms`. Mask metrics are `nan` for decoders without a learned mask (classic, oracle).
-  **`{run_id}.yaml`** — full run context: effective config (after CLI overrides), git commit and dirty flag, Python/torch/CUDA versions, device.

The split keeps the CSV clean for plotting and aggregation, while the YAML serves as an audit trail — given a result, you can recover exactly which code, config, and environment produced it.

### Two-pass design

The benchmark runs each set of decoders twice:

1.  **Metrics pass** — gathers correctness statistics (FER, BER, DFR, mask quality). Per-block bookkeeping is allowed; timing is not measured here.
2.  **Timing pass** — measures per-frame decoding latency on freshly generated data, with warmup iterations and `cuda.synchronize()` around the timed loop.

Separating the two prevents metric collection overhead from polluting timing measurements.

## Metrics

-  **FER** — frame error rate; fraction of blocks where the decoded message differs from the original.
-  **DFR** — decoder failure rate; fraction of blocks where the decoder explicitly returned `None`. The difference `FER − DFR` is the rate of *silent* errors (decoder returned a wrong but valid-looking message).
-  **BER** — bit error rate; Hamming distance between decoded and original messages, normalised by total message bits.
-  **precision** / **recall** — quality of the neural error-position mask, measured against the true error positions.
-  **mask_covers_all** — fraction of blocks where the predicted mask is a superset of the true error positions. A more direct measure of mask usefulness for erasure decoding than recall alone.
-  **num_erasures_mean** / **num_erasures_max** — average and maximum size of the predicted mask across blocks.
-  **overflow_rate** — fraction of blocks where the predicted mask exceeds the BM erasure budget (32 positions). High overflow means the network is too generous and wastes its erasure budget on false positives.
-  **per_frame_ms** — average decoding latency per block.

## Channel models

The benchmark uses synthetic channel models for reproducibility and parameter sweeps. The primary channel is **Gilbert-Elliott**, a two-state Markov model that produces realistic burst errors (typical for wireless and noisy network links). An **erasure channel** is also available as a baseline for known-position errors.

### Gilbert-Elliott presets

| Preset | Mean symbol errors / block | Role |
|--------|---------------------------|------|
| `light` | ~7 | Most blocks decoded by classical BM. Sanity-check regime. |
| `moderate` | ~17 | Borderline of classical capability (`t = 16`). Main demo regime, where ML-augmented decoding can show benefit. |
| `heavy` | ~29 | Beyond classical capability. Stress test for the neural mask. |

Presets are calibrated against the RS(255, 223) block size to land in specific decoding regimes. Exact parameters (`p`, `r`, `h`, `k`) are defined in `src/channel.py:GE_PRESETS`.

## Experiments

Experiments will be documented here as they are conducted, with links to result files and analysis notebooks.

## Development

### Running tests

```bash
pytest tests/ -v
```

### Code style

The repo uses `black`, `isort`, and `flake8` enforced via pre-commit hooks. To check before committing:

```bash
pre-commit run --all-files
```

### Reproducibility

Runs are deterministic given a fixed seed: `set_seed` in `src/utils.py` covers Python's `random`, `numpy`, and `torch` (CPU and CUDA), and forces deterministic algorithms. Each benchmark run records the git commit hash and a `dirty` flag (set if there are uncommitted changes) in the YAML sidecar, so any result can be traced back to the exact code that produced it.

## License

MIT — see [LICENSE](LICENSE).