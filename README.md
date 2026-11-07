# RadarSig — Radar Waveform Recognition from IQ Samples

A reproducible research prototype for generating complex baseband IQ examples, training a 1D CNN, comparing a dense baseline, and evaluating performance by SNR.

## Important reproducibility note

The repository contains code and configuration, not a bundled 40,000-example dataset or pretrained weights. Run the pipeline to generate data and measure results on your machine. Accuracy values should be reported only from your own generated run; the 93% and +12-point figures in the project brief are targets to verify, not precomputed claims.

## Setup

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the full experiment

```bash
python -m radarsig.data.generate --samples 40000 --output data/iq_dataset.npz
python -m radarsig.train --data data/iq_dataset.npz --epochs 20
python -m radarsig.evaluate --data data/iq_dataset.npz --checkpoint artifacts/cnn.pt
```

The generator creates eight waveform classes (BPSK, QPSK, 8PSK, 16QAM, LFM up/down chirps, and two FSK variants), randomizes symbol timing/carrier phase, adds AWGN, and records SNR labels. Dataset split is stratified by waveform class. Evaluation reports overall and per-SNR accuracy, confusion matrices, and a CSV summary.

## Layout

- `radarsig/data/generate.py` — complex IQ simulation and NPZ serialization
- `radarsig/models.py` — 1D CNN and dense baseline
- `radarsig/train.py` — train CNN and baseline; optional noise augmentation
- `radarsig/evaluate.py` — held-out metrics and per-SNR confusion matrices
- `tests/` — generator/model smoke tests

## Notes

This is a synthetic-signal benchmark, not a validated operational radar system. Dataset realism, train/test leakage, class balance, and SNR coverage should be checked before making performance claims.
