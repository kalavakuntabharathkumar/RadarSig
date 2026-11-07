from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import chirp

CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "LFM_UP", "LFM_DOWN", "2FSK", "4FSK"]
SAMPLES_PER_EXAMPLE = 256


def _symbols(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "BPSK":
        return rng.choice([-1.0, 1.0], n).astype(np.complex64)
    if name == "QPSK":
        return np.exp(1j * (np.pi / 4 + rng.integers(0, 4, n) * np.pi / 2)).astype(np.complex64)
    if name == "8PSK":
        return np.exp(1j * (rng.integers(0, 8, n) * 2 * np.pi / 8)).astype(np.complex64)
    if name == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        return ((rng.choice(levels, n) + 1j * rng.choice(levels, n)) / np.sqrt(10)).astype(np.complex64)
    if name in {"LFM_UP", "LFM_DOWN"}:
        t = np.linspace(0, 1, n, endpoint=False)
        return chirp(t, f0=-0.35 if name == "LFM_UP" else 0.35,
                     f1=0.35 if name == "LFM_UP" else -0.35, t1=1,
                     method="linear").astype(np.complex64)
    if name in {"2FSK", "4FSK"}:
        tones = np.array([-0.22, 0.22]) if name == "2FSK" else np.array([-0.3, -0.1, 0.1, 0.3])
        freq = rng.choice(tones, n)
        phase = np.cumsum(2 * np.pi * freq)
        return np.exp(1j * phase).astype(np.complex64)
    raise ValueError(f"Unknown waveform: {name}")


def make_dataset(samples: int = 40000, seed: int = 42, length: int = SAMPLES_PER_EXAMPLE):
    if samples < len(CLASS_NAMES):
        raise ValueError("samples must be at least the number of classes")
    rng = np.random.default_rng(seed)
    x = np.empty((samples, 2, length), dtype=np.float32)
    y = np.empty(samples, dtype=np.int64)
    snrs = rng.choice(np.array([-12, -8, -4, 0, 4, 8, 12, 16, 20]), size=samples)
    labels = np.arange(samples) % len(CLASS_NAMES)
    rng.shuffle(labels)
    for i, label in enumerate(labels):
        name = CLASS_NAMES[int(label)]
        sig = _symbols(name, length, rng)
        # Random carrier phase and small frequency offset make examples less deterministic.
        sig *= np.exp(1j * (rng.uniform(-np.pi, np.pi) + np.arange(length) * rng.uniform(-0.015, 0.015)))
        power = np.mean(np.abs(sig) ** 2) + 1e-12
        noise_power = power / (10 ** (float(snrs[i]) / 10))
        noise = np.sqrt(noise_power / 2) * (rng.normal(size=length) + 1j * rng.normal(size=length))
        noisy = sig + noise
        scale = np.sqrt(np.mean(np.abs(noisy) ** 2)) + 1e-8
        noisy = noisy / scale
        x[i, 0] = noisy.real
        x[i, 1] = noisy.imag
        y[i] = label
    return x, y, snrs.astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic radar IQ examples")
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("data/iq_dataset.npz"))
    args = ap.parse_args()
    x, y, snr = make_dataset(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, snr=snr, class_names=np.array(CLASS_NAMES))
    print(f"Saved {len(y):,} examples to {args.output}; shape={x.shape}")

if __name__ == "__main__":
    main()
