"""WAV reading, validation and normalization. Shared by training and the app so both use identical preprocessing."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

FS, N_LEADS, N_SAMPLES, UNIT_UV, MAX_UV = 100, 12, 1000, 250.0, 32767


MIN_S, MAX_S = 5.0, 300.0


class InputError(ValueError):
    """A problem with the submitted recording, worded for the user."""


def read_wav(path) -> tuple[np.ndarray, int]:
    """Return (samples x channels float array, sampling rate)."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise InputError(f"{Path(path).name} is not a readable PCM WAV file ({e}).") from e
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32)
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32)
    elif width == 1:
        x = np.frombuffer(raw, np.uint8).astype(np.float32) - 128
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = (((b[:, 2] << 24) | (b[:, 1] << 16) | (b[:, 0] << 8)) >> 8).astype(np.float32)
    else:
        raise InputError(f"Unsupported sample width: {width * 8} bit.")
    return x[: x.size - x.size % ch].reshape(-1, ch), fs


def lead_residual(x: np.ndarray) -> float:
    """How far the six limb leads are from Einthoven's and Goldberger's relations, relative to their size.

    Measured on the 3,147 PTB-XL WAVs: 99.9% below 0.01, maximum 0.28. Random 12-channel noise: about 1.4.
    """
    x = x - x.mean(axis=0)
    a, b, c, r, l, f = x[:, :6].T
    e = np.concatenate([b - a - c, r + (a + b) / 2, l - (a - c) / 2, f - (b + c) / 2])
    s = float(x[:, :6].std())
    return float(e.std() / s) if s > 0 else 9.0


def checks(x: np.ndarray, fs: int) -> list[tuple[str, bool, str]]:
    """Validation list of (requirement, passed, found value)."""
    n, ch = x.shape
    flat = [i for i in range(ch) if n == 0 or np.ptp(x[:, i]) == 0]
    ok_leads = ch == N_LEADS and lead_residual(x) < 0.5
    peak = float(np.abs(x).max()) if x.size else 0.0
    ok_scale = 50 <= peak < MAX_UV
    return [
        ("12 leads", ch == N_LEADS, f"{ch} channel{'s' if ch != 1 else ''}"),
        ("100 Hz sampling", fs == FS, f"{fs:,} Hz"),
        (f"{MIN_S:g} s to {MAX_S / 60:g} min long", fs > 0 and MIN_S <= n / fs <= MAX_S, f"{n / fs:.2f} s"),
        ("Signal present", len(flat) < ch, "no signal" if len(flat) == ch else
         ("all leads active" if not flat else f"{len(flat)} flat lead{'s' if len(flat) > 1 else ''}")),
        ("True voltage scale", ok_scale, "1 unit = 1 microvolt" if ok_scale else "scaled to full volume" if peak >= MAX_UV else f"peak {peak / 1000:.3f} mV"),
        ("Leads consistent", ok_leads, "limb leads agree" if ok_leads else "limb leads do not behave like an ECG"),
    ]


def normalize(x: np.ndarray) -> np.ndarray:
    """(1000, 12) microvolts -> (12, 1000) float32. Each lead is centered and divided by a fixed 250 microvolts, clipped at 5 mV."""
    x = x.astype(np.float32)
    return np.clip((x - x.mean(axis=0, keepdims=True)) / UNIT_UV, -20, 20).T.copy()


def peak_normalize(x: np.ndarray) -> np.ndarray:
    """Ablation: one scale per file, which removes the absolute voltage (what the first WAV conversion did)."""
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    s = float(x.std())
    return np.clip(x / (s if s > 0 else 1.0), -20, 20).T.copy()


def segments(x: np.ndarray) -> list[int]:
    """Start samples of the 10 s parts of a recording; the last part is aligned to the end. Shorter recordings are one part."""
    n = x.shape[0]
    if n <= N_SAMPLES:
        return [0]
    s = list(range(0, n - N_SAMPLES + 1, N_SAMPLES))
    return s + [n - N_SAMPLES] if n - s[-1] - N_SAMPLES >= FS else s


def heart_rate(x: np.ndarray, fs: int = FS) -> float | None:
    """Rate estimate from R peaks of the combined absolute slope of all leads."""
    d = np.abs(np.diff(x, axis=0)).sum(axis=1)
    k = max(1, int(0.08 * fs))
    e = np.convolve(d, np.ones(k) / k, "same")
    thr = 0.45 * np.percentile(e, 98)
    peaks, gap = [], int(0.27 * fs)
    for i in range(1, e.size - 1):
        if e[i] >= thr and e[i] >= e[i - 1] and e[i] > e[i + 1]:
            if peaks and i - peaks[-1] < gap:
                if e[i] > e[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    if len(peaks) < 3:
        return None
    return float(60 * fs / np.median(np.diff(peaks)))
