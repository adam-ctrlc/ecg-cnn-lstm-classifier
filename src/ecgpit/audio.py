"""Carry a 12-lead ECG in an ordinary audio file (MP4, M4A, MP3, OGG, FLAC, ...) and read it back.

    python -m ecgpit.audio recording.wav recording.mp4

Ordinary converters drop leads (MP3 is stereo) and audio codecs discard slow signals. Here the twelve leads are written one after
another and played 80 times faster, so 0 to 50 Hz becomes 0 to 4 kHz. A calibration sweep at the start lets the reader recover the
timing and the voltage, even when a codec adds delay or the volume was changed. Files without the sweep are refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .signal import FS, N_LEADS, InputError, read_wav

BASE, RATE = 8000, 48000
UP = RATE // BASE
GAIN = 4.0
GUARD = 160
SYNC_N, SYNC_AMP = 2400, 12000.0
EXTS = {".wav", ".wave", ".mp4", ".m4a", ".mp3", ".aac", ".ogg", ".oga", ".opus", ".flac", ".mov", ".mka", ".mkv", ".webm", ".aif", ".aiff", ".wma", ".caf"}
ENCODERS = {".mp4": ("aac", 256000), ".m4a": ("aac", 256000), ".mov": ("aac", 256000), ".aac": ("aac", 256000), ".mp3": ("libmp3lame", 320000),
            ".flac": ("flac", 0),
            ".aiff": ("pcm_s16be", 0), ".wma": ("wmav2", 192000)}
LOSSLESS = {"flac", "alac", "wavpack", "tta"}


def _sweep() -> np.ndarray:
    t = np.arange(SYNC_N) / BASE
    phase = 2 * np.pi * (200 * t + (3000 - 200) / (2 * SYNC_N / BASE) * t * t)
    return SYNC_AMP * np.sin(phase) * np.sin(np.pi * np.arange(SYNC_N) / SYNC_N) ** 0.5


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    s = np.fft.rfft(x); out = np.zeros(n // 2 + 1, complex); k = min(s.size, out.size)
    out[:k] = s[:k]
    return np.fft.irfft(out, n) * (n / x.size)


def pack(x: np.ndarray) -> np.ndarray:
    """(n, 12) microvolts -> mono int16 at 48 kHz. A reversed sweep marks the end, so the length can be recovered."""
    parts = [np.zeros(800), _sweep(), np.zeros(800)]
    ramp = np.linspace(0, 1, GUARD)
    for i in range(N_LEADS):
        y = x[:, i].astype(float) * GAIN
        parts += [y[0] * ramp, y, y[-1] * ramp[::-1]]
    z = np.concatenate(parts + [np.zeros(800), _sweep()[::-1], np.zeros(2400)])
    return np.clip(np.round(_resample(z, z.size * UP)), -32767, 32767).astype(np.int16)


def unpack(a: np.ndarray, rate: int) -> np.ndarray:
    """Mono samples at any rate -> (n, 12) microvolts."""
    m = rate // np.gcd(rate, BASE)
    a = np.pad(a, (0, -a.size % m))
    z = _resample(a, a.size * BASE // rate)
    ref = _sweep(); least = 2 * SYNC_N + 1600 + N_LEADS * (FS + 2 * GUARD)
    if z.size < least:
        raise InputError("The audio is too short to hold a 12-lead ECG.")
    n = 1 << int(np.ceil(np.log2(z.size + SYNC_N)))
    f = np.fft.rfft(z, n); e = np.concatenate([[0.0], np.cumsum(z * z)])
    norm = np.sqrt(np.maximum(e[SYNC_N:] - e[:-SYNC_N], 1e-9)) * np.sqrt((ref * ref).sum())
    find = lambda r: np.fft.irfft(f * np.conj(np.fft.rfft(r, n)), n)[: norm.size] / norm
    a0, a1 = find(ref), find(ref[::-1].copy())
    k, end = int(np.argmax(a0)), int(np.argmax(a1))
    if min(a0[k], a1[end]) < 0.8 or end <= k:
        raise InputError("No ECG calibration signal was found. This looks like ordinary audio (music, speech or a microphone heartbeat), not an "
                         "ECG. Create compatible files with Save as audio in this app.")
    count = int(round((end - k - SYNC_N - 1600) / N_LEADS)) - 2 * GUARD
    if count < FS or abs(end - k - SYNC_N - 1600 - N_LEADS * (count + 2 * GUARD)) > 3:
        raise InputError("The ECG audio is damaged or was cut.")
    f3 = np.fft.rfft(z[k:k + SYNC_N]); g3 = np.fft.rfft(ref)
    w = 2 * np.pi * np.fft.rfftfreq(SYNC_N); band = (np.abs(g3) > 0.1 * np.abs(g3).max())
    ph = np.unwrap(np.angle(f3[band] * np.conj(g3[band])))
    slope = np.polyfit(w[band], ph, 1, w=np.abs(g3[band]))[0]
    shift = float(np.clip(-slope, -1, 1))
    if abs(shift) > 1e-3:
        z = np.fft.irfft(np.fft.rfft(z) * np.exp(1j * 2 * np.pi * np.fft.rfftfreq(z.size) * shift), z.size)
    s = z[k:k + SYNC_N]; gain = float((s * ref).sum() / (ref * ref).sum())
    z, p, out = z / gain / GAIN, k + SYNC_N + 800, []
    for _ in range(N_LEADS):
        out.append(z[p + GUARD:p + GUARD + count]); p += count + 2 * GUARD
    return np.stack(out, 1)


def _decode(path):
    try:
        import av
    except ImportError as e:
        raise InputError("Formats other than WAV need the av package: pip install av") from e
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise InputError(f"{Path(path).name} has no audio track.")
            st = c.streams.audio[0]; codec, rate, parts = st.codec_context.name, st.rate, []
            for fr in c.decode(st):
                a = fr.to_ndarray(); ch = fr.layout.nb_channels
                a = a if fr.format.is_planar else a.reshape(-1, ch).T
                k = a.dtype.kind
                parts.append(a.astype(np.float64) * (32768.0 if k == "f" else 1.0 if a.dtype.itemsize == 2 else 1 / 65536 if a.dtype.itemsize == 4 else 256.0))
                if sum(q.shape[1] for q in parts) > 60 * rate:
                    raise InputError(f"{Path(path).name} is longer than 60 s and is not an ECG recording.")
    except InputError:
        raise
    except Exception as e:  # noqa: BLE001
        raise InputError(f"{Path(path).name} could not be decoded as audio ({e}).") from e
    if not parts:
        raise InputError(f"{Path(path).name} has no audio data.")
    return np.concatenate(parts, 1), rate, codec


def load(path) -> tuple[np.ndarray, int, str, bool]:
    """Return (samples x channels, sampling rate, description, lossy)."""
    path = Path(path); ext = path.suffix.lower()
    if ext not in EXTS:
        raise InputError(f"{path.name} is not a supported audio file. Supported: " + ", ".join(sorted(e[1:] for e in EXTS)) + ".")
    if ext in (".wav", ".wave"):
        x, fs = read_wav(path)
        if x.shape[1] > 2 or fs <= 1000:
            return x, fs, "WAV, 12 channels" if x.shape[1] == 12 else "WAV", False
        a, rate, codec = x.T, fs, "pcm"
    else:
        a, rate, codec = _decode(path)
    if a.shape[0] == N_LEADS and rate == FS:
        return a.T.astype(np.float32), FS, f"{ext[1:].upper()}, 12 channels", False
    if a.shape[0] > 2:
        raise InputError(f"{path.name} has {a.shape[0]} channels at {rate:,} Hz. Expected 12 leads at 100 Hz or a file made with Save as audio.")
    x = unpack(a.mean(0), rate).astype(np.float32)
    return x, FS, f"{ext[1:].upper()} ({codec})", not (codec in LOSSLESS or codec.startswith("pcm"))


def save(x: np.ndarray, path) -> None:
    """Write (n, 12) microvolts as a packed audio file in the format given by the extension."""
    path = Path(path); ext = path.suffix.lower(); a = pack(x)
    if ext == ".wav":
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(a.tobytes())
        return
    if ext not in ENCODERS:
        raise InputError("Can save as: wav, " + ", ".join(e[1:] for e in ENCODERS))
    import av
    codec, rate = ENCODERS[ext]
    with av.open(str(path), "w") as c:
        st = c.add_stream(codec, rate=RATE); st.layout = "mono"
        if rate:
            st.bit_rate = rate
        fr = av.AudioFrame.from_ndarray(a[None], format="s16", layout="mono"); fr.sample_rate = RATE
        for r in av.AudioResampler(format=st.format, layout="mono", rate=RATE).resample(fr):
            for p in st.encode(r):
                c.mux(p)
        for p in st.encode(None):
            c.mux(p)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    x, *_ = load(src); save(x, dst); print(f"{src} -> {dst}")
