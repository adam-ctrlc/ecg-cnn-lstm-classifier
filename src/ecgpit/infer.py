"""Classify a 12-lead ECG audio file with the saved ONNX model. Needs numpy and onnxruntime, plus av for formats other than WAV.

    python -m ecgpit.infer recording.wav recording.mp4
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import CLASSES, NAMES, resolve
from .audio import load
from .signal import MAX_UV, N_SAMPLES, InputError, checks, heart_rate, lead_residual, normalize, segments

MAX_LOSSY = 0.05


class Predictor:
    def __init__(self, folder="models"):
        import onnxruntime as ort
        folder = resolve(folder)
        self.info = json.loads((folder / "model.json").read_text())
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(folder / "cnn_lstm.onnx"), so, providers=["CPUExecutionProvider"])
        self.thr = np.array([self.info["thresholds"][c] for c in CLASSES])

    def analyze(self, path) -> dict:
        path, t0 = Path(path), time.perf_counter()
        raw, fs, source, lossy = load(path)
        res = checks(raw, fs)
        bad = [f"{n}: found {v}" for n, ok, v in res if not ok]
        bad = bad[:-1] if len(bad) > 1 and not res[-1][1] else bad
        if raw.shape[1] == 12 and np.abs(raw).max() >= MAX_UV:
            raise InputError(f"{path.name} was scaled to full volume, which removes the real voltage the model needs. Use a WAV "
                             "where 1 unit = 1 microvolt. python -m ecgpit.convert makes them from PTB-XL.")
        if bad:
            raise InputError(f"{path.name} does not meet the input requirements. " + "; ".join(bad) + ". Required: 12 ECG leads "
                             "stored as 12 channels at 100 Hz, 5 s to 5 min long. A microphone recording of a heartbeat is not an ECG.")
        if lossy and lead_residual(raw) > MAX_LOSSY:
            raise InputError(f"{path.name} was compressed too strongly: the limb leads no longer agree with each other. Use WAV, FLAC or MP3 at a "
                             "higher bitrate.")
        parts = segments(raw)
        x = np.stack([normalize(raw[s:s + N_SAMPLES]) for s in parts])
        t1 = time.perf_counter()
        logit = self.sess.run(None, {"x": x})[0].astype(float)
        ms = (time.perf_counter() - t1) * 1000
        part_prob = 1 / (1 + np.exp(-logit))
        prob = part_prob.mean(axis=0)
        order = [int(i) for i in np.argsort(-prob)]
        return dict(name=path.stem, path=str(path), source=source, lossy=lossy, raw=raw, fs=fs, duration=len(raw) / fs, checks=res, parts=parts, part_prob=part_prob,
                    x=[(raw[s:s + N_SAMPLES] - raw[s:s + N_SAMPLES].mean(axis=0)).T / 1000.0 for s in parts], bpm=heart_rate(raw),
                    prob={c: float(p) for c, p in zip(CLASSES, prob)}, thr={c: float(t) for c, t in zip(CLASSES, self.thr)},
                    predicted=[CLASSES[i] for i in order if prob[i] >= self.thr[i]], top=CLASSES[order[0]],
                    infer_ms=ms, total_ms=(time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    pred = Predictor()
    for f in sys.argv[1:]:
        try:
            r = pred.analyze(f)
        except InputError as e:
            print(f"rejected: {e}"); continue
        print(f"\nInput ECG recording: {Path(f).name}")
        for c in CLASSES:
            print(f"  {c:5s}{r['prob'][c] * 100:6.1f}%  threshold {r['thr'][c] * 100:4.0f}%{'  <- predicted' if c in r['predicted'] else ''}")
        print("Predicted categories:", ", ".join(f"{c} ({NAMES[c]})" for c in r["predicted"]) or "none reached its threshold")
