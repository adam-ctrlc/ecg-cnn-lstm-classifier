"""Prepare PTB-XL: read every WAV, validate it, map its SCP codes to the five superclasses and cache the arrays.

    python -m ecgpit.data
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from .config import CLASSES, load_config, resolve
from .signal import checks, read_wav


def label_table(root) -> pd.DataFrame:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    to_super = scp[(scp.diagnostic == 1) & scp.diagnostic_class.notna()].diagnostic_class.to_dict()
    sup = db.scp_codes.map(lambda s: {to_super[c] for c in ast.literal_eval(s) if c in to_super})
    for c in CLASSES:
        db[c] = sup.map(lambda s, c=c: int(c in s))
    return db


def prepare(cfg=None):
    """Return dict with x (N, 12, 1000) int16 microvolts, y (N, 5), fold, ecg_id, patient_id. Builds the cache when missing."""
    cfg = cfg or load_config()
    root, cache = resolve(cfg["data"]["root"]), resolve(cfg["data"]["cache"])
    fx, fm = cache / "ptbxl_x.npy", cache / "ptbxl_meta.npz"
    if not (fx.exists() and fm.exists()):
        cache.mkdir(parents=True, exist_ok=True)
        db, rows, xs, ids = label_table(root), [], [], []
        for p in sorted((root / "wav").glob("*/*.wav")):
            i = int(p.stem[:5])
            x, fs = read_wav(p)
            res = checks(x, fs)
            valid, labelled = all(ok for _, ok, _ in res) and x.shape == (1000, 12), bool(db.loc[i, CLASSES].sum())
            rows.append(dict(ecg_id=i, file=f"{p.parent.name}/{p.name}", patient_id=int(db.loc[i, "patient_id"]), fold=int(db.loc[i, "strat_fold"]),
                             valid=valid, problem="; ".join(f"{n}: {v}" for n, ok, v in res if not ok), labelled=labelled,
                             used=valid and labelled, peak_uv=int(np.abs(x).max()), **{c: int(db.loc[i, c]) for c in CLASSES}))
            if valid and labelled:
                xs.append(x.T.astype(np.int16)); ids.append(i)
        pd.DataFrame(rows).to_csv(cache / "dataset.csv", index=False)
        sel = db.loc[ids]
        np.save(fx, np.stack(xs))
        np.savez(fm, y=sel[CLASSES].to_numpy(np.float32), fold=sel.strat_fold.to_numpy(np.int64), ecg_id=np.array(ids),
                 patient_id=sel.patient_id.to_numpy(np.int64))
    with np.load(fm) as z:
        out = {k: z[k] for k in z.files}
    out["x"] = np.load(fx, mmap_mode="r")
    return out


def split(d, cfg):
    c = cfg["data"]
    idx = {"train": np.isin(d["fold"], c["train_folds"]), "val": d["fold"] == c["val_fold"], "test": d["fold"] == c["test_fold"]}
    pats = {k: set(d["patient_id"][v].tolist()) for k, v in idx.items()}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"] or pats["val"] & pats["test"]), "patient overlap"
    return {k: np.flatnonzero(v) for k, v in idx.items()}


if __name__ == "__main__":
    cfg = load_config()
    d = prepare(cfg)
    t = pd.read_csv(resolve(cfg["data"]["cache"]) / "dataset.csv")
    print(f"WAV files {len(t)} | valid {t.valid.sum()} | without a diagnostic superclass {(~t.labelled).sum()} | used {t.used.sum()}")
    print("problems:", t.problem.dropna().loc[lambda s: s != ""].value_counts().head(8).to_dict())
    print("peak microvolts, percentiles 0/1/50/99/99.9/100:", np.percentile(t.peak_uv, [0, 1, 50, 99, 99.9, 100]).round().tolist())
    for k, i in split(d, cfg).items():
        print(f"{k:5s} n={len(i):6d} patients={len(set(d['patient_id'][i])):6d} "
              + " ".join(f"{c}={int(n)}" for c, n in zip(CLASSES, d["y"][i].sum(0)))
              + f" multi-label={(d['y'][i].sum(1) > 1).mean():.1%}")
    print("patient overlap between splits: none")
