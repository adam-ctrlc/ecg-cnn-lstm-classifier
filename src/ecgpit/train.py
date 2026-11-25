"""Train one model, select the epoch on fold 9, evaluate once on fold 10.

    python -m ecgpit.train --model cnn_lstm --input wav --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from .config import CLASSES, load_config, resolve
from .data import prepare, split
from .metrics import best_thresholds, evaluate, macro_auc
from .models import build, device
from .signal import UNIT_UV


def prep(x, inp):
    """Batch version of signal.normalize ("wav") or signal.peak_normalize ("peak")."""
    x = x.float()
    x = x - x.mean(dim=2, keepdim=True)
    s = UNIT_UV if inp == "wav" else x.flatten(1).std(dim=1, unbiased=False).clamp_min(1e-6)[:, None, None]
    return (x / s).clamp(-20, 20)


def augment(x, t):
    s = int(torch.randint(-t["max_shift"], t["max_shift"] + 1, (1,)))
    x = torch.roll(x, s, dims=2) + t["noise"] * torch.randn_like(x)
    return x * (torch.rand(x.shape[0], 12, 1, device=x.device) >= t["lead_drop"])


@torch.no_grad()
def predict(model, x, inp, bs=256):
    model.eval()
    return torch.cat([torch.sigmoid(model(prep(x[i:i + bs], inp))).cpu() for i in range(0, len(x), bs)]).numpy()


def run(name, inp, seed, cfg, limit=0):
    t = cfg["train"]
    torch.manual_seed(seed); np.random.seed(seed)
    dev, d = device(), prepare(cfg)
    idx = split(d, cfg)
    if limit:
        idx["train"] = idx["train"][d["ecg_id"][idx["train"]] <= limit]
    X = {k: torch.from_numpy(np.asarray(d["x"][i])).to(dev) for k, i in idx.items()}
    Y = {k: torch.from_numpy(d["y"][i]).to(dev) for k, i in idx.items()}
    out = resolve(f"runs/{datetime.now():%Y%m%d-%H%M%S}_{name}_{inp}{'_sub' if limit else ''}_s{seed}"); out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "run": dict(model=name, input=inp, seed=seed, train_limit_ecg_id=limit)}))
    model = build(name, cfg["model"]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    loss_fn, best, bad, hist, n, t0 = nn.BCEWithLogitsLoss(), -1.0, 0, [], len(X["train"]), time.time()
    print(f"{out.name} | {params:,} parameters | train {n} val {len(X['val'])} test {len(X['test'])} | {dev}", flush=True)
    for ep in range(1, t["epochs"] + 1):
        model.train(); tot = 0.0
        for b in torch.randperm(n).split(t["batch_size"]):
            b = b.to(dev)
            if len(b) < 2:
                continue
            loss = loss_fn(model(augment(prep(X["train"][b], inp), t)), Y["train"][b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b)
        pv = predict(model, X["val"], inp); yv = d["y"][idx["val"]]
        vl = float(-(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)).mean()); auc = macro_auc(yv, pv)
        sched.step(auc)
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl, val_macro_auc=auc, lr=opt.param_groups[0]["lr"]))
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
        print(f"epoch {ep:3d} train_loss {tot / n:.4f} val_loss {vl:.4f} val_macro_auc {auc:.4f}{' *' if bad == 0 else ''}", flush=True)
        if bad >= t["patience"]:
            break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    P = {k: predict(model, X[k], inp) for k in ("val", "test")}
    thr = best_thresholds(d["y"][idx["val"]], P["val"])
    m = dict(model=name, input=inp, seed=seed, train_records=n, params=params, epochs_run=len(hist), best_val_macro_auc=best,
             train_seconds=round(time.time() - t0, 1), thresholds=dict(zip(CLASSES, map(float, thr))),
             val=evaluate(d["y"][idx["val"]], P["val"], thr), test=evaluate(d["y"][idx["test"]], P["test"], thr, boot=1000))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    np.savez(out / "predictions.npz", **{f"p_{k}": v for k, v in P.items()}, **{f"y_{k}": d["y"][idx[k]] for k in P},
             **{f"ecg_id_{k}": d["ecg_id"][idx[k]] for k in P}, thresholds=thr)
    print(f"TEST macro AUC {m['test']['macro']['roc_auc']:.4f} CI {m['test']['macro_auc_ci95']} PR-AUC {m['test']['macro']['pr_auc']:.4f} "
          f"F1 {m['test']['macro']['f1']:.4f} | {m['train_seconds']} s", flush=True)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="cnn_lstm"); a.add_argument("--input", default="wav", choices=["wav", "peak"])
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--limit", type=int, default=0, help="train only on ecg_id up to this value"); a = a.parse_args()
    run(a.model, a.input, a.seed, load_config(), a.limit)
