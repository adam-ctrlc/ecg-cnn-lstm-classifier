"""Pick the CNN-LSTM WAV run with the best validation macro AUC and export it to models/ for the app.

    python -m ecgpit.export_onnx
"""
import json

import numpy as np
import onnxruntime as ort
import torch
import yaml

from .config import ROOT
from .data import prepare
from .models import build
from .train import prep


def main():
    runs = [(json.loads(f.read_text()), f.parent) for f in (ROOT / "runs").glob("*_cnn_lstm_wav_s[0-9]*/metrics.json")]
    m, run = max(runs, key=lambda r: r[0]["best_val_macro_auc"])
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    net = build("cnn_lstm", cfg["model"]); net.load_state_dict(torch.load(run / "model.pt", map_location="cpu")); net.eval()
    out = ROOT / "models"; out.mkdir(exist_ok=True)
    x = prep(torch.from_numpy(np.array(prepare(cfg)["x"][:64])), "wav")
    torch.onnx.export(net, x[:1], out / "cnn_lstm.onnx", input_names=["x"], output_names=["logit"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "b", 2: "t"}, "logit": {0: "b"}})
    s = ort.InferenceSession(str(out / "cnn_lstm.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        diff = float(np.abs(s.run(None, {"x": x.numpy()})[0] - net(x).numpy()).max())
    assert diff < 1e-3, diff
    info = dict(run=run.name, selected_by="validation macro ROC-AUC (fold 9)", candidates=len(runs), params=m["params"],
                thresholds=m["thresholds"], val_macro_auc=m["best_val_macro_auc"], test=m["test"],
                size_kb=round((out / "cnn_lstm.onnx").stat().st_size / 1024), train_records=m["train_records"])
    (out / "model.json").write_text(json.dumps(info, indent=2))
    print(f"{run.name} of {len(runs)} runs -> models/cnn_lstm.onnx, max logit difference from PyTorch {diff:.1e}")


if __name__ == "__main__":
    main()
