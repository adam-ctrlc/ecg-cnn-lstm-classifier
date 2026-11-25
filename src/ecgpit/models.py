"""Hybrid CNN-LSTM (proposed) and the CNN-only and LSTM-only ablations. Input (B, 12, 1000), output (B, 5) logits."""
import torch
import torch.nn as nn


def conv_block(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride=2, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


def cnn(m):
    layers, cin = [], 12
    for c, k in zip(m["channels"], m["kernels"]):
        layers += [conv_block(cin, c, k), nn.Dropout(m["conv_dropout"])]
        cin = c
    return nn.Sequential(*layers), cin


def head(d, m):
    return nn.Sequential(nn.Dropout(m["head_dropout"]), nn.Linear(d, m["fc"]), nn.ReLU(), nn.Dropout(m["head_dropout"]), nn.Linear(m["fc"], 5))


class CNNLSTM(nn.Module):
    """Five strided convolution blocks (1000 -> 32 steps) learn waveform shape; a bidirectional LSTM models their order in time."""

    def __init__(self, m):
        super().__init__()
        self.cnn, c = cnn(m)
        self.lstm = nn.LSTM(c, m["lstm_hidden"], batch_first=True, bidirectional=True)
        self.head = head(2 * m["lstm_hidden"], m)

    def forward(self, x):
        h, _ = self.lstm(self.cnn(x).transpose(1, 2))
        return self.head(h.mean(dim=1))


class CNNOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.cnn, c = cnn(m)
        self.head = head(c, m)

    def forward(self, x):
        return self.head(self.cnn(x).mean(dim=2))


class LSTMOnly(nn.Module):
    """LSTM over 40 frames of 25 samples x 12 leads, no convolution."""

    def __init__(self, m):
        super().__init__()
        self.lstm = nn.LSTM(12 * 25, m["lstm_hidden"], batch_first=True, bidirectional=True)
        self.head = head(2 * m["lstm_hidden"], m)

    def forward(self, x):
        b = x.shape[0]
        h, _ = self.lstm(x.reshape(b, 12, 40, 25).permute(0, 2, 1, 3).reshape(b, 40, 300))
        return self.head(h.mean(dim=1))


MODELS = {"cnn_lstm": CNNLSTM, "cnn": CNNOnly, "lstm": LSTMOnly}


def build(name, m):
    return MODELS[name](m)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
