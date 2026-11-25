"""Charts painted with QPainter: class probability bars and the 12-lead ECG sheet."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from .config import CLASSES, LEADS, NAMES


class ClassBars(QWidget):
    """One row per class: name, probability bar with a threshold mark, percentage."""

    def __init__(self):
        super().__init__()
        self.r, self.t = None, None
        self.setMinimumSize(380, 176); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set(self, r, t):
        self.r, self.t = r, t; self.update()

    def paintEvent(self, _):
        if not self.r:
            return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        t, r, rh = self.t, self.r, self.height() / 5
        f, fb = QFont(self.font()), QFont(self.font()); f.setPixelSize(13); fb.setPixelSize(13); fb.setBold(True)
        lw = max(QFontMetricsF(fb).horizontalAdvance(NAMES[c]) for c in CLASSES) + 58
        x0, x1 = lw, self.width() - 58
        for i, c in enumerate(CLASSES):
            y, on, v, th = i * rh, c in r["predicted"], r["prob"][c], r["thr"][c]
            col = QColor(t["normal"] if c == "NORM" else t["abn"]) if on else QColor(t["muted"])
            p.setFont(fb if on else f); p.setPen(QColor(t["text"] if on else t["text2"]))
            p.drawText(QRectF(0, y, 50, rh), int(Qt.AlignVCenter | Qt.AlignLeft), c)
            p.drawText(QRectF(50, y, lw - 50, rh), int(Qt.AlignVCenter | Qt.AlignLeft), NAMES[c])
            bar = QRectF(x0, y + rh / 2 - 5, x1 - x0, 10)
            p.setPen(Qt.NoPen); p.setBrush(QColor(t["surface2"])); p.drawRoundedRect(bar, 5, 5)
            if not on:
                col.setAlphaF(0.55)
            p.setBrush(col); p.drawRoundedRect(QRectF(x0, bar.y(), max(6.0, bar.width() * v), 10), 5, 5)
            tx = x0 + bar.width() * th
            p.setPen(QPen(QColor(t["text"]), 2)); p.drawLine(QPointF(tx, bar.y() - 4), QPointF(tx, bar.bottom() + 4))
            p.setFont(fb if on else f); p.setPen(QColor(t["text"] if on else t["text2"]))
            p.drawText(QRectF(x1, y, 58, rh), int(Qt.AlignVCenter | Qt.AlignRight), f"{v * 100:.1f}%")


class EcgSheet(QWidget):
    """Standard 3 x 4 layout with a lead II rhythm strip. Click a lead to see its full 10 s, click again to go back."""

    def __init__(self):
        super().__init__()
        self.x, self.c, self.sel, self.fs = None, None, None, 100
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setMinimumHeight(200)
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, x):
        self.x, self.sel = x, None
        self.peak = float(np.percentile(np.abs(x), 99.7)); self.update()

    def set_colors(self, t):
        self.c = t; self.update()

    def cells(self):
        w, h, d = self.width(), self.height(), self.x.shape[1] / self.fs
        if self.sel is not None:
            return [(self.sel, QRectF(0, 0, w, h), 0.0, d)]
        out = [(col * 3 + row, QRectF(col * w / 4, row * h / 4, w / 4, h / 4), d / 4 * col, d / 4) for col in range(4) for row in range(3)]
        return out + [(1, QRectF(0, 3 * h / 4, w, h / 4), 0.0, d)]

    def paintEvent(self, _):
        p = QPainter(self)
        if self.c is None:
            return
        t, w, h = self.c, self.width(), self.height()
        p.fillRect(self.rect(), QColor(t["paper"]))
        if self.x is None:
            return
        dur = self.x.shape[1] / self.fs; px = w / dur
        for step, key in ((0.04, "grid_minor"), (0.2, "grid_major")):
            if step * px < 4:
                continue
            p.setPen(QPen(QColor(t[key]), 1))
            for k in range(int(dur / step) + 1):
                gx = round(k * step * px) + 0.5; p.drawLine(QPointF(gx, 0), QPointF(gx, h))
            gy = 0.5
            while gy < h:
                p.drawLine(QPointF(0, gy), QPointF(w, gy)); gy += step * px
        f = QFont(self.font()); f.setPixelSize(12); f.setBold(True)
        mm = px / 25
        room = (h / 4 if self.sel is None else h / 2) * 0.85
        gain = next((g for g in (10, 5, 2.5) if self.peak * g * mm <= room), 2.5)
        p.setRenderHint(QPainter.Antialiasing)
        for j, (lead, r, t0, span) in enumerate(self.cells()):
            a, b = int(t0 * self.fs), min(int((t0 + span) * self.fs) + 1, self.x.shape[1])
            y = self.x[lead, a:b]
            xs = r.x() + (np.arange(a, b) / self.fs - t0) / span * r.width()
            ys = r.y() + r.height() * 0.56 - y * gain * mm
            pen = QPen(QColor(t["ink"]), 1.6 if self.sel is None else 2.0); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen)
            p.drawPolyline(QPolygonF([QPointF(u, v) for u, v in zip(xs.tolist(), ys.tolist())]))
            p.setPen(QPen(QColor(t["muted"]), 1))
            if self.sel is None and j < 12 and lead >= 3:
                p.drawLine(QPointF(r.x() + 0.5, r.y() + 8), QPointF(r.x() + 0.5, r.y() + 26))
            s = LEADS[lead] + (f" rhythm, {dur:g} s" if j == 12 else (f", {dur:g} s" if self.sel is not None else ""))
            bg = QColor(t["paper"]); bg.setAlphaF(0.8); tw = QFontMetricsF(f).horizontalAdvance(s)
            p.fillRect(QRectF(r.x() + 5, r.y() + 5, tw + 6, 17), bg)
            p.setFont(f); p.setPen(QColor(t["text"])); p.drawText(QRectF(r.x() + 8, r.y() + 5, tw + 4, 17), int(Qt.AlignLeft | Qt.AlignVCenter), s)

        s = f"25 mm/s, {gain:g} mm/mV"; f.setBold(False); f.setPixelSize(11); p.setFont(f); p.setPen(QColor(t["muted"]))
        p.drawText(QRectF(0, h - 20, w - 8, 16), int(Qt.AlignRight | Qt.AlignVCenter), s)

    def mousePressEvent(self, e):
        if self.x is None:
            return
        self.sel = next((lead for lead, r, *_ in self.cells() if r.contains(e.position())), None) if self.sel is None else None
        self.update()
