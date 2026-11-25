"""ECG Analyzer: native desktop app (PySide6, ONNX Runtime, numpy).

    python -m ecgpit.desktop
"""
from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QMarginsF, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QGuiApplication, QIcon, QLinearGradient, QPageLayout, QPageSize, QPainter,
                           QPainterPath, QPdfWriter, QPen, QPixmap, QPolygonF)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QPushButton, QSizePolicy, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from .audio import EXTS, save
from .charts import ClassBars, EcgSheet
from .config import CLASSES, NAMES, resolve
from .signal import InputError

ICON = Path(__file__).parent / "static" / "app_icon.png"

THEMES = {
    "dark": dict(bg="#0d1117", surface="#151b24", surface2="#1b2330", border="#263041", text="#e8edf5",
                 text2="#b4bfd0", muted="#8592a6", accent="#5b8cff", accent_ink="#0b1020", abn="#ff6b6f",
                 normal="#3ccf91", warn="#f2c94c", warnsoft="#332a10", abnsoft="#3a1c20",
                 paper="#10161f", grid_minor="#19212d", grid_major="#2b384a", ink="#7ef0c0", winfill=(91, 140, 255, 34)),
    "light": dict(bg="#f4f6fa", surface="#ffffff", surface2="#f0f3f8", border="#e2e7ef", text="#121826",
                  text2="#4a5468", muted="#7a8599", accent="#2f6fed", accent_ink="#ffffff", abn="#d93a3f",
                  normal="#12805c", warn="#9a6700", warnsoft="#fff3cd", abnsoft="#fdecec",
                  paper="#fff7f7", grid_minor="#f6d9d9", grid_major="#eba9a9", ink="#1d1d1f", winfill=(47, 111, 237, 24)),
}

# ---------------------------------------------------------------- SVG icons (24x24, stroke = currentColor)
SVG = {
    "heart": '<path d="M12 20.5s-7.5-4.6-9.3-9.3C1.5 7.9 3.6 4.5 7 4.5c2 0 3.5 1.1 5 3 1.5-1.9 3-3 5-3 3.4 0 5.5 3.4 4.3 6.7-1.8 4.7-9.3 9.3-9.3 9.3z" fill="currentColor" stroke="none"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.1"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "alert": '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.4v.1"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/>',
    "folder": '<path d="M3 7.5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "left": '<path d="M15 5l-7 7 7 7"/>',
    "right": '<path d="M9 5l7 7-7 7"/>',
    "download": '<path d="M12 4v11M7 10.5l5 5 5-5M5 19.5h14"/>',
    "list": '<path d="M8 6.5h12M8 12h12M8 17.5h12M4 6.5h.1M4 12h.1M4 17.5h.1"/>',
    "circle": '<circle cx="12" cy="12" r="7"/>',
    "dot": '<circle cx="12" cy="12" r="5" fill="currentColor" stroke="none"/>',
    "wave": '<path d="M4 10v4M8 6.5v11M12 4v16M16 7.5v9M20 10.5v3"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
}
_svg_cache: dict = {}


def svg_pixmap(name: str, color: str, size: int = 16) -> QPixmap:
    """Render an inline SVG icon once per (name, color, size, dpr) and cache it."""
    app = QGuiApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    key = (name, color, size, dpr)
    pm = _svg_cache.get(key)
    if pm is None:
        doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
               + SVG[name].replace("currentColor", color) + "</svg>")
        pm = QPixmap(int(size * dpr), int(size * dpr)); pm.fill(Qt.transparent)
        p = QPainter(pm); QSvgRenderer(QByteArray(doc.encode())).render(p); p.end()
        pm.setDevicePixelRatio(dpr); _svg_cache[key] = pm
    return pm


def svg_icon(name: str, color: str, size: int = 16) -> QIcon:
    return QIcon(svg_pixmap(name, color, size))


def swatch_pixmap(color: str, w: int = 10, h: int = 10, dashed: bool = False) -> QPixmap:
    """Small vector legend mark: a filled rounded square or a dashed line."""
    app = QGuiApplication.instance(); dpr = app.devicePixelRatio() if app else 1.0
    pm = QPixmap(int(w * dpr), int(h * dpr)); pm.fill(Qt.transparent); pm.setDevicePixelRatio(dpr)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    if dashed:
        pen = QPen(QColor(color)); pen.setWidthF(1.5); pen.setStyle(Qt.DashLine); p.setPen(pen)
        p.drawLine(QPointF(0, h / 2), QPointF(w, h / 2))
    else:
        p.setPen(Qt.NoPen); p.setBrush(QColor(color)); p.drawRoundedRect(QRectF(0, 0, w, h), 2, 2)
    p.end()
    return pm


def swatch(color: str, w: int = 10, h: int = 10) -> QLabel:
    lab = QLabel(); lab.setPixmap(swatch_pixmap(color, w, h)); lab.setFixedSize(w, h); return lab


def stylesheet(t: dict) -> str:
    return f"""
    QMainWindow, QWidget#root, QWidget#page {{ background: {t['bg']}; }}
    QWidget {{ color: {t['text']}; font-size: 13px; }}
    QFrame#card {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 14px; }}
    QFrame#topbar {{ background: {t['surface']}; border-bottom: 1px solid {t['border']}; }}
    QFrame#sidebar {{ background: {t['surface']}; border-right: 1px solid {t['border']}; }}
    QFrame#divider {{ background: {t['border']}; border: none; }}
    QFrame#vdivider {{ background: {t['border']}; border: none; }}
    QFrame#logobadge {{ background: {t['abn']}; border-radius: 10px; }}
    QLabel#apptitle {{ font-size: 15px; font-weight: 700; }}
    QLabel#muted {{ color: {t['muted']}; font-size: 12px; }}
    QLabel#lead {{ color: {t['text2']}; font-size: 15px; }}
    QLabel#h2 {{ color: {t['text']}; font-size: 15px; font-weight: 650; }}
    QLabel#sidehead {{ font-size: 14px; font-weight: 650; }}
    QLabel#sidelabel {{ color: {t['text2']}; }}
    QLabel#modelsum {{ font-size: 12px; }}
    QLabel#monlbl {{ color: {t['text2']}; font-size: 12px; font-weight: 600; }}
    QLabel#emptytitle {{ font-size: 22px; font-weight: 800; }}
    QLabel#rectitle {{ font-size: 22px; font-weight: 700; }}
    QLabel#eyebrow {{ color: {t['muted']}; font-size: 12px; font-weight: 600; }}
    QLabel#verdicttext {{ font-size: 22px; font-weight: 750; }}
    QLabel#body {{ color: {t['text2']}; font-size: 13px; }}
    QLabel#reftext {{ color: {t['text2']}; font-size: 12px; }}
    QLabel#metrick {{ color: {t['text2']}; }}
    QLabel#metricv {{ font-size: 15px; font-weight: 650; }}
    QLabel#legend {{ color: {t['text2']}; font-size: 12px; }}
    QLabel#segprob {{ color: {t['text2']}; font-size: 12px; font-weight: 600; }}
    QLabel#kvk {{ color: {t['muted']}; }}
    QLabel#recname {{ font-weight: 600; }}
    QLabel#recsub {{ color: {t['muted']}; font-size: 11px; }}
    QLabel#ovtitle {{ font-size: 19px; font-weight: 750; }}
    QLabel#steptext {{ color: {t['text2']}; }}
    QLabel#steptext[state="current"] {{ color: {t['text']}; font-weight: 700; }}
    QLabel#steptext[state="pending"] {{ color: {t['muted']}; }}
    QFrame#errorcard {{ background: {t['abnsoft']}; border: 1px solid {t['abn']}; border-radius: 12px; }}
    QFrame#warnbar {{ background: {t['warnsoft']}; border: 1px solid {t['warn']}; border-radius: 12px; }}
    QFrame#overlaycard {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 20px; }}
    QFrame#segbox {{ background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 9px; }}
    QPushButton {{ background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 9px; padding: 7px 14px; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: {t['accent_ink']}; border: none; font-weight: 700; padding: 11px 18px; font-size: 14px; }}
    QPushButton#flat {{ background: transparent; border: none; color: {t['text2']}; padding: 6px 10px; }}
    QPushButton#flat:hover {{ color: {t['text']}; }}
    QPushButton#link {{ background: transparent; border: none; color: {t['accent']}; padding: 6px 0; font-weight: 600; text-align: left; }}
    QPushButton#iconbtn {{ padding: 0; }}
    QPushButton#segbtn {{ background: transparent; border: none; border-radius: 7px; padding: 5px 7px; color: {t['text2']}; }}
    QPushButton#segbtn:hover {{ color: {t['text']}; }}
    QPushButton#segbtn:checked {{ background: {t['surface']}; color: {t['text']}; font-weight: 600; border: 1px solid {t['border']}; }}
    QListWidget#reclist {{ background: transparent; border: none; outline: none; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 40px; }}
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; width: 0; background: none; }}
    QScrollBar:horizontal {{ height: 0; }}
    QProgressBar {{ border: none; background: {t['surface2']}; border-radius: 4px; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 4px; }}
    QToolTip {{ background: {t['text']}; color: {t['surface']}; border: none; padding: 5px 9px; }}
    QMenuBar {{ background: {t['surface']}; color: {t['text']}; }}
    QDialog {{ background: {t['surface']}; }}
    QStatusBar {{ background: {t['surface']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
    """


def card(title=None, margins=(20, 16, 20, 14)):
    f = QFrame(objectName="card")
    lay = QVBoxLayout(f); lay.setContentsMargins(*margins); lay.setSpacing(10)
    if title:
        lay.addWidget(QLabel(title, objectName="h2"))
    return f, lay


def wrap(label: QLabel) -> QLabel:
    label.setWordWrap(True); label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum); return label


def divider(vertical=False) -> QFrame:
    d = QFrame(objectName="vdivider" if vertical else "divider")
    (d.setFixedWidth if vertical else d.setFixedHeight)(1)
    return d


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.1f} s"
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {int(s % 60)} s"


def fmt_clock(s: float) -> str:
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------- animation
class AnimClock(QObject):
    """One shared 20 fps timer for every animation (measured: 30 fps = 2.2% CPU idle, 20 fps lower); paused when not in front."""
    tick = Signal(float)

    def __init__(self):
        super().__init__()
        self.timer = QTimer(self); self.timer.setInterval(50); self.timer.setTimerType(Qt.CoarseTimer)
        self.timer.timeout.connect(lambda: self.tick.emit(0.050))
        self.timer.start()

    def set_active(self, on: bool):
        if on and not self.timer.isActive():
            self.timer.start()
        elif not on:
            self.timer.stop()


CLOCK: AnimClock | None = None


def heart_path(w, h, x0=0.0, y0=0.0) -> QPainterPath:
    p = QPainterPath(); P = lambda x, y: QPointF(x0 + x * w, y0 + y * h)
    p.moveTo(P(0.5, 0.92))
    p.cubicTo(P(0.12, 0.64), P(-0.02, 0.38), P(0.16, 0.17)); p.cubicTo(P(0.30, 0.02), P(0.48, 0.08), P(0.5, 0.27))
    p.cubicTo(P(0.52, 0.08), P(0.70, 0.02), P(0.84, 0.17)); p.cubicTo(P(1.02, 0.38), P(0.88, 0.64), P(0.5, 0.92))
    return p


def beat_envelope(t: float) -> float:
    """'lub-dub' contraction profile over one cardiac cycle (t in [0, 1))."""
    return float(np.exp(-((t - 0.06) / 0.045) ** 2) + 0.55 * np.exp(-((t - 0.28) / 0.05) ** 2))


_T = np.linspace(0, 1, 400, endpoint=False)
_ECG_LUT = (0.12 * np.exp(-((_T - 0.12) / 0.025) ** 2) - 0.12 * np.exp(-((_T - 0.255) / 0.008) ** 2)
            + 1.0 * np.exp(-((_T - 0.27) / 0.009) ** 2) - 0.22 * np.exp(-((_T - 0.285) / 0.01) ** 2)
            + 0.28 * np.exp(-((_T - 0.48) / 0.045) ** 2))   # one synthetic beat, precomputed once


class BeatingHeart(QWidget):
    def __init__(self, size=26, bpm=66.0, color="#ff6b6f"):
        super().__init__()
        self.setFixedSize(size, size); self.bpm, self.color, self.phase = bpm, QColor(color), 0.0
        self.base = heart_path(1.0, 1.0)
        CLOCK.tick.connect(self._tick)

    def set(self, bpm=None, color=None):
        if bpm:
            self.bpm = float(np.clip(bpm, 30, 200))
        if color:
            self.color = QColor(color)
        self.update()

    def _tick(self, dt):
        self.phase = (self.phase + dt * self.bpm / 60.0) % 1.0
        if self.isVisible():
            self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        s = self.width() * (0.78 + 0.16 * beat_envelope(self.phase))
        p.translate((self.width() - s) / 2, (self.height() - s) / 2 + self.height() * 0.03); p.scale(s, s)
        p.setPen(Qt.NoPen); p.setBrush(self.color); p.drawPath(self.base)


class MonitorStrip(QWidget):
    """Bedside-monitor style sweeping ECG trace.
    Measured: per-frame cost dominated by building polylines; now one gradient-pen polyline per run and a cached grid."""
    def __init__(self, width=240, height=34, bpm=66.0):
        super().__init__()
        self.setFixedSize(width, height)
        self.bpm, self.n, self.head, self.phase = bpm, width // 3, 0, 0.0
        self.buf = np.zeros(self.n, np.float32)
        self.xs = np.linspace(0, width, self.n)
        self.color, self.grid, self._grid_pm = QColor("#3ccf91"), QColor("#263041"), None
        CLOCK.tick.connect(self._tick)

    def set(self, bpm=None, color=None):
        if bpm:
            self.bpm = float(np.clip(bpm, 30, 200))
        if color:
            self.color = QColor(color)

    def set_grid(self, color):
        self.grid, self._grid_pm = QColor(color), None

    def _tick(self, dt):
        for _ in range(3):
            self.phase = (self.phase + dt / 3 * self.bpm / 60.0) % 1.0
            self.buf[self.head] = _ECG_LUT[int(self.phase * 400) % 400]
            self.head = (self.head + 1) % self.n
        if self.isVisible():
            self.update()

    def _grid(self):
        if self._grid_pm is None:                              # grid drawn once, reused every frame
            dpr = self.devicePixelRatioF()
            pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr)); pm.setDevicePixelRatio(dpr); pm.fill(Qt.transparent)
            g = QPainter(pm); g.setPen(QPen(self.grid, 1))
            for gx in range(0, self.width(), 12):
                g.drawLine(gx, 0, gx, self.height())
            for gy in range(0, self.height(), 12):
                g.drawLine(0, gy, self.width(), gy)
            g.end(); self._grid_pm = pm
        return self._grid_pm

    def paintEvent(self, _):
        p = QPainter(self); p.drawPixmap(0, 0, self._grid()); p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        ys = h * 0.72 - self.buf * h * 0.62
        newest = (self.head - 1) % self.n
        start = (self.head + 3) % self.n                       # small gap ahead of the sweep head
        runs = [(start, self.n), (0, newest + 1)] if start > newest else [(start, newest + 1)]
        faded = QColor(self.color); faded.setAlphaF(0.15)
        for a, b in runs:
            if b - a < 2:
                continue
            grad = QLinearGradient(self.xs[a], 0, self.xs[b - 1], 0)
            age = lambda j: ((newest - j) % self.n) / self.n
            c0 = QColor(self.color); c0.setAlphaF(max(0.15, 1 - age(a)))
            c1 = QColor(self.color); c1.setAlphaF(max(0.15, 1 - age(b - 1)))
            grad.setColorAt(0, c0); grad.setColorAt(1, c1)
            pen = QPen(QBrush(grad), 1.8); p.setPen(pen)
            p.drawPolyline(QPolygonF(list(map(QPointF, self.xs[a:b], ys[a:b]))))
        p.setPen(Qt.NoPen); p.setBrush(self.color); p.drawEllipse(QPointF(self.xs[newest], ys[newest]), 2.6, 2.6)


class Pulse(QWidget):
    """Animated ECG trace for the empty state and the loading screen."""
    def __init__(self, w=340, h=90):
        super().__init__()
        self.setFixedSize(w, h); self.phase, self.color, self.dim = 0.0, QColor("#ff6b6f"), QColor("#263041")
        beat = [0, 0, 0.05, 0.12, 0.05, 0, 0, -0.15, 1.0, -0.35, 0, 0, 0.08, 0.2, 0.25, 0.2, 0.08, 0, 0, 0, 0, 0, 0, 0]
        wave = np.array(beat * 4, float); n = len(wave)
        self.pts = [QPointF(i / (n - 1) * w, h * 0.62 - v * h * 0.5) for i, v in enumerate(wave)]
        self.base = QPolygonF(self.pts)
        CLOCK.tick.connect(self._tick)

    def _tick(self, dt):
        if self.isVisible():
            self.phase = (self.phase + dt * 0.7) % 1.0; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.dim); pen.setWidthF(2.0); p.setPen(pen); p.drawPolyline(self.base)
        head = int(self.phase * (len(self.pts) - 1))
        tail = self.pts[max(0, head - 14):head + 1]
        if len(tail) > 1:
            pen = QPen(self.color); pen.setWidthF(3.0); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
            p.drawPolyline(QPolygonF(tail))
        p.setPen(Qt.NoPen); p.setBrush(self.color); p.drawEllipse(self.pts[head], 4.5, 4.5)


class RecDelegate(QStyledItemDelegate):
    """Paints a recording row; widgets per row are too slow for thousands of rows."""
    colors = THEMES["dark"]

    def sizeHint(self, *_):
        return QSize(0, 46)

    def paint(self, p, opt, index):
        t, (path, labels), r = self.colors, index.data(Qt.UserRole), QRectF(opt.rect).adjusted(0.5, 1.5, -0.5, -1.5)
        p.save(); p.setRenderHint(QPainter.Antialiasing)
        sel, hov = bool(opt.state & QStyle.State_Selected), bool(opt.state & QStyle.State_MouseOver)
        if sel or hov:
            p.setPen(QPen(QColor(t["accent"]), 1) if sel else Qt.NoPen); p.setBrush(QColor(t["surface2"])); p.drawRoundedRect(r, 9, 9)
        p.setPen(Qt.NoPen); p.setBrush(QColor(t["normal"] if tuple(labels) == ("NORM",) else t["abn"]))
        p.drawRoundedRect(QRectF(r.x() + 10, r.center().y() - 4.5, 9, 9), 2, 2)
        f = QFont(opt.font); f.setPixelSize(13); f.setBold(True); p.setFont(f); p.setPen(QColor(t["text"]))
        p.drawText(QRectF(r.x() + 29, r.y() + 4, r.width() - 34, 18), int(Qt.AlignVCenter), Path(path).stem)
        f.setPixelSize(11); f.setBold(False); p.setFont(f); p.setPen(QColor(t["muted"]))
        p.drawText(QRectF(r.x() + 29, r.y() + 22, r.width() - 34, 16), int(Qt.AlignVCenter), "Reference: " + ", ".join(labels)); p.restore()


class Bridge(QObject):
    done = Signal(object)
    failed = Signal(str)
    ready = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Analyzer")
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1480, avail.width() - 20), min(960, avail.height() - 20)); self.setMinimumSize(1120, 700)
        self.setAcceptDrops(True)
        self.theme, self.pred, self.res, self.truth, self.current, self.filter = "dark", None, None, None, None, "All"
        self.samples, self.busy = [], False
        self.bridge = Bridge()
        self.bridge.done.connect(self.on_result); self.bridge.failed.connect(self.on_error); self.bridge.ready.connect(self.on_ready)
        root = QWidget(objectName="root"); self.setCentralWidget(root)
        v = QVBoxLayout(root); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self.build_topbar())
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(0)
        row.addWidget(self.build_sidebar())
        right = QWidget(objectName="page"); rl = QVBoxLayout(right); rl.setContentsMargins(20, 16, 20, 14); rl.addWidget(self.build_right())
        row.addWidget(right, 1); v.addLayout(row, 1)
        self.statusBar().hide(); self.build_menu(); self.apply_theme()
        threading.Thread(target=self.load_model, daemon=True).start()

    @property
    def t(self):
        return THEMES[self.theme]

    def build_topbar(self):
        bar = QFrame(objectName="topbar"); h = QHBoxLayout(bar); h.setContentsMargins(20, 10, 20, 10); h.setSpacing(12)
        badge = QFrame(objectName="logobadge"); badge.setFixedSize(38, 38)
        bl = QHBoxLayout(badge); bl.setContentsMargins(0, 0, 0, 0)
        self.logo_heart = BeatingHeart(24, 66, "#ffffff"); bl.addWidget(self.logo_heart, 0, Qt.AlignCenter); h.addWidget(badge)
        tv = QVBoxLayout(); tv.setSpacing(0)
        tv.addWidget(QLabel("ECG Analyzer", objectName="apptitle"))
        tv.addWidget(QLabel("12-lead ECG classification · hybrid CNN-LSTM", objectName="muted")); h.addLayout(tv); h.addStretch(1)
        self.monitor = MonitorStrip(240, 34); self.monitor.setToolTip("Follows the heart rate of the analyzed recording"); h.addWidget(self.monitor)
        self.mon_icon = QLabel(); self.mon_icon.setFixedSize(13, 13); h.addWidget(self.mon_icon)
        self.mon_lbl = QLabel("66 bpm", objectName="monlbl"); self.mon_lbl.setMinimumWidth(70); h.addWidget(self.mon_lbl)
        self.theme_btn = QPushButton(objectName="flat"); self.theme_btn.setFixedSize(36, 32); self.theme_btn.setIconSize(QSize(18, 18))
        self.theme_btn.setToolTip("Light / dark"); self.theme_btn.clicked.connect(self.toggle_theme); h.addWidget(self.theme_btn)
        return bar

    def build_sidebar(self):
        side = QFrame(objectName="sidebar"); side.setFixedWidth(320)
        lay = QVBoxLayout(side); lay.setContentsMargins(22, 22, 22, 18); lay.setSpacing(0)
        self.open_btn = QPushButton("  Upload ECG audio", objectName="primary"); self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.setIconSize(QSize(18, 18)); self.open_btn.clicked.connect(self.browse); lay.addWidget(self.open_btn)
        hint = QHBoxLayout(); hint.setContentsMargins(2, 8, 2, 0)
        self.drop_hint = QLabel("or drop a file anywhere in the window", objectName="muted"); hint.addWidget(self.drop_hint); hint.addStretch(1)
        self.info_icon = QLabel(); self.info_icon.setFixedSize(15, 15)
        self.info_icon.setToolTip("WAV with 12 channels (the 12 ECG leads), 100 Hz, 5 s to 5 min, 1 unit = 1 microvolt,\nor MP4, M4A, MP3, AAC, FLAC, MOV, WMA and others made with Save as audio.\nMusic and microphone recordings of a heartbeat are not ECG and are refused.")
        hint.addWidget(self.info_icon); lay.addLayout(hint)
        lay.addSpacing(18); lay.addWidget(divider()); lay.addSpacing(18)
        hr = QHBoxLayout(); hr.addWidget(QLabel("Test recordings", objectName="sidehead")); hr.addStretch(1)
        self.count_lbl = QLabel("", objectName="muted"); hr.addWidget(self.count_lbl); lay.addLayout(hr); lay.addSpacing(10)
        box = QFrame(objectName="segbox"); bh = QHBoxLayout(box); bh.setContentsMargins(3, 3, 3, 3); bh.setSpacing(0)
        grp = QButtonGroup(box); grp.setExclusive(True); names = ["All"] + CLASSES
        for i, n in enumerate(names):
            b = QPushButton(n, objectName="segbtn"); b.setCheckable(True); b.setChecked(i == 0); b.setCursor(Qt.PointingHandCursor)
            f = b.font(); f.setPixelSize(11); b.setFont(f); grp.addButton(b, i); bh.addWidget(b, 1)
        grp.idClicked.connect(lambda i: self.set_filter(names[i])); lay.addWidget(box); lay.addSpacing(10)
        self.rec_list = QListWidget(objectName="reclist"); self.rec_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rec_list.setVerticalScrollMode(QListWidget.ScrollPerPixel); self.rec_list.setFocusPolicy(Qt.NoFocus)
        self.rec_list.setUniformItemSizes(True); self.rec_list.setMouseTracking(True)
        self.delegate = RecDelegate(self.rec_list); self.rec_list.setItemDelegate(self.delegate)
        self.rec_list.itemClicked.connect(lambda it: self.analyze(it.data(Qt.UserRole)[0], it.data(Qt.UserRole)[1]))
        lay.addWidget(self.rec_list, 1); lay.addSpacing(6)
        self.rec_note = wrap(QLabel("PTB-XL fold 10. The model never saw these recordings or patients in training.", objectName="muted"))
        lay.addWidget(self.rec_note); lay.addSpacing(16); lay.addWidget(divider()); lay.addSpacing(14)
        self.model_sum = wrap(QLabel("Loading model...", objectName="modelsum")); self.model_sum.setTextFormat(Qt.RichText); lay.addWidget(self.model_sum)
        about = QPushButton("About the model", objectName="link"); about.setCursor(Qt.PointingHandCursor); about.clicked.connect(self.show_about)
        lay.addWidget(about, 0, Qt.AlignLeft)
        return side

    def build_right(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        self.err = QFrame(objectName="errorcard"); el = QHBoxLayout(self.err); el.setContentsMargins(16, 12, 16, 12); el.setSpacing(12)
        self.err_icon = QLabel(); self.err_icon.setFixedSize(20, 20); el.addWidget(self.err_icon, 0, Qt.AlignTop)
        self.err_text = wrap(QLabel()); el.addWidget(self.err_text, 1); self.err.hide(); lay.addWidget(self.err)

        self.empty, ec = card(margins=(60, 40, 60, 40)); ec.setSpacing(14); ec.addStretch(1)
        self.pulse = Pulse(300, 80); ec.addWidget(self.pulse, 0, Qt.AlignHCenter)
        h = wrap(QLabel("Upload an ECG audio file or pick a test recording", objectName="emptytitle")); h.setAlignment(Qt.AlignCenter); ec.addWidget(h)
        s = wrap(QLabel("The audio is checked, normalized and passed through the trained CNN-LSTM, which gives a probability for each "
                        "of five classes: " + ", ".join(NAMES[c] for c in CLASSES) + ".", objectName="lead"))
        s.setAlignment(Qt.AlignCenter); ec.addWidget(s); ec.addStretch(1); lay.addWidget(self.empty, 1)

        self.results = QWidget(); rl = QVBoxLayout(self.results); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(12)
        head = QHBoxLayout(); head.setSpacing(10); tc = QVBoxLayout(); tc.setSpacing(2)
        self.rec_title = QLabel(objectName="rectitle"); self.rec_meta = QLabel(objectName="muted"); tc.addWidget(self.rec_title); tc.addWidget(self.rec_meta)
        head.addLayout(tc, 1)
        self.details_btn = QPushButton("  Technical details"); self.details_btn.clicked.connect(self.show_details)
        self.export_btn = QPushButton("  Export PDF report"); self.export_btn.clicked.connect(lambda: self.export_pdf())
        self.audio_btn = QPushButton("  Save as audio"); self.audio_btn.clicked.connect(self.save_audio)
        self.audio_btn.setToolTip("Save this ECG as MP4, MP3, M4A, FLAC and others that this app can read back")
        for b in (self.details_btn, self.audio_btn, self.export_btn):
            b.setIconSize(QSize(16, 16)); b.setCursor(Qt.PointingHandCursor); head.addWidget(b, 0, Qt.AlignVCenter)
        rl.addLayout(head)

        summ = QFrame(objectName="card"); sl = QHBoxLayout(summ); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(0)
        a = QVBoxLayout(); a.setContentsMargins(22, 16, 18, 14); a.setSpacing(6)
        a.addWidget(QLabel("Identified classes", objectName="eyebrow"))
        self.verdict = wrap(QLabel(objectName="verdicttext")); self.verdict.setTextFormat(Qt.RichText); a.addWidget(self.verdict)
        self.summary = wrap(QLabel(objectName="body")); a.addWidget(self.summary)
        self.truth_lbl = wrap(QLabel(objectName="reftext")); a.addWidget(self.truth_lbl); a.addStretch(1); sl.addLayout(a, 4)
        sl.addWidget(divider(True))
        b = QVBoxLayout(); b.setContentsMargins(22, 14, 18, 12); b.setSpacing(6)
        bh = QHBoxLayout(); bh.addWidget(QLabel("Probability per class", objectName="eyebrow")); bh.addStretch(1)
        bh.addWidget(QLabel("vertical mark = threshold", objectName="muted")); b.addLayout(bh)
        self.bars = ClassBars(); b.addWidget(self.bars, 1); sl.addLayout(b, 6)
        sl.addWidget(divider(True))
        c = QVBoxLayout(); c.setContentsMargins(20, 16, 20, 14); c.setSpacing(7); c.addWidget(QLabel("Input check", objectName="eyebrow"))
        self.check_rows = []
        for _ in range(6):
            r = QHBoxLayout(); r.setSpacing(8); ic = QLabel(); ic.setFixedSize(15, 15); tx = QLabel(objectName="metrick"); r.addWidget(ic); r.addWidget(tx, 1)
            c.addLayout(r); self.check_rows.append((ic, tx))
        hrw = QHBoxLayout(); hrw.setSpacing(8); self.hr_heart = BeatingHeart(15); hrw.addWidget(self.hr_heart)
        self.hr_lbl = QLabel(objectName="metrick"); hrw.addWidget(self.hr_lbl, 1); c.addLayout(hrw); c.addStretch(1); sl.addLayout(c, 3)
        rl.addWidget(summ)

        cf, cl = card(margins=(20, 12, 20, 12)); eh = QHBoxLayout(); eh.addWidget(QLabel("12-lead ECG", objectName="h2")); eh.addStretch(1)
        self.part_prev, self.part_next = QPushButton(objectName="iconbtn"), QPushButton(objectName="iconbtn")
        self.part_lbl = QLabel(objectName="muted")
        for b_, d_ in ((self.part_prev, -1), (self.part_next, 1)):
            b_.setFixedSize(30, 26); b_.setIconSize(QSize(15, 15)); b_.clicked.connect(lambda _=False, d=d_: self.show_part(self.part + d))
        for w_ in (self.part_prev, self.part_lbl, self.part_next):
            eh.addWidget(w_)
        eh.addSpacing(14); eh.addWidget(QLabel("click a lead to enlarge it", objectName="muted")); cl.addLayout(eh)
        self.sheet = EcgSheet(); cl.addWidget(self.sheet, 1); rl.addWidget(cf, 1)
        foot = QLabel("Research demonstration only, not a medical device.", objectName="muted"); foot.setAlignment(Qt.AlignRight); rl.addWidget(foot)
        self.results.hide(); lay.addWidget(self.results, 1)
        return page

    def build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in (("Upload ECG audio...", "Ctrl+O", self.browse), ("Toggle light/dark", "Ctrl+T", self.toggle_theme), ("Quit", "Ctrl+Q", self.close)):
            a = QAction(text, self); a.setShortcut(key); a.triggered.connect(fn); m.addAction(a)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"; self.apply_theme()

    def apply_theme(self):
        t = self.t
        QApplication.instance().setStyleSheet(stylesheet(t))
        self.theme_btn.setIcon(svg_icon("moon" if self.theme == "light" else "sun", t["text2"], 18))
        self.open_btn.setIcon(svg_icon("folder", t["accent_ink"], 18)); self.info_icon.setPixmap(svg_pixmap("info", t["muted"], 15))
        self.err_icon.setPixmap(svg_pixmap("alert", t["abn"], 20)); self.mon_icon.setPixmap(svg_pixmap("heart", t["abn"], 13))
        self.part_prev.setIcon(svg_icon("left", t["text"], 15)); self.part_next.setIcon(svg_icon("right", t["text"], 15))
        self.details_btn.setIcon(svg_icon("list", t["text2"], 16)); self.export_btn.setIcon(svg_icon("download", t["text2"], 16)); self.audio_btn.setIcon(svg_icon("wave", t["text2"], 16))
        self.pulse.color, self.pulse.dim = QColor(t["accent"]), QColor(t["border"])
        self.monitor.set_grid(t["grid_major"] if self.theme == "dark" else t["border"]); self.monitor.set(color=t["normal"])
        self.hr_heart.set(color=t["abn"]); self.sheet.set_colors(t); self.render_samples(); self.update_model_sum()
        if self.res:
            self.render()

    def load_model(self):
        try:
            from .infer import Predictor
            self.bridge.ready.emit(Predictor())
        except Exception as e:  # noqa: BLE001
            self.bridge.failed.emit(f"Could not load the trained model: {e}")

    def on_ready(self, pred):
        self.pred = pred
        table, wav = resolve("samples/samples.csv"), resolve("data/raw/ptbxl/wav")
        full = resolve("data/processed/dataset.csv")
        folder = wav if full.exists() and wav.exists() else resolve("samples")
        src = full if folder == wav else table
        if src.exists():
            with open(src, newline="") as f:
                for r in csv.DictReader(f):
                    if r["fold"] == "10" and r["used"] == "True" and (folder / r["file"]).exists():
                        self.samples.append((str(folder / r["file"]), tuple(c for c in CLASSES if r[c] == "1")))
        if not self.samples:
            self.rec_note.setText("Test recordings not found. You can still upload your own files.")
        self.update_model_sum(); self.render_samples()

    def update_model_sum(self):
        if not self.pred:
            return
        t, i = self.t, self.pred.info; m = i["test"]["macro"]
        self.model_sum.setText(f"<b>Hybrid CNN-LSTM</b><br><span style='color:{t['text2']}'>test macro AUC {m['roc_auc']:.3f} · F1 {m['f1']:.3f}</span><br>"
                               f"<span style='color:{t['muted']}'>{i['test']['n']:,} unseen recordings · {i['params'] / 1000:.0f}k params</span>")

    def set_filter(self, n):
        self.filter = n; self.render_samples()

    def render_samples(self):
        pos = self.rec_list.verticalScrollBar().value(); self.rec_list.clear(); self.delegate.colors = self.t
        items = [s for s in self.samples if self.filter == "All" or self.filter in s[1]]
        self.count_lbl.setText(f"{len(items):,}" if items else "")
        for path, labels in items:
            it = QListWidgetItem(); it.setData(Qt.UserRole, (path, labels)); self.rec_list.addItem(it)
            if path == self.current:
                it.setSelected(True)
        self.rec_list.verticalScrollBar().setValue(pos)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self.drop_hint.setText("Release to analyze...")

    def dragLeaveEvent(self, _):
        self.drop_hint.setText("or drop a file anywhere in the window")

    def dropEvent(self, e):
        self.dragLeaveEvent(None)
        f = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if f:
            self.analyze(f[0])

    def browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Upload ECG audio", str(Path.home()), "ECG audio (" + " ".join("*" + e for e in sorted(EXTS)) + ");;All files (*)")
        if f:
            self.analyze(f)

    def analyze(self, path, truth=None):
        if self.pred is None or self.busy:
            return
        self.busy, self.err_shown = True, False; self.err.hide(); self.current, self.truth = path, truth

        def job():
            try:
                self.bridge.done.emit(self.pred.analyze(path))
            except InputError as e:
                self.bridge.failed.emit(str(e))
            except Exception as e:  # noqa: BLE001
                self.bridge.failed.emit(f"Unexpected error: {e}")
        threading.Thread(target=job, daemon=True).start()

    def on_error(self, msg):
        self.busy = False; self.results.hide(); self.res = None; self.empty.show()
        self.err_text.setText(f"<b>This file can't be analyzed.</b><br>{msg}"); self.err.show()

    def on_result(self, r):
        self.busy, self.res = False, r; self.empty.hide(); self.results.show(); self.show_part(0); self.render()
        for i in range(self.rec_list.count()):
            it = self.rec_list.item(i); it.setSelected(it.data(Qt.UserRole)[0] == self.current)

    def show_part(self, k):
        r = self.res; n = len(r["parts"]); self.part = k = max(0, min(k, n - 1)); s = r["parts"][k] / r["fs"]
        for w in (self.part_prev, self.part_lbl, self.part_next):
            w.setVisible(n > 1)
        self.part_prev.setEnabled(k > 0); self.part_next.setEnabled(k < n - 1)
        hit = [c for c, p in zip(CLASSES, r["part_prob"][k]) if p >= r["thr"][c]]
        self.part_lbl.setText(f"Part {k + 1} of {n}, {s:.0f} to {s + r['x'][k].shape[1] / r['fs']:.0f} s: {', '.join(hit) or 'none'}")
        self.sheet.set_data(r["x"][k])

    def render(self):
        r, t = self.res, self.t; pr = r["predicted"]
        self.rec_title.setText(r["name"])
        self.rec_meta.setText(f"{'PTB-XL test recording' if self.truth else 'Uploaded file'}  ·  {r['source']}  ·  {r['duration']:g} s  ·  {r['fs']} Hz  ·  12 leads")
        if pr:
            self.verdict.setText("<br>".join(f"<span style='color:{t['normal'] if c == 'NORM' else t['abn']}'>{k}.&nbsp; {NAMES[c]}</span>" for k, c in enumerate(pr, 1)))
            n = len(pr); txt = f"{n} class{'es' if n > 1 else ''} reached the decision threshold."
            if "NORM" in pr and n > 1:
                txt += " Normal appears together with an abnormal class, so treat this result as uncertain."
        else:
            self.verdict.setText(f"<span style='color:{t['warn']}'>No class identified</span>")
            txt = f"No class reached its threshold. Highest: {NAMES[r['top']]} at {r['prob'][r['top']] * 100:.1f}%."
        self.summary.setText(txt); self.truth_lbl.setVisible(bool(self.truth))
        if self.truth:
            same, some = set(self.truth) == set(pr), set(self.truth) & set(pr)
            word, col = ("exact match", t["normal"]) if same else (("partial match", t["warn"]) if some else ("no match", t["abn"]))
            self.truth_lbl.setText(f"PTB-XL reference: <b>{', '.join(self.truth)}</b> <span style='color:{col}'>({word})</span>")
        if len(r["parts"]) > 1:
            self.summary.setText(self.summary.text() + f" Average of {len(r['parts'])} parts of 10 s.")
            some = [f"{c} in part " + ", ".join(str(i + 1) for i, p in enumerate(r["part_prob"]) if p[j] >= r["thr"][c])
                    for j, c in enumerate(CLASSES) if c not in pr and (r["part_prob"][:, j] >= r["thr"][c]).any()]
            if some:
                self.summary.setText(self.summary.text() + " Only in some parts: " + "; ".join(some) + ".")
        if r["duration"] < 10:
            self.summary.setText(self.summary.text() + " Shorter than 10 s, so this result is less reliable.")
        if r["lossy"]:
            self.summary.setText(self.summary.text() + " This file is compressed, so the ECG was restored from it with small changes.")
        self.bars.set(r, t)
        for (ic, tx), (n, ok, v) in zip(self.check_rows, r["checks"]):
            ic.setPixmap(svg_pixmap("check" if ok else "x", t["normal"] if ok else t["abn"], 15)); tx.setText(n); tx.setToolTip(v)
        bpm = r["bpm"]; self.hr_lbl.setText(f"{bpm:.0f} bpm (estimated)" if bpm else "Heart rate not found")
        col = t["normal"] if pr == ["NORM"] else t["abn"]
        self.hr_heart.set(bpm=bpm, color=col); self.logo_heart.set(bpm=bpm); self.monitor.set(bpm=bpm, color=col)
        self.mon_icon.setPixmap(svg_pixmap("heart", col, 13)); self.mon_lbl.setText(f"{bpm:.0f} bpm" if bpm else "- bpm")

    def dialog(self, title, heading, intro, rows, note):
        d = QDialog(self); d.setWindowTitle(title); d.setMinimumWidth(580)
        lay = QVBoxLayout(d); lay.setContentsMargins(24, 20, 24, 20); lay.setSpacing(12); lay.addWidget(QLabel(heading, objectName="h2"))
        if intro:
            lay.addWidget(wrap(QLabel(intro)))
        g = QGridLayout(); g.setVerticalSpacing(7); g.setHorizontalSpacing(16); g.setColumnStretch(1, 1)
        for i, (k, v) in enumerate(rows):
            g.addWidget(QLabel(k, objectName="kvk"), i, 0, Qt.AlignTop); g.addWidget(wrap(QLabel(v)), i, 1)
        lay.addLayout(g)
        if note:
            lay.addWidget(wrap(QLabel(note, objectName="muted")))
        ok = QPushButton("Close", objectName="primary"); ok.clicked.connect(d.accept); lay.addWidget(ok, 0, Qt.AlignRight); d.exec()

    def show_about(self):
        if not self.pred:
            return
        i = self.pred.info; m = i["test"]; ci = m["macro_auc_ci95"]
        rows = [("Training data", f"PTB-XL, folds 1 to 8 ({i['train_records']:,} recordings)"),
                ("Evaluation", f"fold 10, {m['n']} recordings, no shared patients"),
                ("Macro ROC-AUC", f"{m['macro']['roc_auc']:.4f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})"),
                ("Macro PR-AUC / F1", f"{m['macro']['pr_auc']:.4f} / {m['macro']['f1']:.4f}"),
                ("Precision / recall", f"{m['macro']['precision']:.4f} / {m['macro']['recall']:.4f}")]
        rows += [(f"AUC {c}", f"{m['per_class'][c]['roc_auc']:.3f} ({m['per_class'][c]['support']} positive)") for c in CLASSES]
        rows += [("Parameters / size", f"{i['params']:,} / {i['size_kb']:,} KB"), ("Weights", f"runs/{i['run']}")]
        self.dialog("About the model", "Hybrid CNN-LSTM, five classes",
                    "Convolution blocks learn the waveform shape of the 12 leads; a bidirectional LSTM models how it changes over time; "
                    "five sigmoid outputs score each class independently.", rows,
                    "Thresholds were chosen on fold 9 only. Research demonstration, not for diagnosis.")

    def show_details(self):
        r = self.res
        if r:
            self.dialog("Technical details", r["name"], None, [("File", r["path"])] + [(n, v) for n, _, v in r["checks"]]
                        + [("Preprocessing", "each lead centered, fixed scale (true voltage kept)")]
                        + [(f"Threshold {c}", f"{r['thr'][c] * 100:.1f}%") for c in CLASSES]
                        + [("Inference time", f"{r['infer_ms']:.1f} ms"), ("Total time", f"{r['total_ms']:.1f} ms")], None)

    def save_audio(self):
        if not self.res:
            return
        names = {"mp4": "MP4", "mp3": "MP3", "m4a": "M4A", "flac": "FLAC", "wav": "WAV, one channel", "mov": "MOV", "aac": "AAC", "aiff": "AIFF", "wma": "WMA"}
        path, _ = QFileDialog.getSaveFileName(self, "Save as audio", str(Path.home() / "Desktop" / f"{self.res['name']}.mp4"),
                                              ";;".join(f"{v} (*.{k})" for k, v in names.items()))
        if not path:
            return
        try:
            save(self.res["raw"], path); msg = f"Saved to {path}"
        except Exception as e:  # noqa: BLE001
            msg = f"Could not save: {e}"
        self.statusBar().show(); self.statusBar().showMessage(msg, 6000); QTimer.singleShot(6000, self.statusBar().hide)

    def export_pdf(self, path=None):
        r = self.res
        if not r:
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Export PDF report", str(Path.home() / "Desktop" / f"{r['name']}_ECG_report.pdf"), "PDF (*.pdf)")
            if not path:
                return
        dark = self.theme == "dark"
        if dark:
            self.toggle_theme()
        QApplication.processEvents(); fig, t = self.sheet.grab(), THEMES["light"]
        w = QPdfWriter(path); w.setPageSize(QPageSize(QPageSize.A4)); w.setResolution(200); w.setPageMargins(QMarginsF(16, 14, 16, 14), QPageLayout.Millimeter)
        W, H = w.width(), w.height(); u = W / 100; p = QPainter(w); y = 0.0

        def text(s, size, color, bold=False, gap=0.8):
            nonlocal y
            f = QFont(self.font()); f.setPixelSize(int(size * u)); f.setBold(bold); p.setFont(f); p.setPen(QColor(color))
            b = p.boundingRect(QRectF(0, y, W, H), int(Qt.AlignLeft | Qt.TextWordWrap), s)
            p.drawText(QRectF(0, y, W, b.height()), int(Qt.AlignLeft | Qt.TextWordWrap), s); y += b.height() + gap * u
        text(f"ECG Analyzer  ·  {datetime.now():%Y-%m-%d %H:%M}", 1.7, t["muted"], gap=1.5)
        text(r["name"], 4.2, t["text"], True, 0.4)
        text(self.rec_meta.text(), 1.8, t["text2"], gap=2)
        text("Identified classes", 1.8, t["muted"], True, 0.4)
        text("\n".join(f"{k}. {NAMES[c]}" for k, c in enumerate(r["predicted"], 1)) or "No class reached its threshold", 3.0, t["text"], True, 1.6)
        for i, c in enumerate(CLASSES):
            rh = 3.4 * u
            if i % 2 == 0:
                p.fillRect(QRectF(0, y, W, rh), QColor(t["surface2"]))
            f = QFont(self.font()); f.setPixelSize(int(1.9 * u)); f.setBold(c in r["predicted"]); p.setFont(f); p.setPen(QColor(t["text"]))
            for x, s in ((1.5, c), (12, NAMES[c]), (48, f"{r['prob'][c] * 100:.1f}%"), (62, f"threshold {r['thr'][c] * 100:.1f}%"), (84, "identified" if c in r["predicted"] else "")):
                p.drawText(QRectF(x * u, y, 40 * u, rh), int(Qt.AlignLeft | Qt.AlignVCenter), s)
            y += rh
        y += 3 * u; text("12-lead ECG", 2.2, t["text"], True)
        fh = W * fig.height() / fig.width(); p.drawPixmap(QRectF(0, y, W, fh), fig, QRectF(0, 0, fig.width(), fig.height())); y += fh + 2 * u
        text("Hybrid CNN-LSTM trained on PTB-XL. Research demonstration only, not a medical device.", 1.5, t["muted"]); p.end()
        if dark:
            self.toggle_theme()
        self.statusBar().show(); self.statusBar().showMessage(f"Report saved to {path}", 6000); QTimer.singleShot(6000, self.statusBar().hide)

    def changeEvent(self, e):
        if e.type() == e.Type.WindowStateChange and CLOCK:
            CLOCK.set_active(not self.isMinimized())
        super().changeEvent(e)


def main():
    global CLOCK
    app = QApplication(sys.argv); app.setApplicationName("ECG Analyzer"); app.setStyle("Fusion")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    CLOCK = AnimClock()
    app.applicationStateChanged.connect(lambda s: CLOCK.set_active(s == Qt.ApplicationActive))
    args, dev = list(sys.argv), resolve("logs/.test_args")
    if dev.exists():
        args += dev.read_text().split("\n"); dev.unlink()
    opt = lambda k: args[args.index(k) + 1] if k in args else None
    w = MainWindow()
    if opt("--size"):
        w.resize(*map(int, opt("--size").split("x")))
    w.show()
    if "--light" in args:
        w.toggle_theme()
    if opt("--sample"):
        w.bridge.ready.connect(lambda _: next((w.analyze(*s) for s in w.samples if Path(s[0]).stem == opt("--sample")), None))
    if opt("--file"):
        w.bridge.ready.connect(lambda _: w.analyze(opt("--file")))
    if opt("--snapshot"):
        def shot():
            if opt("--export"):
                w.export_pdf(opt("--export"))
            w.grab().save(opt("--snapshot")); app.quit()
        sig = [w.bridge.done, w.bridge.failed] if opt("--sample") or opt("--file") else [w.bridge.ready]
        for s in sig:
            s.connect(lambda _: QTimer.singleShot(1200, shot))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
