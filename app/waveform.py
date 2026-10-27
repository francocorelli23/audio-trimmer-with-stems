"""Zoomable, rekordbox-style colored waveform displays.

WaveformView    main view: zoom, pan, selection handles, playhead
                (also used, without axis, for each stem lane)
OverviewView    whole-track strip with a box showing the zoomed-in part
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal

from .analysis import HOP, WaveformData

BACKGROUND = "#0b0b0d"
# How far the inner (RMS) part of the wave is lightened toward white.
CORE_LIGHTEN = 0.45
OUTER_BRIGHTNESS = 0.85
DIMMED_OPACITY = 0.18

SELECTION_BRUSH = (255, 255, 255, 26)
SELECTION_HOVER_BRUSH = (255, 255, 255, 40)
HANDLE_COLOR = "#ffc940"
PLAYHEAD_COLOR = "#ffffff"


def format_time(seconds: float, millis: bool = True) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60)
    if millis:
        return f"{int(minutes):02d}:{secs:06.3f}"
    return f"{int(minutes):02d}:{int(secs):02d}"


def parse_time(text: str) -> float:
    """Parse 'mm:ss.mmm', 'ss.mmm' or 'h:mm:ss' into seconds. Raises ValueError."""
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    if seconds < 0:
        raise ValueError(text)
    return seconds


class TimeAxis(pg.AxisItem):
    """Bottom axis that shows mm:ss instead of raw seconds."""

    def tickStrings(self, values, scale, spacing):
        return [format_time(v, millis=spacing < 1) for v in values]


class WaveViewBox(pg.ViewBox):
    """Horizontal-only zoom/pan. Click = seek, Shift+drag = new selection."""

    sigClicked = Signal(float)
    sigShiftDrag = Signal(float, float)

    def __init__(self):
        super().__init__(enableMenu=False)
        self.setMouseEnabled(x=True, y=False)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ev.accept()
            self.sigClicked.emit(self.mapSceneToView(ev.scenePos()).x())
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier:
            ev.accept()
            a = self.mapSceneToView(ev.buttonDownScenePos()).x()
            b = self.mapSceneToView(ev.scenePos()).x()
            self.sigShiftDrag.emit(min(a, b), max(a, b))
        else:
            super().mouseDragEvent(ev, axis)


class SelectionRegion(pg.LinearRegionItem):
    """Selection whose edges can be dragged; dragging or clicking inside it
    falls through to the view (pan / seek) instead of moving the region."""

    def mouseDragEvent(self, ev):
        ev.ignore()

    def mouseClickEvent(self, ev):
        ev.ignore()


def _reduce_columns(values: np.ndarray, starts: np.ndarray, stop: int, ufunc) -> np.ndarray:
    """Apply ufunc to values[starts[i]:starts[i+1]] for each column (last one ends at stop)."""
    first = starts[0]
    return ufunc.reduceat(values[first:stop], starts - first, axis=0)


class WaveformView(pg.PlotWidget):
    """Draws the waveform as an image, re-rendered for the visible range.

    Each pixel column gets its min/max (the outer shape), its RMS (the lighter
    core) and a color from the data's colorize function.
    """

    sigSeek = Signal(float)              # user clicked or dragged the playhead
    sigSelectionChanged = Signal(float, float)

    def __init__(self, parent=None, show_axis: bool = True, editable: bool = True):
        self.view_box = WaveViewBox()
        axes = {"bottom": TimeAxis("bottom")} if show_axis else {}
        super().__init__(parent, viewBox=self.view_box, axisItems=axes)
        self.setBackground(BACKGROUND)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.hideAxis("left")
        if show_axis:
            self.showGrid(x=True, y=False, alpha=0.12)
        else:
            self.hideAxis("bottom")
        self.setYRange(-1, 1, padding=0.04)

        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)

        handle_pen = pg.mkPen(HANDLE_COLOR, width=2)
        self.selection = SelectionRegion(
            brush=SELECTION_BRUSH, hoverBrush=SELECTION_HOVER_BRUSH,
            pen=handle_pen, hoverPen=pg.mkPen("#ffe08a", width=3), movable=editable,
        )
        self.selection.setZValue(5)
        self.selection.hide()
        self.addItem(self.selection, ignoreBounds=True)

        self.playhead = pg.InfiniteLine(
            pos=0, movable=editable,
            pen=pg.mkPen(PLAYHEAD_COLOR, width=1.5),
            hoverPen=pg.mkPen(HANDLE_COLOR, width=2.5),
        )
        self.playhead.setZValue(10)
        self.playhead.hide()
        self.addItem(self.playhead, ignoreBounds=True)

        self.data: WaveformData | None = None
        self.duration = 0.0
        self._quiet = False  # suppresses signals while we move things ourselves

        self.view_box.sigClicked.connect(self.sigSeek)
        self.view_box.sigShiftDrag.connect(self._on_shift_drag)
        self.selection.sigRegionChanged.connect(self._on_region_changed)
        self.playhead.sigDragged.connect(lambda: self.sigSeek.emit(self.playhead.value()))

        # Zooming fires many range changes in a row; render once per batch.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render)
        self.view_box.sigXRangeChanged.connect(self._render_timer.start)
        self.view_box.sigResized.connect(self._render_timer.start)

    # --- data --------------------------------------------------------------

    def set_data(self, data: WaveformData | None, duration: float | None = None) -> None:
        """Show new waveform data. Passing a duration also resets the view."""
        self.data = data
        if duration is not None:
            self.duration = duration
            self.view_box.setLimits(
                xMin=0, xMax=duration, minXRange=min(0.005, duration),
                yMin=-1.05, yMax=1.05,
            )
            self.selection.setBounds([0, duration])
            self.playhead.setBounds([0, duration])
            self.selection.show()
            self.playhead.show()
            self.zoom_to_fit()
        if data is None:
            self.image.clear()
        self._render_timer.start()

    def set_dimmed(self, dimmed: bool) -> None:
        self.image.setOpacity(DIMMED_OPACITY if dimmed else 1.0)

    # --- view, selection, playhead ------------------------------------------

    def zoom_to_fit(self) -> None:
        if self.duration > 0:
            self.setXRange(0, self.duration, padding=0)

    def visible_range(self) -> tuple[float, float]:
        return tuple(self.view_box.viewRange()[0])

    def set_selection(self, start: float, end: float) -> None:
        self._quiet = True
        self.selection.setRegion((start, end))
        self._quiet = False

    def set_playhead(self, seconds: float) -> None:
        self.playhead.setValue(seconds)

    def _on_shift_drag(self, start: float, end: float) -> None:
        start, end = max(start, 0.0), min(end, self.duration)
        if end > start:
            self.selection.setRegion((start, end))

    def _on_region_changed(self) -> None:
        if not self._quiet:
            start, end = self.selection.getRegion()
            self.sigSelectionChanged.emit(start, end)

    # --- rendering ----------------------------------------------------------

    def _render(self) -> None:
        data = self.data
        if data is None:
            return
        sr = data.sample_rate
        n = data.mono.size

        (view_x0, view_x1), _ = self.view_box.viewRange()
        x0, x1 = max(view_x0, 0.0), min(view_x1, n / sr)
        if x1 <= x0:
            return
        dpr = self.devicePixelRatioF()
        width = int(self.view_box.width() * dpr * (x1 - x0) / (view_x1 - view_x0))
        height = int(self.view_box.height() * dpr)
        if width < 1 or height < 2:
            return

        mins, maxs, rms, rgb = self._columns(data, x0 * sr, x1 * sr, width)
        self.image.setImage(self._paint(mins, maxs, rms, rgb, height), autoLevels=False)
        self.image.setRect(QRectF(x0, -1, x1 - x0, 2))

    @staticmethod
    def _columns(data: WaveformData, s0: float, s1: float, width: int):
        """Per-pixel-column min, max, RMS and color for samples s0..s1."""
        n = data.mono.size
        edges = np.linspace(s0, s1, width + 1)
        starts = np.clip(edges[:-1].astype(np.int64), 0, n - 1)
        stop = max(min(int(np.ceil(edges[-1])), n), starts[-1] + 1)

        if (s1 - s0) / width >= HOP:
            # Zoomed out: combine precomputed blocks.
            b_starts = starts // HOP
            b_stop = max(min(-(-stop // HOP), data.blocks), b_starts[-1] + 1)
            counts = np.maximum(np.diff(np.append(b_starts, b_stop)), 1)
            mins = _reduce_columns(data.block_min, b_starts, b_stop, np.minimum)
            maxs = _reduce_columns(data.block_max, b_starts, b_stop, np.maximum)
            power = _reduce_columns(data.block_power, b_starts, b_stop, np.add) / counts
            energy = _reduce_columns(data.block_energy, b_starts, b_stop, np.add)
        else:
            # Zoomed in: use raw samples so the true wave shape shows.
            counts = np.maximum(np.diff(np.append(starts, stop)), 1)
            segment = data.mono[starts[0]:stop]
            local = starts - starts[0]
            mins = np.minimum.reduceat(segment, local)
            maxs = np.maximum.reduceat(segment, local)
            power = np.add.reduceat(segment * segment, local) / counts
            centers = ((edges[:-1] + edges[1:]) / 2 // HOP).astype(np.int64)
            energy = data.block_energy[np.clip(centers, 0, data.blocks - 1)]

        return mins, maxs, np.sqrt(power), np.clip(data.colorize(energy), 0, 1)

    @staticmethod
    def _paint(mins, maxs, rms, rgb, height: int) -> np.ndarray:
        """Build an RGBA image (height x columns) of the wave."""
        # Row 0 is drawn at the bottom (y = -1).
        y = (-1 + (np.arange(height) + 0.5) * (2 / height))[:, None]
        half_px = 1 / height  # keeps quiet parts visible as a thin line
        inside = (y >= mins - half_px) & (y <= maxs + half_px)
        core = inside & (np.abs(y) <= rms)

        outer = (rgb * OUTER_BRIGHTNESS * 255).astype(np.uint8)
        light = ((rgb + (1 - rgb) * CORE_LIGHTEN) * 255).astype(np.uint8)

        img = np.zeros((height, rgb.shape[0], 4), dtype=np.uint8)
        img[..., :3] = np.where(core[..., None], light, outer)
        img[..., 3] = inside * 255
        return img


class OverviewView(WaveformView):
    """Whole track at a glance. The bright box marks the part shown in the
    main view; drag it (or its edges) to scroll/zoom, click to jump."""

    sigWindowChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent, show_axis=False, editable=False)
        self.view_box.setMouseEnabled(x=False, y=False)
        self.selection.setBrush(pg.mkBrush(255, 201, 64, 30))
        self.selection.lines[0].setPen(pg.mkPen(HANDLE_COLOR, width=1))
        self.selection.lines[1].setPen(pg.mkPen(HANDLE_COLOR, width=1))

        self.window = pg.LinearRegionItem(
            brush=(255, 255, 255, 22), hoverBrush=(255, 255, 255, 40),
            pen=pg.mkPen("#9aa0a6", width=1), hoverPen=pg.mkPen("#ffffff", width=2),
        )
        self.window.setZValue(8)
        self.window.hide()
        self.addItem(self.window, ignoreBounds=True)
        self.window.sigRegionChanged.connect(self._on_window_changed)

    def set_data(self, data, duration=None):
        super().set_data(data, duration)
        if duration is not None:
            self.window.setBounds([0, duration])
            self.window.show()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "duration", 0):  # also called by the base class before __init__ finishes
            self.zoom_to_fit()

    def set_window(self, start: float, end: float) -> None:
        self._quiet = True
        self.window.setRegion((start, end))
        self._quiet = False

    def _on_window_changed(self) -> None:
        if not self._quiet:
            self.sigWindowChanged.emit(*self.window.getRegion())
