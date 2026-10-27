"""The main application window."""
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QSizePolicy, QStackedWidget, QStyle, QToolButton, QVBoxLayout, QWidget,
)

from . import analysis, audio_io, stems
from .player import Player
from .waveform import OverviewView, WaveformView, format_time, parse_time

HEADER_WIDTH = 168
OVERVIEW_HEIGHT = 54
LANE_HEIGHT = 62
COLOR_MODES = ("Frequency colors", "Stem colors")

STYLE = """
QMainWindow, QWidget#central { background: #141416; }
QToolBar { background: #1b1c1f; border: none; border-bottom: 1px solid #2a2b30; padding: 4px 6px; spacing: 2px; }
QToolBar QToolButton { color: #e6e6e6; padding: 5px 9px; border-radius: 5px; }
QToolBar QToolButton:hover { background: #2a2c31; }
QToolBar QToolButton:checked { background: #33363d; color: #ffc940; }
QToolBar QToolButton:disabled { color: #5d6066; }
QLabel#trackTitle { color: #f2f2f2; font-size: 13px; font-weight: 600; padding-right: 8px; }
QLabel#trackInfo { color: #8a8f98; padding-right: 10px; }
QLabel#sectionLabel { color: #8a8f98; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
QFrame#rowHeader { background: #18191c; }
QFrame#trimBar { background: #1b1c1f; border-top: 1px solid #2a2b30; }
QFrame#trimBar QLabel { color: #a0a4ab; }
QLabel#timeDisplay { font-family: Consolas, monospace; font-size: 22px; color: #ffffff; }
QLabel#lengthDisplay { font-family: Consolas, monospace; color: #ffc940; font-size: 13px; }
QLineEdit, QDoubleSpinBox, QComboBox {
    background: #0f1012; color: #f0f0f0; border: 1px solid #33363d; border-radius: 4px; padding: 3px 6px;
}
QLineEdit { font-family: Consolas, monospace; }
QLineEdit:focus, QDoubleSpinBox:focus { border-color: #ffc940; }
QToolButton#mute, QToolButton#solo {
    background: #24262b; color: #c9ccd1; border: 1px solid #33363d; border-radius: 4px; font-weight: 700;
}
QToolButton#mute:checked { background: #e0525b; color: #ffffff; border-color: #e0525b; }
QToolButton#solo:checked { background: #f2c94c; color: #141416; border-color: #f2c94c; }
QStatusBar { background: #1b1c1f; color: #a0a4ab; }
QProgressBar { background: #0f1012; border: 1px solid #33363d; border-radius: 3px; height: 10px; }
QProgressBar::chunk { background: #3d9bff; border-radius: 2px; }
QLabel#dropHint { color: #8a8f98; font-size: 16px; }
"""


def app_icon() -> QIcon:
    """Small colored-bars icon, drawn in code so there is no image file to ship."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    bars = [(18, "#ff3d6e"), (36, "#ff9f1c"), (54, "#36e07f"), (30, "#3d9bff"), (44, "#36e07f"), (22, "#ff9f1c")]
    for i, (h, color) in enumerate(bars):
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(4 + i * 10, 32 - h / 2, 7, h, 3, 3)
    painter.end()
    return QIcon(pixmap)


class Task(QObject):
    """Runs a function on a background thread and reports back on the UI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e) or type(e).__name__)


def _row_header(*widgets, border_color: str | None = None) -> QFrame:
    frame = QFrame()
    frame.setObjectName("rowHeader")
    frame.setFixedWidth(HEADER_WIDTH)
    if border_color:
        frame.setStyleSheet(f"QFrame#rowHeader {{ border-left: 5px solid {border_color}; }}")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 4, 8, 4)
    for w in widgets:
        if w is None:
            layout.addStretch(1)
        else:
            layout.addWidget(w)
    return frame


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


class StemRow:
    """Header (color, name, mute/solo) plus waveform lane for one stem."""

    def __init__(self, info: stems.StemInfo, on_change):
        self.info = info
        name = QLabel(info.label)
        name.setStyleSheet(f"color: {info.color}; font-weight: 600;")
        self.mute = self._button("M", "mute", "Mute this stem")
        self.solo = self._button("S", "solo", "Solo: hear only soloed stems")
        self.mute.toggled.connect(on_change)
        self.solo.toggled.connect(on_change)
        self.header = _row_header(name, None, self.mute, self.solo, border_color=info.color)
        self.header.setFixedHeight(LANE_HEIGHT)

        self.lane = WaveformView(show_axis=False, editable=False)
        self.lane.setFixedHeight(LANE_HEIGHT)
        self.rgb = stems.hex_to_rgb(info.color)

    @staticmethod
    def _button(text: str, name: str, tip: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setObjectName(name)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.NoFocus)
        button.setFixedSize(26, 24)
        return button

    def set_visible(self, visible: bool) -> None:
        self.header.setVisible(visible)
        self.lane.setVisible(visible)

    def reset(self) -> None:
        for b in (self.mute, self.solo):
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Trimmer")
        self.setWindowIcon(app_icon())
        self.resize(1280, 760)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        self.settings = QSettings("AudioTrimmer", "AudioTrimmer")

        self.audio: audio_io.Audio | None = None
        self.base_data: analysis.WaveformData | None = None  # frequency colors of the original
        self.stems: dict[str, np.ndarray] | None = None
        self.stem_monos: dict[str, np.ndarray] = {}
        self.stem_powers: dict[str, np.ndarray] = {}
        self.sel_start = 0.0
        self.sel_end = 0.0
        self.playhead = 0.0
        self.player = Player()
        self._play_origin = 0.0
        self._playing_selection = False
        self._was_playing = False
        self._busy = False
        self._stem_worker: stems.StemWorker | None = None
        self._tasks: set[Task] = set()

        self._build_actions()
        self._build_toolbar()
        self._build_body()
        self._build_status_bar()
        self._update_enabled()

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------ UI

    def _action(self, text, slot, shortcut=None, icon=None, tip=None, checkable=False) -> QAction:
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(f"{tip or text}  ({QKeySequence(shortcut).toString()})" if shortcut else (tip or text))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)  # shortcuts work even if the action is not on the toolbar
        return action

    def _build_actions(self) -> None:
        S = QStyle.StandardPixmap
        self.open_action = self._action("Open", self.open_file, QKeySequence.Open, S.SP_DialogOpenButton, "Open an audio file")
        self.play_action = self._action("Play", self.toggle_play, "Space", S.SP_MediaPlay, "Play / pause from the playhead")
        self.play_sel_action = self._action("Play selection", self.play_selection, "Shift+Space", S.SP_MediaSeekForward)
        self.stop_action = self._action("Stop", self.stop, "Esc", S.SP_MediaStop, "Stop and return to where playback started")
        self.loop_action = self._action("Loop", self._on_loop_toggled, "L", S.SP_BrowserReload, "Loop the selection", checkable=True)
        self.fit_action = self._action("Fit", lambda: self.waveform.zoom_to_fit(), "Ctrl+0", tip="Zoom to the whole track")
        self.zoom_sel_action = self._action("Zoom to selection", self.zoom_to_selection, "Ctrl+E")
        self.stems_action = self._action("Separate stems", self.separate_stems, "Ctrl+T", tip="Split into drums, bass, instruments and vocals")
        self.export_action = self._action("Export", self.export_selection, QKeySequence.Save, S.SP_DialogSaveButton, "Export the selection")
        self.export_stems_action = self._action("Export stems", self.export_stems, "Ctrl+Shift+S", tip="Export each audible stem of the selection")
        self.mark_in_action = self._action("Set start", lambda: self._mark(True), "I", tip="Set selection start at the playhead")
        self.mark_out_action = self._action("Set end", lambda: self._mark(False), "O", tip="Set selection end at the playhead")
        self.select_all_action = self._action("Select all", self.select_all, QKeySequence.SelectAll)
        self.to_start_action = self._action("Go to selection start", lambda: self.seek(self.sel_start), "Home")
        self.to_end_action = self._action("Go to selection end", lambda: self.seek(self.sel_end), "End")
        self.loop_action.setChecked(self.settings.value("loop", False, type=bool))

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for item in (
            self.open_action, None,
            self.play_action, self.play_sel_action, self.stop_action, self.loop_action, None,
            self.fit_action, self.zoom_sel_action, None,
            self.stems_action, None,
            self.export_action, self.export_stems_action,
        ):
            if item is None:
                toolbar.addSeparator()
            else:
                toolbar.addAction(item)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self.title_label = QLabel()
        self.title_label.setObjectName("trackTitle")
        self.info_label = QLabel()
        self.info_label.setObjectName("trackInfo")
        toolbar.addWidget(self.title_label)
        toolbar.addWidget(self.info_label)

    def _build_body(self) -> None:
        # Views: overview, main waveform, then one lane per stem. A grid keeps
        # the headers in one column so every waveform lines up in time.
        self.overview = OverviewView()
        self.overview.setFixedHeight(OVERVIEW_HEIGHT)
        self.waveform = WaveformView()
        self.waveform.setMinimumHeight(180)

        self.color_combo = QComboBox()
        self.color_combo.addItems(COLOR_MODES)
        self.color_combo.setFocusPolicy(Qt.NoFocus)
        self.color_combo.setToolTip("Color the waveform by frequency (rekordbox style) or by stem")
        self.color_combo.currentIndexChanged.connect(self._refresh_mix)

        mix_header = QFrame()
        mix_header.setObjectName("rowHeader")
        mix_header.setFixedWidth(HEADER_WIDTH)
        mix_layout = QVBoxLayout(mix_header)
        mix_layout.setContentsMargins(10, 8, 8, 8)
        mix_layout.addWidget(_section_label("WAVEFORM"))
        mix_layout.addWidget(self.color_combo)
        legend = QLabel(
            '<span style="color:#ff4040">■</span> lows  '
            '<span style="color:#40ff40">■</span> mids  '
            '<span style="color:#4080ff">■</span> highs'
        )
        legend.setStyleSheet("color: #8a8f98; font-size: 11px;")
        self.legend = legend
        mix_layout.addWidget(legend)
        mix_layout.addStretch(1)
        mix_layout.addWidget(_section_label("SHIFT+DRAG TO SELECT"))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(2)
        grid.addWidget(_row_header(_section_label("OVERVIEW")), 0, 0)
        grid.addWidget(self.overview, 0, 1)
        grid.addWidget(mix_header, 1, 0)
        grid.addWidget(self.waveform, 1, 1)
        grid.setRowStretch(1, 1)

        self.stem_rows: list[StemRow] = []
        for i, info in enumerate(stems.STEMS):
            row = StemRow(info, self._refresh_mix)
            row.lane.setXLink(self.waveform)
            row.lane.sigSeek.connect(self.seek)
            row.lane.sigSelectionChanged.connect(self.set_selection)
            row.set_visible(False)
            grid.addWidget(row.header, 2 + i, 0)
            grid.addWidget(row.lane, 2 + i, 1)
            self.stem_rows.append(row)

        self.waveform.sigSeek.connect(self.seek)
        self.waveform.sigSelectionChanged.connect(self.set_selection)
        self.waveform.view_box.sigXRangeChanged.connect(self._on_view_changed)
        self.overview.sigSeek.connect(lambda t: self.seek(t, center=True))
        self.overview.sigSelectionChanged.connect(self.set_selection)
        self.overview.sigWindowChanged.connect(lambda a, b: self.waveform.setXRange(a, b, padding=0))

        editor = QWidget()
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(grid, 1)
        layout.addWidget(self._build_trim_bar())

        hint = QLabel("Drop an audio file here\nor press Ctrl+O\n\nMP3 · M4A · WAV · FLAC")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignCenter)

        self.stack = QStackedWidget()
        self.stack.setObjectName("central")
        self.stack.addWidget(hint)
        self.stack.addWidget(editor)
        self.setCentralWidget(self.stack)

    def _build_trim_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("trimBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(8)

        self.start_edit = QLineEdit()
        self.end_edit = QLineEdit()
        for edit, is_start in ((self.start_edit, True), (self.end_edit, False)):
            edit.setFixedWidth(96)
            edit.setToolTip("mm:ss.mmm - press Enter to apply")
            edit.editingFinished.connect(lambda e=edit, s=is_start: self._on_time_edited(e, s))
        self.length_label = QLabel()
        self.length_label.setObjectName("lengthDisplay")

        self.fade_in = self._fade_spin("fade_in")
        self.fade_out = self._fade_spin("fade_out")

        self.time_label = QLabel(format_time(0))
        self.time_label.setObjectName("timeDisplay")

        layout.addWidget(QLabel("Start"))
        layout.addWidget(self.start_edit)
        layout.addWidget(QLabel("End"))
        layout.addWidget(self.end_edit)
        layout.addWidget(QLabel("Length"))
        layout.addWidget(self.length_label)
        layout.addSpacing(24)
        layout.addWidget(QLabel("Fade in"))
        layout.addWidget(self.fade_in)
        layout.addWidget(QLabel("Fade out"))
        layout.addWidget(self.fade_out)
        layout.addStretch(1)
        layout.addWidget(self.time_label)
        return bar

    def _fade_spin(self, key: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 30)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setSuffix(" s")
        spin.setFixedWidth(80)
        spin.setToolTip("Applied when exporting")
        spin.setValue(self.settings.value(key, 0.0, type=float))
        spin.valueChanged.connect(lambda v: self.settings.setValue(key, v))
        return spin

    def _build_status_bar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready")

    def _update_enabled(self) -> None:
        has_audio = self.audio is not None and not self._busy
        for action in (
            self.play_action, self.play_sel_action, self.stop_action, self.fit_action,
            self.zoom_sel_action, self.export_action, self.mark_in_action, self.mark_out_action,
            self.select_all_action, self.to_start_action, self.to_end_action,
        ):
            action.setEnabled(has_audio)
        self.open_action.setEnabled(not self._busy)
        self.stems_action.setEnabled(has_audio and self.stems is None and self._stem_worker is None)
        self.export_stems_action.setEnabled(has_audio and self.stems is not None)
        self.color_combo.setEnabled(self.stems is not None)

    def _set_busy(self, message: str | None, fraction: float | None = None) -> None:
        """Show a message with a progress bar (fraction < 0 = busy), or clear it."""
        if message is None:
            self.progress.hide()
            return
        self.statusBar().showMessage(message)
        if fraction is None:
            self.progress.hide()
        elif fraction < 0:
            self.progress.setRange(0, 0)
            self.progress.show()
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
            self.progress.show()

    def _run_task(self, fn, on_done, on_failed) -> None:
        task = Task(fn)
        self._tasks.add(task)
        task.done.connect(on_done)
        task.failed.connect(on_failed)
        task.done.connect(lambda _=None, t=task: self._tasks.discard(t))
        task.failed.connect(lambda _=None, t=task: self._tasks.discard(t))
        task.start()

    # ------------------------------------------------------------- loading

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio file", self.settings.value("last_dir", str(Path.home() / "Music")),
            audio_io.OPEN_FILTER,
        )
        if path:
            self.load(path)

    def load(self, path) -> None:
        if self._busy:
            return
        path = Path(path)
        self.stop()
        self._busy = True
        self._update_enabled()
        self._set_busy(f"Opening {path.name}…", -1)

        def job():
            audio = audio_io.load(path)
            data = analysis.analyze(audio.samples, audio.sample_rate)
            return audio, data, stems.load_cached(audio)

        self._run_task(job, self._on_loaded, self._on_load_failed)

    def _on_loaded(self, result) -> None:
        audio, data, cached_stems = result
        self._busy = False
        self.player.stop()
        self.audio, self.base_data = audio, data
        self.stems = None
        self._stem_worker = None
        self.settings.setValue("last_dir", str(audio.path.parent))

        for row in self.stem_rows:
            row.set_visible(False)
            row.reset()
            # Lanes share the main view's zoom, so their limits must match the new file.
            row.lane.set_data(None, audio.duration)
        self.color_combo.blockSignals(True)
        self.color_combo.setCurrentIndex(0)
        self.color_combo.blockSignals(False)
        self.legend.show()

        self.waveform.set_data(data, audio.duration)
        self.overview.set_data(data, audio.duration)
        self.player.set_buffer(audio.samples, audio.sample_rate)
        self.stack.setCurrentIndex(1)
        self.set_selection(0.0, audio.duration)
        self.seek(0.0)

        self.setWindowTitle(f"{audio.path.name} – Audio Trimmer")
        self.title_label.setText(audio.path.name)
        self.info_label.setText(
            f"{format_time(audio.duration, millis=False)}  ·  {audio.sample_rate / 1000:g} kHz  ·  "
            f"{'stereo' if audio.channels == 2 else f'{audio.channels} ch'}"
        )
        self._set_busy(None)
        if cached_stems is not None:
            self._set_stems(cached_stems)
            self.statusBar().showMessage(f"Opened {audio.path.name} (stems loaded from cache)")
        else:
            self.statusBar().showMessage(f"Opened {audio.path.name}")
        self._update_enabled()

    def _on_load_failed(self, message: str) -> None:
        self._busy = False
        self._set_busy(None)
        self.statusBar().showMessage("Could not open file")
        self._update_enabled()
        QMessageBox.critical(self, "Could not open file", message)

    def dragEnterEvent(self, event) -> None:
        if self._dropped_file(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = self._dropped_file(event)
        if path is not None:
            self.load(path)

    @staticmethod
    def _dropped_file(event) -> Path | None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in audio_io.AUDIO_EXTS:
                return path
        return None

    # ----------------------------------------------------------- selection

    def set_selection(self, start: float, end: float) -> None:
        if self.audio is None:
            return
        start = float(max(0.0, min(start, self.audio.duration)))
        end = float(max(start, min(end, self.audio.duration)))
        self.sel_start, self.sel_end = start, end
        for view in self._views():
            view.set_selection(start, end)
        self.start_edit.setText(format_time(start))
        self.end_edit.setText(format_time(end))
        self.length_label.setText(format_time(end - start))
        if self.player.playing and self._playing_selection:
            sr = self.audio.sample_rate
            self.player.set_range(int(start * sr), int(end * sr))

    def select_all(self) -> None:
        if self.audio is not None:
            self.set_selection(0.0, self.audio.duration)

    def zoom_to_selection(self) -> None:
        if self.sel_end > self.sel_start:
            pad = (self.sel_end - self.sel_start) * 0.05
            self.waveform.setXRange(self.sel_start - pad, self.sel_end + pad, padding=0)

    def _mark(self, is_start: bool) -> None:
        if is_start:
            self.set_selection(self.playhead, max(self.sel_end, self.playhead))
        else:
            self.set_selection(min(self.sel_start, self.playhead), self.playhead)

    def _on_time_edited(self, edit: QLineEdit, is_start: bool) -> None:
        try:
            value = parse_time(edit.text())
        except ValueError:
            self.set_selection(self.sel_start, self.sel_end)  # restore the text
            return
        if is_start:
            self.set_selection(value, max(value, self.sel_end))
        else:
            self.set_selection(min(value, self.sel_start), value)

    def _views(self) -> list[WaveformView]:
        return [self.waveform, self.overview] + [r.lane for r in self.stem_rows]

    def _on_view_changed(self) -> None:
        self.overview.set_window(*self.waveform.visible_range())

    # ------------------------------------------------------------ playback

    def seek(self, seconds: float, center: bool = False) -> None:
        if self.audio is None:
            return
        seconds = float(max(0.0, min(seconds, self.audio.duration)))
        self._set_playhead(seconds)
        if center:
            x0, x1 = self.waveform.visible_range()
            half = (x1 - x0) / 2
            self.waveform.setXRange(seconds - half, seconds + half, padding=0)
        if self.player.playing:
            self._start_playback(self._playing_selection, seconds)

    def toggle_play(self) -> None:
        if self.player.playing:
            self.player.stop()
        elif self.audio is not None:
            # With loop on, Space plays the selection; otherwise to the end.
            loop = self.loop_action.isChecked()
            inside = self.sel_start <= self.playhead < self.sel_end
            position = self.playhead if (inside or not loop) else self.sel_start
            if not loop and self.playhead >= self.audio.duration - 0.01:
                position = 0.0  # at the very end: start over
            self._play_origin = position
            self._start_playback(loop, position)

    def play_selection(self) -> None:
        if self.audio is not None:
            self._play_origin = self.sel_start
            self._start_playback(True, self.sel_start)

    def stop(self) -> None:
        if self.player.playing:
            self.player.stop()
            self.seek(self._play_origin)

    def _start_playback(self, selection_only: bool, position: float) -> None:
        sr = self.audio.sample_rate
        self._playing_selection = selection_only
        if selection_only and self.sel_end > self.sel_start:
            self.player.play(
                int(self.sel_start * sr), int(self.sel_end * sr),
                loop=self.loop_action.isChecked(), position=int(position * sr),
            )
        else:
            self._playing_selection = False
            self.player.play(0, None, loop=False, position=int(position * sr))

    def _on_loop_toggled(self, checked: bool) -> None:
        self.settings.setValue("loop", checked)
        if self._playing_selection:
            self.player.set_loop(checked)

    def _set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        for view in self._views():
            view.set_playhead(seconds)
        total = self.audio.duration if self.audio else 0.0
        self.time_label.setText(f"{format_time(seconds)} / {format_time(total, millis=False)}")

    def _tick(self) -> None:
        playing = self.player.playing
        if playing:
            t = self.player.position / self.audio.sample_rate
            self._set_playhead(t)
            # Keep the playhead in view, paging like a DJ app.
            x0, x1 = self.waveform.visible_range()
            if not x0 <= t <= x1:
                width = x1 - x0
                self.waveform.setXRange(t - width * 0.05, t + width * 0.95, padding=0)
        elif self._was_playing:
            self.player.stop()  # finished on its own: release the audio device
        if playing != self._was_playing:
            S = QStyle.StandardPixmap
            self.play_action.setIcon(self.style().standardIcon(S.SP_MediaPause if playing else S.SP_MediaPlay))
            self.play_action.setText("Pause" if playing else "Play")
            self._was_playing = playing

    # --------------------------------------------------------------- stems

    def separate_stems(self) -> None:
        if self.audio is None or self.stems is not None or self._stem_worker is not None:
            return
        if not stems.is_available():
            QMessageBox.information(
                self, "Stems not installed",
                "Stem separation needs PyTorch and Demucs.\n\n"
                "Double-click install_stems.bat in the app folder, then restart the app.",
            )
            return
        worker = stems.StemWorker(self.audio)
        worker.progress.connect(self._on_stem_progress)
        worker.finished.connect(self._on_stems_ready)
        worker.failed.connect(self._on_stems_failed)
        self._stem_worker = worker
        self._update_enabled()
        self._set_busy("Preparing stem separation…", -1)
        threading.Thread(target=worker.run, daemon=True).start()

    def _is_current_worker(self) -> bool:
        worker = self._stem_worker
        return worker is not None and worker is self.sender() and worker.audio is self.audio

    def _on_stem_progress(self, fraction: float, message: str) -> None:
        if self._is_current_worker():
            self._set_busy(message, fraction)

    def _on_stems_ready(self, result: dict) -> None:
        if not self._is_current_worker():
            return  # a different file was opened meanwhile
        self._stem_worker = None
        self._set_busy(None)
        self._set_stems(result)
        self.statusBar().showMessage("Stems ready. Use M / S to mute or solo them.")

    def _on_stems_failed(self, message: str) -> None:
        if not self._is_current_worker():
            return
        self._stem_worker = None
        self._set_busy(None)
        self.statusBar().showMessage("Stem separation failed")
        self._update_enabled()
        QMessageBox.critical(self, "Stem separation failed", message)

    def _set_stems(self, result: dict[str, np.ndarray]) -> None:
        audio = self.audio
        self.stems = result
        self.stem_monos = {k: analysis.to_mono(v) for k, v in result.items()}
        self.stem_powers = {k: analysis.block_power(m) for k, m in self.stem_monos.items()}

        view_range = self.waveform.visible_range()
        for row in self.stem_rows:
            key = row.info.key
            data = analysis.analyze_layers(
                [self.stem_monos[key]], [self.stem_powers[key]], [row.rgb], audio.sample_rate
            )
            row.reset()
            row.lane.set_data(data, audio.duration)
            row.lane.set_selection(self.sel_start, self.sel_end)
            row.lane.set_playhead(self.playhead)
            row.set_visible(True)
        self.waveform.setXRange(*view_range, padding=0)  # lanes reset the shared zoom

        self.color_combo.blockSignals(True)
        self.color_combo.setCurrentIndex(1)
        self.color_combo.blockSignals(False)
        self._refresh_mix()
        self._update_enabled()

    def _enabled_stems(self) -> list[str]:
        any_solo = any(r.solo.isChecked() for r in self.stem_rows)
        return [
            r.info.key for r in self.stem_rows
            if (r.solo.isChecked() if any_solo else not r.mute.isChecked())
        ]

    def _current_mix(self) -> np.ndarray:
        """What you hear: the original, or the sum of the audible stems."""
        if self.stems is None:
            return self.audio.samples
        keys = self._enabled_stems()
        if len(keys) == len(self.stem_rows):
            return self.audio.samples  # all stems on: use the exact original
        if not keys:
            return np.zeros((self.audio.frames, next(iter(self.stems.values())).shape[1]), np.float32)
        return np.sum([self.stems[k] for k in keys], axis=0, dtype=np.float32)

    def _mix_label(self) -> str:
        if self.stems is None:
            return ""
        keys = self._enabled_stems()
        if len(keys) == len(self.stem_rows):
            return ""
        labels = [r.info.label for r in self.stem_rows if r.info.key in keys]
        return " + ".join(labels) if labels else "silence"

    def _refresh_mix(self) -> None:
        """Re-apply mute/solo and color mode to playback and the waveform."""
        if self.audio is None:
            return
        stem_mode = self.stems is not None and self.color_combo.currentIndex() == 1
        self.legend.setVisible(not stem_mode)
        if self.stems is None:
            data = self.base_data
        else:
            keys = self._enabled_stems()
            for row in self.stem_rows:
                row.lane.set_dimmed(row.info.key not in keys)
            mix = self._current_mix()
            self.player.set_buffer(mix, self.audio.sample_rate)

            if stem_mode:
                rows = [r for r in self.stem_rows if r.info.key in keys] or self.stem_rows[:1]
                gain = 1.0 if keys else 0.0  # nothing audible: flat line
                data = analysis.analyze_layers(
                    [self.stem_monos[r.info.key] * gain for r in rows],
                    [self.stem_powers[r.info.key] for r in rows],
                    [r.rgb for r in rows],
                    self.audio.sample_rate,
                )
            elif len(keys) == len(self.stem_rows):
                data = self.base_data
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    data = analysis.analyze(mix, self.audio.sample_rate)
                finally:
                    QApplication.restoreOverrideCursor()

        self.waveform.set_data(data)
        self.overview.set_data(data)
        label = self._mix_label()
        self.title_label.setText(self.audio.path.name + (f"  [{label}]" if label else ""))

    # -------------------------------------------------------------- export

    def _selection_frames(self) -> tuple[int, int]:
        sr = self.audio.sample_rate
        return int(round(self.sel_start * sr)), int(round(self.sel_end * sr))

    def _ask_format(self, title: str) -> str | None:
        formats = list(audio_io.SAVE_FORMATS)
        last = self.settings.value("export_format", formats[2])
        choice, ok = QInputDialog.getItem(
            self, title, "Format:", formats, formats.index(last) if last in formats else 0, False
        )
        if not ok:
            return None
        self.settings.setValue("export_format", choice)
        return audio_io.SAVE_FORMATS[choice]

    def export_selection(self) -> None:
        if self.audio is None:
            return
        s0, s1 = self._selection_frames()
        if s1 - s0 < 2:
            QMessageBox.warning(self, "Nothing selected", "Select a part of the track first (Shift+drag).")
            return

        label = self._mix_label()
        suffix = f" ({label})" if label else " (trim)"
        source_ext = self.audio.path.suffix.lower()
        filters = list(audio_io.SAVE_FORMATS)
        default_filter = next((f for f, e in audio_io.SAVE_FORMATS.items() if e == source_ext), filters[2])
        ext = audio_io.SAVE_FORMATS[default_filter]
        folder = self.settings.value("export_dir", str(self.audio.path.parent))
        suggested = str(Path(folder) / f"{self.audio.path.stem}{suffix}{ext}")

        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Export selection", suggested, ";;".join(filters), default_filter
        )
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() not in audio_io.SAVE_FORMATS.values():
            path = path.with_suffix(audio_io.SAVE_FORMATS.get(chosen_filter, ext))
        self.settings.setValue("export_dir", str(path.parent))

        samples = self._current_mix()[s0:s1]
        sr = self.audio.sample_rate
        fade_in, fade_out = self.fade_in.value(), self.fade_out.value()

        def job():
            audio_io.save(path, audio_io.apply_fades(samples, sr, fade_in, fade_out), sr)
            return path

        self._export(job, f"Exporting {path.name}…")

    def export_stems(self) -> None:
        if self.audio is None or self.stems is None:
            return
        s0, s1 = self._selection_frames()
        if s1 - s0 < 2:
            QMessageBox.warning(self, "Nothing selected", "Select a part of the track first (Shift+drag).")
            return
        ext = self._ask_format("Export stems")
        if ext is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export stems to folder", self.settings.value("export_dir", str(self.audio.path.parent))
        )
        if not folder:
            return
        self.settings.setValue("export_dir", folder)

        keys = self._enabled_stems() or [r.info.key for r in self.stem_rows]
        labels = {r.info.key: r.info.label for r in self.stem_rows}
        parts = {k: self.stems[k][s0:s1] for k in keys}
        sr, name = self.audio.sample_rate, self.audio.path.stem
        fade_in, fade_out = self.fade_in.value(), self.fade_out.value()

        def job():
            for key, samples in parts.items():
                audio_io.save(
                    Path(folder) / f"{name} - {labels[key]}{ext}",
                    audio_io.apply_fades(samples, sr, fade_in, fade_out), sr,
                )
            return Path(folder)

        self._export(job, f"Exporting {len(parts)} stems…")

    def _export(self, job, message: str) -> None:
        self._busy = True
        self._update_enabled()
        self._set_busy(message, -1)
        self._run_task(job, self._on_exported, self._on_export_failed)

    def _on_exported(self, path: Path) -> None:
        self._busy = False
        self._set_busy(None)
        self._update_enabled()
        self.statusBar().showMessage(f"Exported to {path}")

    def _on_export_failed(self, message: str) -> None:
        self._busy = False
        self._set_busy(None)
        self._update_enabled()
        self.statusBar().showMessage("Export failed")
        QMessageBox.critical(self, "Export failed", message)

    def closeEvent(self, event) -> None:
        self.player.close()
        super().closeEvent(event)
