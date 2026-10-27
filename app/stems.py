"""Stem separation with Demucs (htdemucs), run in a background thread.

Demucs and PyTorch are optional: the rest of the app works without them.
Separated stems are cached on disk, so reopening a song is instant.
"""
import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, Signal

from .audio_io import Audio

MODEL_NAME = "htdemucs"
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AudioTrimmer" / "stems"


@dataclass(frozen=True)
class StemInfo:
    key: str      # Demucs source name
    label: str
    color: str    # hex, used for the lane swatch and the waveform


# Order matches Demucs' output order for htdemucs.
STEMS = (
    StemInfo("drums", "Drums", "#ff9f1c"),
    StemInfo("bass", "Bass", "#ff3d6e"),
    StemInfo("other", "Instruments", "#3d9bff"),
    StemInfo("vocals", "Vocals", "#36e07f"),
)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def is_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))


def _cache_folder(path: Path) -> Path:
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{MODEL_NAME}"
    return CACHE_DIR / hashlib.sha1(key.encode()).hexdigest()[:16]


def load_cached(audio: Audio) -> dict[str, np.ndarray] | None:
    folder = _cache_folder(audio.path)
    files = {s.key: folder / f"{s.key}.flac" for s in STEMS}
    if not all(f.exists() for f in files.values()):
        return None
    stems = {k: sf.read(f, dtype="float32", always_2d=True)[0] for k, f in files.items()}
    if any(s.shape[0] != audio.frames for s in stems.values()):
        return None
    return stems


def _save_cache(audio: Audio, stems: dict[str, np.ndarray]) -> None:
    folder = _cache_folder(audio.path)
    folder.mkdir(parents=True, exist_ok=True)
    for key, samples in stems.items():
        sf.write(folder / f"{key}.flac", np.clip(samples, -1, 1), audio.sample_rate, subtype="PCM_24")


class _ProgressShim:
    """Stands in for the tqdm module inside demucs.apply, so we get progress
    callbacks instead of console output."""

    def __init__(self, callback):
        self.callback = callback

    def tqdm(self, iterable, **_kwargs):
        items = list(iterable)
        for i, item in enumerate(items):
            yield item
            self.callback((i + 1) / len(items))


class StemWorker(QObject):
    """Separates audio into stems. Move to a QThread and call run()."""

    progress = Signal(float, str)  # fraction 0..1 (or -1 = unknown), message
    finished = Signal(dict)        # {stem key: (frames, channels) float32}
    failed = Signal(str)

    def __init__(self, audio: Audio):
        super().__init__()
        self.audio = audio

    def run(self) -> None:
        try:
            cached = load_cached(self.audio)
            if cached is not None:
                self.finished.emit(cached)
                return
            stems = self._separate()
            try:
                _save_cache(self.audio, stems)
            except OSError:
                pass  # caching is a bonus; separation still succeeded
            self.finished.emit(stems)
        except Exception as e:  # report anything to the UI instead of dying silently
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _separate(self) -> dict[str, np.ndarray]:
        self.progress.emit(-1, "Loading PyTorch…")
        import julius
        import torch
        from demucs import apply as demucs_apply
        from demucs.pretrained import get_model

        self.progress.emit(-1, "Loading Demucs model (downloads ~80 MB the first time)…")
        model = get_model(MODEL_NAME)
        model.eval()

        audio = self.audio
        wav = torch.from_numpy(np.ascontiguousarray(audio.samples.T))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav[:2]
        if audio.sample_rate != model.samplerate:
            wav = julius.resample_frac(wav, audio.sample_rate, model.samplerate)

        # Same normalisation the demucs command-line tool uses.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std

        sources = None
        for device in self._devices(torch):
            label = "GPU" if device == "cuda" else "CPU"
            demucs_apply.tqdm = _ProgressShim(
                lambda f, label=label: self.progress.emit(f, f"Separating stems on {label}… {f:.0%}")
            )
            self.progress.emit(0.0, f"Separating stems on {label}…")
            try:
                with torch.no_grad():
                    sources = demucs_apply.apply_model(
                        model, wav[None], device=device, split=True, overlap=0.25,
                        shifts=1, progress=True,
                    )[0]
                break
            except RuntimeError:
                # Typically out of GPU memory or an unsupported (old) GPU:
                # fall back to the CPU, which always works.
                if device == "cpu":
                    raise
                torch.cuda.empty_cache()

        sources = sources * std + mean
        if audio.sample_rate != model.samplerate:
            sources = julius.resample_frac(sources, model.samplerate, audio.sample_rate)

        stems = {}
        for name, source in zip(model.sources, sources):
            samples = source.cpu().numpy().T.astype(np.float32)
            samples = samples[:audio.frames]
            if samples.shape[0] < audio.frames:
                samples = np.pad(samples, ((0, audio.frames - samples.shape[0]), (0, 0)))
            if audio.channels == 1:
                samples = samples.mean(axis=1, keepdims=True)
            stems[name] = samples
        return stems

    @staticmethod
    def _devices(torch) -> list[str]:
        return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
