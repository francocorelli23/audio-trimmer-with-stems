"""Loading audio files into numpy arrays."""
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Formats libsndfile reads accurately on its own. Everything else (mp3, m4a, ...)
# is decoded by ffmpeg.
NATIVE_EXTS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}

OPEN_FILTER = "Audio files (*.mp3 *.m4a *.wav *.flac *.ogg *.aac *.aif *.aiff);;All files (*)"
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".aif", ".aiff"}

# Export formats: file-dialog filter -> extension. ffmpeg encodes mp3/m4a.
SAVE_FORMATS = {
    "WAV (*.wav)": ".wav",
    "FLAC (*.flac)": ".flac",
    "MP3 320 kbps (*.mp3)": ".mp3",
    "M4A / AAC 256 kbps (*.m4a)": ".m4a",
}
_FFMPEG_CODECS = {
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    ".m4a": ["-c:a", "aac", "-b:a", "256k"],
}

# Stops ffmpeg from flashing a console window when called from the GUI.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AudioLoadError(Exception):
    pass


class AudioSaveError(Exception):
    pass


@dataclass
class Audio:
    samples: np.ndarray  # float32, shape (frames, channels), values in -1..1
    sample_rate: int
    path: Path

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def load(path) -> Audio:
    path = Path(path)
    if path.suffix.lower() in NATIVE_EXTS:
        try:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            raise AudioLoadError(str(e)) from e
    else:
        samples, sample_rate = _load_with_ffmpeg(path)

    if samples.shape[0] == 0:
        raise AudioLoadError("The file contains no audio.")
    return Audio(samples, sample_rate, path)


def _find_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise AudioLoadError(
            f"{name} was not found. Install it with: winget install Gyan.FFmpeg"
        )
    return exe


def _probe(path: Path) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream."""
    result = subprocess.run(
        [
            _find_tool("ffprobe"), "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        raise AudioLoadError(result.stderr.strip() or "ffprobe could not read the file.")
    streams = json.loads(result.stdout).get("streams")
    if not streams:
        raise AudioLoadError("No audio stream found in the file.")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def _load_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, channels = _probe(path)
    # Decode to raw 32-bit float samples on stdout. Rate and channel count are
    # passed explicitly so the output always matches what we probed.
    result = subprocess.run(
        [
            _find_tool("ffmpeg"), "-v", "error",
            "-i", str(path), "-map", "0:a:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(sample_rate), "-ac", str(channels),
            "-",
        ],
        capture_output=True, creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise AudioLoadError(message or "ffmpeg could not decode the file.")
    samples = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, channels)
    return samples.copy(), sample_rate  # copy: frombuffer arrays are read-only


def apply_fades(samples: np.ndarray, sample_rate: int, fade_in: float, fade_out: float) -> np.ndarray:
    """Return a copy with equal-power fades of the given lengths (seconds)."""
    out = samples.copy()
    for seconds, at_start in ((fade_in, True), (fade_out, False)):
        n = min(int(seconds * sample_rate), out.shape[0])
        if n <= 0:
            continue
        curve = np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32)) ** 2
        if at_start:
            out[:n] *= curve[:, None]
        else:
            out[-n:] *= curve[::-1, None]
    return out


def save(path, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples (frames, channels) to path; the format follows the extension."""
    path = Path(path)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
    ext = path.suffix.lower()
    try:
        if ext in (".wav", ".flac"):
            sf.write(path, samples, sample_rate, subtype="PCM_24")
        elif ext in _FFMPEG_CODECS:
            _save_with_ffmpeg(path, samples, sample_rate, _FFMPEG_CODECS[ext])
        else:
            raise AudioSaveError(f"Unsupported export format: {ext or '(none)'}")
    except (OSError, RuntimeError, sf.LibsndfileError) as e:
        raise AudioSaveError(str(e)) from e


def _save_with_ffmpeg(path: Path, samples: np.ndarray, sample_rate: int, codec: list[str]) -> None:
    try:
        ffmpeg = _find_tool("ffmpeg")
    except AudioLoadError as e:
        raise AudioSaveError(str(e)) from e
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(samples.shape[1]), "-i", "-",
            *codec, str(path),
        ],
        input=samples.tobytes(), capture_output=True, creationflags=_NO_WINDOW,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise AudioSaveError(message or "ffmpeg could not encode the file.")
