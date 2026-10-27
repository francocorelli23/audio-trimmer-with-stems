"""Per-block analysis used to draw the colored (rekordbox-style) waveform.

The audio is split into short blocks. For each block we store its min/max
sample, its average power, and an energy vector that decides its color:

* Frequency mode: energy in the low, mid and high bands. Low maps to red, mid
  to green and high to blue, so kicks show up red, vocals and synths green,
  and hi-hats blue.
* Stem mode: the energy of each stem. The color is a blend of the stem
  colors, weighted by how loud each stem is in that block.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HOP = 256                    # samples per block (~5.8 ms at 44.1 kHz)
FFT_SIZE = 1024              # window used to measure each block's frequencies
FFT_BATCH = 4096             # blocks per FFT batch, keeps memory use bounded
BAND_EDGES_HZ = (200, 2500)  # low | mid | high
MIN_FREQ_HZ = 20

# Lows naturally carry far more energy than highs, so mids and highs are
# boosted to make them show up as color at all.
BAND_GAIN = np.array([1.0, 2.0, 5.0], dtype=np.float32)
# Above 1 pushes each color toward its strongest band/stem, giving vivid hues
# instead of washed-out mixes.
COLOR_SHARPNESS = 1.6
STEM_SHARPNESS = 2.0


@dataclass
class WaveformData:
    mono: np.ndarray          # float32 samples, used when zoomed in very far
    sample_rate: int
    block_min: np.ndarray     # (blocks,)
    block_max: np.ndarray     # (blocks,)
    block_power: np.ndarray   # (blocks,) mean square
    block_energy: np.ndarray  # (blocks, k) what the color is computed from
    colorize: Callable[[np.ndarray], np.ndarray]  # (n, k) energy -> (n, 3) RGB 0..1

    @property
    def blocks(self) -> int:
        return self.block_min.size


def to_mono(samples: np.ndarray) -> np.ndarray:
    mono = samples.mean(axis=1, dtype=np.float32) if samples.shape[1] > 1 else samples[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32)


def _blocks(mono: np.ndarray) -> np.ndarray:
    n_blocks = max(-(-mono.size // HOP), 1)
    padded = np.zeros(n_blocks * HOP, dtype=np.float32)
    padded[:mono.size] = mono
    return padded.reshape(n_blocks, HOP)


def block_power(mono: np.ndarray) -> np.ndarray:
    blocks = _blocks(mono)
    return np.einsum("ij,ij->i", blocks, blocks) / HOP


def analyze(samples: np.ndarray, sample_rate: int) -> WaveformData:
    """Frequency-colored analysis of a whole track."""
    mono = to_mono(samples)
    blocks = _blocks(mono)
    return WaveformData(
        mono=mono,
        sample_rate=sample_rate,
        block_min=blocks.min(axis=1),
        block_max=blocks.max(axis=1),
        block_power=np.einsum("ij,ij->i", blocks, blocks) / HOP,
        block_energy=_band_energy(blocks.reshape(-1), blocks.shape[0], sample_rate),
        colorize=spectral_colors,
    )


def analyze_layers(
    monos: list[np.ndarray], powers: list[np.ndarray], colors: np.ndarray, sample_rate: int
) -> WaveformData:
    """Stem-colored analysis: the shape is the sum of the given stems, the
    color of each block blends the stem colors by their energy."""
    mix = np.sum(monos, axis=0, dtype=np.float32)
    blocks = _blocks(mix)
    return WaveformData(
        mono=mix,
        sample_rate=sample_rate,
        block_min=blocks.min(axis=1),
        block_max=blocks.max(axis=1),
        block_power=np.einsum("ij,ij->i", blocks, blocks) / HOP,
        block_energy=np.stack(powers, axis=1),
        colorize=partial(layer_colors, colors=np.asarray(colors, dtype=np.float32)),
    )


def _band_energy(padded: np.ndarray, n_blocks: int, sample_rate: int) -> np.ndarray:
    # One FFT window per block, centered on it.
    margin = (FFT_SIZE - HOP) // 2
    extended = np.concatenate([np.zeros(margin, np.float32), padded, np.zeros(margin, np.float32)])
    frames = sliding_window_view(extended, FFT_SIZE)[::HOP][:n_blocks]

    # Matrix that sums FFT bins into the three bands.
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / sample_rate)
    band_of_bin = np.digitize(freqs, BAND_EDGES_HZ)
    band_matrix = np.zeros((freqs.size, 3), dtype=np.float32)
    usable = freqs >= MIN_FREQ_HZ
    band_matrix[np.flatnonzero(usable), band_of_bin[usable]] = 1

    window = np.hanning(FFT_SIZE).astype(np.float32)
    bands = np.empty((n_blocks, 3), dtype=np.float32)
    for start in range(0, n_blocks, FFT_BATCH):
        spectrum = np.fft.rfft(frames[start:start + FFT_BATCH] * window, axis=1)
        bands[start:start + FFT_BATCH] = (np.abs(spectrum) ** 2).astype(np.float32) @ band_matrix
    return bands


def spectral_colors(band_energy: np.ndarray) -> np.ndarray:
    """Turn (n, 3) band energies into (n, 3) RGB values in 0..1."""
    amplitude = np.sqrt(band_energy) * BAND_GAIN
    peak = amplitude.max(axis=1, keepdims=True)
    rgb = np.divide(amplitude, peak, out=np.zeros_like(amplitude), where=peak > 0)
    return rgb ** COLOR_SHARPNESS


def layer_colors(energy: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Blend (k, 3) layer colors by (n, k) layer energies."""
    weight = energy ** (STEM_SHARPNESS / 2)  # amplitude ** sharpness
    total = weight.sum(axis=1, keepdims=True)
    weight = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    return weight @ colors
