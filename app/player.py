"""Audio playback from a numpy buffer, with live buffer swapping and looping."""
import threading

import numpy as np
import sounddevice as sd


class Player:
    """Plays a (frames, channels) float32 buffer on the default output device.

    The buffer can be swapped while playing (e.g. when a stem is muted), as
    long as it has the same length, and playback continues seamlessly.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._buffer: np.ndarray | None = None
        self._sample_rate = 0
        self._pos = 0
        self._start = 0
        self._end = 0
        self._loop = False
        self._playing = False

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> int:
        """Frame currently being heard (corrected for output latency)."""
        with self._lock:
            pos = self._pos
            if self._playing and self._stream is not None:
                pos -= int(self._stream.latency * self._sample_rate)
            return max(pos, self._start) if self._playing else pos

    def set_buffer(self, samples: np.ndarray, sample_rate: int) -> None:
        samples = np.ascontiguousarray(samples[:, :2], dtype=np.float32)
        if (
            self._stream is not None
            and (sample_rate != self._sample_rate or samples.shape[1] != self._stream.channels)
        ):
            self.close()
        with self._lock:
            self._buffer = samples
            self._sample_rate = sample_rate
            self._end = min(self._end, samples.shape[0])
            self._pos = min(self._pos, samples.shape[0])

    def play(self, start: int, end: int | None = None, loop: bool = False,
             position: int | None = None) -> None:
        """Play frames start..end (looping back to start if loop), beginning at
        position (defaults to start)."""
        if self._buffer is None:
            return
        frames = self._buffer.shape[0]
        end = frames if end is None else min(end, frames)
        start = max(0, min(start, end - 1))
        position = start if position is None else max(start, min(position, end - 1))
        with self._lock:
            self._start, self._end, self._pos = start, end, position
            self._loop = loop
            self._playing = True
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._buffer.shape[1],
                dtype="float32",
                callback=self._callback,
            )
        if not self._stream.active:
            self._stream.start()

    def set_loop(self, loop: bool) -> None:
        with self._lock:
            self._loop = loop

    def set_range(self, start: int, end: int) -> None:
        """Move the play/loop bounds while playing (e.g. dragging the selection)."""
        with self._lock:
            if self._buffer is not None and end > start:
                self._start, self._end = max(start, 0), min(end, self._buffer.shape[0])

    def stop(self) -> None:
        with self._lock:
            self._playing = False
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _callback(self, outdata, frames, _time, _status) -> None:
        with self._lock:
            written = 0
            while self._playing and written < frames:
                if self._pos >= self._end:
                    if self._loop and self._end > self._start:
                        self._pos = self._start
                    else:
                        self._playing = False  # the UI notices and stops the stream
                        break
                n = min(frames - written, self._end - self._pos)
                outdata[written:written + n] = self._buffer[self._pos:self._pos + n]
                self._pos += n
                written += n
            outdata[written:] = 0
