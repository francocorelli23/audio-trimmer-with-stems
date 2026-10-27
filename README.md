# Audio Trimmer

A Windows audio trimmer with a rekordbox-style colored waveform and optional
AI stem separation (drums, bass, instruments, vocals).

## Setup (one time)

1. Install [ffmpeg](https://ffmpeg.org): `winget install Gyan.FFmpeg`
2. Double-click **install.bat** (creates `.venv` with Python 3.13 and installs the app)
3. Optional, for stems: double-click **install_stems.bat** (~3 GB: PyTorch + Demucs)

## Run

Double-click **run.bat**, or drag an audio file onto it. Supported: MP3, M4A, WAV, FLAC (and OGG, AAC, AIFF).

## Using it

| Do this | How |
|---|---|
| Open a file | Ctrl+O, or drop it on the window |
| Zoom / scroll | Mouse wheel / drag the waveform |
| Jump around | Click the waveform, or click / drag the box in the overview strip |
| Select a part | **Shift+drag** on any waveform, drag the yellow handles, or type times in Start / End |
| Set start / end at the playhead | I / O |
| Play / pause from the playhead | Space |
| Play the selection | Shift+Space |
| Loop the selection | L (toggle) |
| Stop (back to where you started) | Esc |
| Go to selection start / end | Home / End |
| Zoom to whole track / to selection | Ctrl+0 / Ctrl+E |
| Select everything | Ctrl+A |
| Separate stems | Ctrl+T |
| Export the selection | Ctrl+S |
| Export each audible stem of the selection | Ctrl+Shift+S |

**Waveform colors.** In *Frequency colors* mode, red means bass, green mids and blue highs.
After separating stems, *Stem colors* mode colors each moment by which stem is
loudest: drums orange, bass pink-red, instruments blue, vocals green.

**Stems.** Each stem gets its own lane with **M** (mute) and **S** (solo).
Muting or soloing changes what you hear, the main waveform, and what gets
exported, so you can e.g. export just the vocals of a section. The first run
downloads the Demucs model (~80 MB). Results are cached in
`%LOCALAPPDATA%\AudioTrimmer\stems`, so reopening a song loads its stems instantly.
On an NVIDIA GPU a 3-minute song takes roughly half a minute; on CPU a few minutes.

**Export.** WAV/FLAC are 24-bit, MP3 is 320 kbps, M4A is AAC 256 kbps.
Fade in / fade out (bottom bar) are applied to exports.

## Project layout

```
main.py              starts the app (dark theme)
app/audio_io.py      load (soundfile / ffmpeg), save, fades
app/analysis.py      per-block peaks + frequency / stem energy for coloring
app/waveform.py      waveform, overview and lane widgets
app/player.py        playback (sounddevice)
app/stems.py         Demucs separation in a background thread + cache
app/main_window.py   window, toolbar, stem lanes, trim bar
```

For debugging, run `.venv\Scripts\python.exe main.py` in a terminal to see error output.

## License

MIT, see [LICENSE](LICENSE). Stem separation uses [Demucs](https://github.com/facebookresearch/demucs) (MIT).
