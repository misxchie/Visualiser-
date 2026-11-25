# KEN.OS — Analog CRT Audio Visualizer

A minimal, black-screen, green-phosphor oscilloscope that reacts to real audio.
Bass-heavy sections drive the waveform into jagged spikes and analog-style clipping.

## Install

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

On Windows, if `sounddevice` cannot find an audio device, install/update your audio drivers
and check Windows Sound settings. The app plays through your default output device.

## Run

```bash
python ken_os.py
```

A file picker opens. Or pass a file directly:

```bash
python ken_os.py "C:\Music\my-track.mp3"
```

Supports MP3, WAV, FLAC, OGG and M4A (decoding uses bundled FFmpeg from imageio-ffmpeg).

Press **ESC** or **Q** to quit. The visualizer is designed for local audio files; it does not
download or stream music.
