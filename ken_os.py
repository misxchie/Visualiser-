#!/usr/bin/env python3
"""
KEN.OS — a minimal, dark CRT audio oscilloscope.
Usage: python ken_os.py "path/to/song.mp3"
Dependencies: see requirements.txt
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44100
WINDOW = 2048
SCREEN_W, SCREEN_H = 1100, 700
GREEN = (126, 255, 153)
DIM_GREEN = (25, 75, 43)
GRID_GREEN = (19, 49, 31)


def load_audio(path: Path) -> np.ndarray:
    """Load audio as float32 stereo. soundfile handles WAV/FLAC; ffmpeg handles MP3."""
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            # Use ffmpeg for reliable resampling when needed.
            raise RuntimeError("resample")
    except Exception:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Could not read this audio file. Install imageio-ffmpeg "
                "(included in requirements.txt) and try again."
            ) from exc

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE), "-ac", "2", "pipe:1"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"FFmpeg couldn't decode the file:\n{message}")
        data = np.frombuffer(result.stdout, dtype=np.float32)
        if data.size == 0:
            raise RuntimeError("The audio file appears to be empty or unsupported.")
        data = data.reshape(-1, 2)

    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


class AudioPlayer:
    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.position = 0
        self.lock = threading.Lock()
        self.latest = np.zeros(WINDOW, dtype=np.float32)
        self.playing = True
        self.finished = False
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=1024,
            callback=self.callback,
        )

    def callback(self, outdata, frames, time_info, status):
        if status:
            pass
        with self.lock:
            start = self.position
            end = min(start + frames, len(self.audio))
            count = end - start
            outdata[:] = 0
            if count > 0:
                outdata[:count] = self.audio[start:end]
                mono = self.audio[start:end].mean(axis=1)
                if count >= WINDOW:
                    self.latest[:] = mono[-WINDOW:]
                else:
                    self.latest[:-count] = self.latest[count:]
                    self.latest[-count:] = mono
                self.position = end
            if end >= len(self.audio):
                self.finished = True
                raise sd.CallbackStop()

    def snapshot(self):
        with self.lock:
            return self.latest.copy(), self.finished, self.position


def draw_crt(screen, samples, elapsed, font, small_font):
    screen.fill((0, 0, 0))

    # CRT glass: nearly black, with a restrained phosphor-green edge.
    crt = pygame.Rect(105, 75, SCREEN_W - 210, SCREEN_H - 150)
    pygame.draw.rect(screen, (2, 5, 3), crt, border_radius=22)
    pygame.draw.rect(screen, (22, 43, 29), crt, width=2, border_radius=22)

    inner = crt.inflate(-34, -34)
    pygame.draw.rect(screen, (1, 4, 2), inner)

    # Oscilloscope grid.
    cols, rows = 10, 8
    for i in range(cols + 1):
        x = inner.left + i * inner.width / cols
        color = (30, 61, 39) if i == cols // 2 else GRID_GREEN
        pygame.draw.line(screen, color, (int(x), inner.top), (int(x), inner.bottom), 1)
    for i in range(rows + 1):
        y = inner.top + i * inner.height / rows
        color = (30, 61, 39) if i == rows // 2 else GRID_GREEN
        pygame.draw.line(screen, color, (inner.left, int(y)), (inner.right, int(y)), 1)

    # Small grid subdivisions.
    for i in range(cols):
        for j in range(rows):
            x1 = inner.left + (i + .5) * inner.width / cols
            y1 = inner.top + (j + .5) * inner.height / rows
            pygame.draw.circle(screen, (13, 33, 21), (int(x1), int(y1)), 1)

    # Analyze the low end and transients to drive the aggressive CRT behavior.
    window = samples.astype(np.float32, copy=False)
    if len(window) < 16:
        window = np.pad(window, (16 - len(window), 0))
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    freqs = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE)
    bass_mask = (freqs >= 35) & (freqs <= 150)
    bass = float(np.mean(spectrum[bass_mask])) if np.any(bass_mask) else 0.0
    level = float(np.sqrt(np.mean(window ** 2)) + 1e-8)
    bass_drive = np.clip(bass / (np.mean(spectrum) + 1e-6) / 5.0, 0.0, 1.8)
    drive = float(np.clip(level * 3.5 + bass_drive * .55, 0.0, 1.7))

    # Resample the real audio waveform across the screen.
    count = inner.width
    indices = np.linspace(0, len(window) - 1, count).astype(np.int32)
    trace = window[indices].copy()
    trace -= np.mean(trace)

    # Automatic gain with bass-driven overload and jagged electrical instability.
    trace *= (inner.height * .24) * (1.0 + drive * .65)
    rng = np.random.default_rng(int(elapsed * 30) % 100000)
    hit = max(0.0, drive - .48)
    if hit > 0:
        spikes = rng.normal(0, 1, count) * hit * inner.height * .055
        trace += spikes
        # A few sharp, short-lived voltage kicks.
        for _ in range(int(hit * 8)):
            p = int(rng.integers(0, count))
            width = int(rng.integers(2, 13))
            amp = float(rng.choice([-1, 1])) * hit * inner.height * .22
            trace[p:min(count, p + width)] += np.linspace(amp, 0, min(width, count - p))

    # Soft clipping gives the trace a driven, distorted analog feel.
    limit = inner.height * .39
    trace = limit * np.tanh(trace / max(1, limit * (1.0 - min(.65, hit * .28))))

    center_y = inner.centery
    points = [(inner.left + i, int(center_y - v)) for i, v in enumerate(trace)]

    # Phosphor persistence: several faint echoes behind the main trace.
    for offset, alpha_color in ((-5, (24, 90, 45)), (-2, (42, 145, 72)), (2, (36, 112, 55))):
        ghost = [(x, y + offset) for x, y in points]
        pygame.draw.lines(screen, alpha_color, False, ghost, 2)
    pygame.draw.lines(screen, (55, 190, 91), False, points, 3)
    pygame.draw.lines(screen, GREEN, False, points, 1)

    # Tiny phosphor bloom around the brightest center trace.
    bloom = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
    pygame.draw.lines(
        bloom, (70, 255, 120, 30), False,
        [(x - inner.left, y - inner.top) for x, y in points], 7
    )
    screen.blit(bloom, inner.topleft, special_flags=pygame.BLEND_RGBA_ADD)

    # Scanlines and very subtle glass shading.
    for y in range(inner.top, inner.bottom, 4):
        pygame.draw.line(screen, (0, 2, 1), (inner.left, y), (inner.right, y), 1)

    # Minimal instrument markings, kept intentionally understated.
    label = small_font.render("KEN.OS  /  ANALOG SIGNAL", True, (67, 125, 81))
    screen.blit(label, (inner.left + 12, inner.top + 10))
    status = small_font.render("CH1  •  AUDIO", True, (67, 125, 81))
    screen.blit(status, (inner.right - status.get_width() - 12, inner.top + 10))

    # Soft dark vignette around the screen edges.
    vignette = pygame.Surface((crt.width, crt.height), pygame.SRCALPHA)
    for inset in range(0, 42, 3):
        alpha = int(3 + (42 - inset) * .45)
        rect = pygame.Rect(inset, inset, crt.width - inset * 2, crt.height - inset * 2)
        if rect.width > 0 and rect.height > 0:
            pygame.draw.rect(vignette, (0, 0, 0, alpha), rect, width=3, border_radius=18)
    screen.blit(vignette, crt.topleft)


def choose_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a track for KEN.OS",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("All files", "*.*")]
        )
        root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="KEN.OS retro CRT audio visualizer")
    parser.add_argument("audio", nargs="?", help="Audio file (MP3, WAV, FLAC, OGG, M4A)")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser() if args.audio else choose_file()
    if not audio_path:
        print("No audio file selected. Try: python ken_os.py song.mp3")
        return 1
    if not audio_path.is_file():
        print(f"File not found: {audio_path}")
        return 1

    print(f"Loading {audio_path.name}...")
    try:
        audio = load_audio(audio_path)
    except Exception as exc:
        print(f"Audio error: {exc}")
        return 1

    pygame.init()
    pygame.display.set_caption("KEN.OS — Analog Audio Visualizer")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 12)

    try:
        player = AudioPlayer(audio)
        player.stream.start()
    except Exception as exc:
        pygame.quit()
        print(f"Audio output error: {exc}")
        print("Check that your speakers/headphones are connected and available.")
        return 1

    running = True
    start_time = pygame.time.get_ticks() / 1000
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        samples, finished, position = player.snapshot()
        elapsed = pygame.time.get_ticks() / 1000 - start_time
        draw_crt(screen, samples, elapsed, font, small_font)

        if finished:
            end_text = small_font.render("PLAYBACK COMPLETE  /  ESC TO EXIT", True, (67, 125, 81))
            screen.blit(end_text, (SCREEN_W // 2 - end_text.get_width() // 2, SCREEN_H - 35))

        pygame.display.flip()
        clock.tick(60)

        if finished:
            # Keep the final frame visible until the user exits.
            pass

    try:
        player.stream.stop()
        player.stream.close()
    except Exception:
        pass
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
