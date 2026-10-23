"""Deterministic synthetic stimulus for the demo pipeline.

The fixture builds small closed-form clips (no random source) with both a
video and an audio stream, so that the pretraining run, the checkpoint and the
evaluation metrics are reproducible byte-for-bit inside the environment pinned
by ``uv.lock``. The codecs (``mpeg4`` video, ``aac`` audio) are the ones the
test suite already relies on.

Usage:
    uv run demo/generate_fixture.py
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

SECONDS = 2
RATE = 8
SAMPLE_RATE = 16_000
SIZE = 32
FRAMES = SECONDS * RATE
ROOT = Path("demo/data/sportvista")
PLAN = (("group_001", 4), ("group_002", 4), ("group_003", 4))
# One label carries a moving block with a rising tone, the other a static
# block with a constant tone.
LABELS = {"football": True, "tennis": False}


def video_pixels(
    t: int, moving: bool, level: int, block: int, phase: int
) -> np.ndarray:
    """Return one deterministic, in-normalisation-range synthetic frame."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dither = 6 * (((xx // 4) + (yy // 4) + t + phase) % 2) - 3
    frame = np.full((SIZE, SIZE, 3), 130, dtype=np.int16)
    frame += dither[..., None]
    value = float(np.clip(level + 15 * np.cos(2 * np.pi * t / FRAMES), 0, 255))
    offset = int(round(6 * np.sin(2 * np.pi * t / FRAMES))) if moving else 0
    top = (SIZE - block) // 3
    left = (top + offset) % (SIZE - block)
    frame[top : top + block, left : left + block] = int(round(value))
    return frame.astype(np.uint8)


def write_clip(
    path: Path,
    moving: bool,
    tone: float,
    level: int,
    block: int,
    phase: int,
    volume: float,
) -> None:
    """Encode one deterministic synthetic audio-visual clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=RATE)
        video_stream.width = SIZE
        video_stream.height = SIZE
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=SAMPLE_RATE)
        audio_stream.layout = "mono"
        for t in range(FRAMES):
            frame = av.VideoFrame.from_ndarray(
                video_pixels(t, moving, level, block, phase), format="rgb24"
            )
            frame.pts = t
            frame.time_base = Fraction(1, RATE)
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        for offset in range(0, SAMPLE_RATE * SECONDS, 1024):
            count = min(1024, SAMPLE_RATE * SECONDS - offset)
            time = np.arange(offset, offset + count) / SAMPLE_RATE
            samples = (volume * np.sin(2 * np.pi * tone * time)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(
                samples.reshape(1, -1), format="fltp", layout="mono"
            )
            frame.sample_rate = SAMPLE_RATE
            frame.pts = offset
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


def main() -> None:
    if ROOT.exists():
        import shutil

        shutil.rmtree(ROOT)
    counter = 0
    for label, moving in LABELS.items():
        for group, count in PLAN:
            for _index in range(count):
                counter += 1
                # Per-clip, closed-form variation keeps the reliability
                # indicators strictly inside the normaliser bounds: the
                # min-max fit must see continuous, per-clip values.
                level = 160 + 10 * ((counter * 7) % 8)
                block = 8 + 2 * (counter % 4)
                phase = counter % 2
                tone = (440.0 if moving else 660.0) + 10.0 * (counter % 5)
                volume = 0.18 + 0.02 * ((counter * 3) % 6)
                path = ROOT / label / group / f"clip_{counter:03d}.mp4"
                write_clip(path, moving, tone, level, block, phase, volume)
    print(f"wrote {2 * sum(count for _, count in PLAN)} synthetic clips under {ROOT}")


if __name__ == "__main__":
    main()
