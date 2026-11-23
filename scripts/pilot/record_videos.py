"""Record one idle-policy episode for 4 representative configs (mp4 + gif).

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/pilot/record_videos.py
"""

from __future__ import annotations

import shutil
import time

import numpy as np
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip

from pilot_common import OUT_DIR, make_env, record_timing

CONFIGS = [(10, 0.7), (20, 1.0), (30, 1.5), (40, 2.0)]  # (vehicles_count, density)
LANES = 3
SEED = 1000
VIDEO_DIR = OUT_DIR / "videos"


def main() -> None:
    t0 = time.perf_counter()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VIDEO_DIR / "_tmp"
    for count, density in CONFIGS:
        name = f"veh{count}_dens{density}_lanes{LANES}"
        env = make_env(
            {"lanes_count": LANES, "vehicles_count": count, "vehicles_density": density,
             # larger frame for readability; does not influence the dynamics
             "screen_width": 1200, "screen_height": 300},
            render_mode="rgb_array",
        )
        env = RecordVideo(env, str(tmp), episode_trigger=lambda e: True, name_prefix=name)
        # Also capture the intermediate simulation frames (5 per decision step).
        env.unwrapped.set_record_video_wrapper(env)
        env.reset(seed=SEED)
        done, steps = False, 0
        while not done:
            _, _, term, trunc, info = env.step(1)  # IDLE
            steps += 1
            done = term or trunc
        env.close()

        src = next(tmp.glob(f"{name}-episode-*.mp4"))
        dst = VIDEO_DIR / f"{name}.mp4"
        shutil.move(str(src), dst)
        clip = VideoFileClip(str(dst))
        clip.write_gif(str(VIDEO_DIR / f"{name}.gif"), fps=10)
        clip.close()
        print(f"{name}: {steps} steps, crashed={info['crashed']}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    record_timing("record_videos", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
