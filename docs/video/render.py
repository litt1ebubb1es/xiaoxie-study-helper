"""渲染驱动：stills / sheet / video。

用法（在仓库根目录）：
  python docs/video/render.py stills --t 1.5,4.0,12.0 --out docs/video/out/wip
  python docs/video/render.py sheet  --n 24 --cols 6 --out docs/video/out/wip/sheet.png
  python docs/video/render.py video  --out docs/video/out/xiaoxie-promo.mp4

视频：1920×1080 @60fps，x264 CRF16 / preset slow / AAC 256k，音频取 happy.mp3 的 12–60s
并在结尾 1.5s 淡出，保证最后一下落在拍子上。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from vlib import core, shots  # noqa: E402
from vlib.core import H, W, Canvas  # noqa: E402

WORK = HERE / "work"
OUT = HERE / "out"
AUDIO_JSON = WORK / "audio.json"
MP3 = HERE / "assets" / "happy.mp3"

DATA: dict = {}


def load_audio() -> dict:
    global DATA
    if not DATA:
        DATA = json.loads(AUDIO_JSON.read_text(encoding="utf-8"))
        DATA["_feat"] = {k: np.asarray(v, np.float32) for k, v in DATA["features"].items()}
        DATA["_beats"] = np.asarray(DATA["beats"], np.float32)
    return DATA


def feat(name: str, t: float) -> float:
    d = load_audio()["_feat"][name]
    i = int(round(t * DATA["fps"]))
    return float(d[min(max(i, 0), len(d) - 1)])


def kick(t: float, decay: float = 0.16) -> float:
    """上一拍之后按指数衰减的脉冲，权重取这一拍的 onset 强度 —— 画面跟着鼓点呼吸。"""
    d = load_audio()
    period = d["period"]
    i = math.floor(t / period)
    out = 0.0
    for j in (i, i - 1):
        tb = j * period
        if tb > t:
            continue
        w = feat("onset", tb)
        p = math.exp(-(t - tb) / decay)
        out = max(out, w * p)
    return min(1.0, out * 1.15)


def beat_index(t: float) -> float:
    return t / load_audio()["period"]


def make_ctx() -> dict:
    d = load_audio()
    return {"dur": d["clip_len"], "beat": beat_index, "kick": kick,
            "feat": feat, "fps": d["fps"]}


def render_frame(i: int, ctx: dict | None = None) -> np.ndarray:
    ctx = ctx or make_ctx()
    d = load_audio()
    t = i / d["fps"]
    shot, lt = shots.shot_at(t)
    ctx["t"] = t
    bg = "ink"
    cv = Canvas(bg=bg)
    post = shot.plate.render(cv, lt, ctx) or {}

    # 硬切就是硬切：切点那一帧不加闪白（整幅会被抬成灰片）
    params = dict(bloom=0.55, threshold=0.85, halation=0.25, grain=0.012, vig=0.35,
                  ca=1.2, exposure=1.0, flash=0.0)
    params.update({k: v for k, v in post.items() if k in params})
    return core.finish(cv, i, **params)


# ---------------------------------------------------------------- 并行渲染


def _init() -> None:
    load_audio()


def _render_chunk(args) -> int:
    start, end, outdir = args
    ctx = make_ctx()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(start, end):
        img = render_frame(i, ctx)
        Image.fromarray(img).save(outdir / f"{i:05d}.png", compress_level=1)
    return end - start


def render_video(out_path: Path, jobs: int, frames_dir: Path, reuse: bool) -> None:
    d = load_audio()
    n = d["frames"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    todo = n if not reuse else len([p for p in frames_dir.glob("*.png")])
    if not reuse or todo < n:
        chunk = max(8, math.ceil(n / (jobs * 4)))
        ranges = [(s, min(n, s + chunk), str(frames_dir)) for s in range(0, n, chunk)]
        import multiprocessing as mp
        t0 = time.time()
        done = 0
        with mp.Pool(jobs, initializer=_init) as pool:
            for got in pool.imap_unordered(_render_chunk, ranges):
                done += got
                el = time.time() - t0
                rate = done / el
                print(f"\r  渲染 {done}/{n} 帧  {rate:5.1f} fps  剩 {max(0, (n - done) / max(rate, 1e-6)):4.0f}s",
                      end="", flush=True)
        print()

    # ---- 编码 ----
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if not ff or not Path(ff).exists():
        raise SystemExit("找不到 ffmpeg")
    fade_start = shots.DUR - 1.5
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "warning", "-stats",
        "-framerate", "60", "-i", str(frames_dir / "%05d.png"),
        "-ss", f"{d['clip_start']:.3f}", "-t", f"{d['clip_len']:.3f}", "-i", str(MP3),
        "-af", f"afade=t=out:st={fade_start:.3f}:d=1.5,aresample=48000",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-x264-params", "aq-mode=3",
        "-vf", "scale=out_color_matrix=bt709,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart", str(out_path),
    ]
    print("  编码…")
    subprocess.run(cmd, check=True)
    print(f"  写出 {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "sheet", "video"])
    ap.add_argument("--t", default="1.2,4.0,7.0,12.0,18.0,23.0,30.0,36.0,42.0")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 4))
    ap.add_argument("--out", default="")
    ap.add_argument("--reuse", action="store_true", help="复用已经渲染好的帧")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = load_audio()

    if args.mode == "stills":
        tgt = Path(args.out or OUT / "wip")
        tgt.mkdir(parents=True, exist_ok=True)
        for ts in args.t.split(","):
            t = float(ts)
            i = int(round(t * d["fps"]))
            shot, lt = shots.shot_at(t)
            img = render_frame(i)
            p = tgt / f"still_{t:05.1f}_{shot.name}.png"
            Image.fromarray(img).save(p)
            print(f"  {t:5.1f}s  {shot.name:4s} → {p}")

    elif args.mode == "sheet":
        tgt = Path(args.out or OUT / "wip" / "sheet.png")
        tgt.parent.mkdir(parents=True, exist_ok=True)
        n = args.n
        cols = args.cols
        rows = math.ceil(n / cols)
        tw, th = 384, 216
        sheet = Image.new("RGB", (cols * tw + 8 * (cols + 1), rows * th + 8 * (rows + 1)), (16, 16, 18))
        for k in range(n):
            t = shots.DUR * k / n
            i = int(round(t * d["fps"]))
            img = Image.fromarray(render_frame(i)).resize((tw, th), Image.LANCZOS)
            r, c = divmod(k, cols)
            sheet.paste(img, (8 + c * (tw + 8), 8 + r * (th + 8)))
        sheet.save(tgt)
        print(f"  联系表 → {tgt}  ({n} 帧, {cols} 列)")

    else:
        tgt = Path(args.out or OUT / "小蟹学习助手-介绍.mp4")
        tgt.parent.mkdir(parents=True, exist_ok=True)
        render_video(tgt, args.jobs, OUT / "frames", args.reuse)


if __name__ == "__main__":
    main()