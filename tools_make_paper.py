"""生成羊皮纸质感底图（比纯色有温度，又不抢文字）。

做法：底色 + 细纤维噪点 + 大尺度低频斑驳 + 四角轻微压暗。
全部用 numpy 若可用，退化到纯 PIL 也能跑。输出 1920×1080 PNG。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BASE = (246, 234, 210)       # #F6EAD2 羊皮纸


def make_paper(w: int = 1920, h: int = 1080, seed: int = 7) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), BASE)

    # 1) 低频斑驳：先画小噪声再放大，得到自然的色块起伏
    low = Image.new("RGB", (w // 24, h // 24))
    px = low.load()
    for y in range(low.height):
        for x in range(low.width):
            d = rnd.randint(-9, 9)
            px[x, y] = (BASE[0] + d, BASE[1] + d - 1, BASE[2] + d - 3)
    low = low.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(9))
    img = Image.blend(img, low, 0.55)

    # 2) 纤维：稀疏的短浅线
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(int(w * h / 9000)):
        x, y = rnd.randrange(w), rnd.randrange(h)
        ln = rnd.randint(6, 22)
        ang = rnd.uniform(-0.35, 0.35)
        dx, dy = ln * 0.96, ln * ang
        dark = rnd.randint(120, 170)
        d.line([(x, y), (x + dx, y + dy)], fill=(dark, dark - 18, dark - 40, 16), width=1)

    # 3) 细颗粒
    grain = Image.effect_noise((w, h), 14).convert("L")
    grain = Image.merge("RGB", (grain, grain, grain))
    img = Image.blend(img, grain, 0.045)

    # 4) 四角压暗（径向 vignette，很轻）
    vign = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vign)
    cx, cy, r = w / 2, h / 2, max(w, h) * 0.72
    steps = 80
    for i in range(steps, 0, -1):
        rr = r * i / steps
        v = int(30 * (i / steps) ** 2.4)
        vd.ellipse([cx - rr, cy - rr * 0.62, cx + rr, cy + rr * 0.62], fill=v)
    vign = vign.filter(ImageFilter.GaussianBlur(40))
    shade = Image.new("RGB", (w, h), (120, 96, 62))
    img = Image.composite(shade, img, vign.point(lambda v: int(v * 0.9)))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="生成羊皮纸底图")
    ap.add_argument("out", nargs="?", default="docs/ppt/assets/paper.png")
    ap.add_argument("--size", default="1920x1080")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = make_paper(w, h)
    img.save(out, optimize=True)
    print(f"{out}  {img.size}  {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())