"""实机画面版：把 appseq 里的真截图当素材，做推轨 / 焦点 / 指针 / 字幕。

素材是 `capture_app.py` 逐帧 grab 出来的 2560×1421 PNG（30 fps，本机就是 2K 屏），
所以镜头推到 1.9 倍时源像素还够（1.32 倍时是 1:1），推轨才敢做大。这里只做
"镜头运动"和"告诉人看哪里"：从整屏推到某一块面板，橙圈指位置，底部字幕条报章节。

规矩和别的版面一样：画面只依赖 lt（本版局部时间），不读全局状态、不随机。
指针轨迹不是录出来的，是按 capture 记下的点击表算出来的——所以它永远是旧的：
在点击前 0.5 秒滑到位，点完停住，再滑向下一个点。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import (C, H, W, Canvas, Layer, clamp, font, in_out_cubic, out_expo,
                   out_quint, ramp, text_width)

SEQ = Path(__file__).resolve().parent.parent / "work" / "appseq"
SRC_FPS = 30

_MANIFEST: dict | None = None


def _manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads((SEQ / "manifest.json").read_text(encoding="utf-8"))
    return _MANIFEST


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


SW, SH = (int(v) for v in _manifest().get("size", [W, H]))


def _src_image(src_t: float) -> Image.Image:
    n = int(_manifest()["frames"])
    idx = max(0, min(n - 1, int(src_t * SRC_FPS)))
    p = SEQ / f"{idx:05d}.png"
    if not p.exists():
        return Image.new("RGB", (SW, SH), (250, 250, 248))
    return Image.open(p).convert("RGB")


def _keys_at(clicks: list[dict], t: float) -> tuple[float, float]:
    """点击表 → 指针位置：点完停 0.5s，然后缓动滑向下一个点。"""
    if not clicks:
        return (SW / 2, SH / 2)
    prev = None
    nxt = None
    for k in clicks:
        if k["t"] <= t:
            prev = k
        elif nxt is None:
            nxt = k
    if prev is None:
        return float(clicks[0]["x"]), float(clicks[0]["y"])
    if nxt is None:
        return float(prev["x"]), float(prev["y"])
    hold = prev["t"] + 0.5
    if t <= hold:
        return float(prev["x"]), float(prev["y"])
    dur = min(0.62, max(0.24, nxt["t"] - hold))
    u = in_out_cubic(clamp((t - hold) / dur))
    return (prev["x"] + (nxt["x"] - prev["x"]) * u,
            prev["y"] + (nxt["y"] - prev["y"]) * u)


def _lerp_keys(keys, lt: float):
    """keys = [(t, zoom, cx, cy)]，t 是本版局部时间。"""
    if lt <= keys[0][0]:
        return keys[0][1:]
    for (t0, z0, x0, y0), (t1, z1, x1, y1) in zip(keys, keys[1:]):
        if t0 <= lt <= t1:
            k = in_out_cubic((lt - t0) / max(1e-6, t1 - t0))
            return (z0 + (z1 - z0) * k, x0 + (x1 - x0) * k, y0 + (y1 - y0) * k)
    return keys[-1][1:]


BAR_H = 108          # 底部字幕条高度：画面只用上面 H-BAR_H 这块
HA = H - BAR_H       # 实机画面的高度（1080 - 108 = 972）


class AppFootage:
    """一段实机画面。src0/src1 是采集时间线里的起止；keys 是镜头；caps 是字幕。"""

    def __init__(self, title: str, src0: float, src1: float, keys, caps, *,
                 focus=(), hud_name: str = "实机"):
        self.title = title
        self.src0 = float(src0)
        self.src1 = float(src1)
        self.keys = keys
        self.caps = caps
        self.focus = focus
        self.hud_name = hud_name

    def _plate(self, lt: float):
        """按高度定裁切窗口：z=1 是整幅高，z=1.32 时源像素和成片 1:1。"""
        z, cx, cy = _lerp_keys(self.keys, lt)
        im = _src_image(self.src0 + lt)
        ch = SH / z
        cw = ch * W / H
        x0 = clamp(cx * SW - cw / 2, 0, SW - cw)
        y0 = clamp(cy * SH - ch / 2, 0, SH - ch)
        crop = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        if crop.size != (W, H):
            crop = crop.resize((W, H), Image.LANCZOS)
        return srgb_to_linear(np.asarray(crop)), x0, y0, z

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        img, x0, y0, z = self._plate(lt)
        cv.rgb = img

        sx = W / (SH / z * W / H)       # 源像素 → 屏幕像素（x/y 同比）

        def to_screen(px: float, py: float) -> tuple[float, float]:
            return ((px - x0) * sx, (py - y0) * sx)

        # ---- 焦点框
        for f0, f1, rect in self.focus:
            a = ramp(lt, f0, f0 + 0.3) * (1 - ramp(lt, f1, f1 + 0.25))
            if a <= 0.01:
                continue
            vx0, vy0, vx1, vy1 = x0, y0, x0 + W / z, y0 + H / z
            # 只画"整只框都在画面里"的那种：越界的框会只剩一只角挂在边上，像脏点
            if not (vx0 + 12 <= rect[0] and rect[2] <= vx1 - 12
                    and vy0 + 12 <= rect[1] and rect[3] <= vy1 - 12):
                continue
            sx0, sy0 = to_screen(rect[0], rect[1])
            sx1, sy1 = to_screen(rect[2], rect[3])
            arm = 26.0
            box = Layer()
            for (cx_, cy_, dx, dy) in ((sx0, sy0, 1, 1), (sx1, sy0, -1, 1),
                                       (sx0, sy1, 1, -1), (sx1, sy1, -1, -1)):
                box.line([(cx_, cy_ + dy * arm), (cx_, cy_), (cx_ + dx * arm, cy_)],
                         width=3, v=255)
            cv.paint(box, "signal", 0.98 * a, mode="over")

        # ---- 指针 + 点击涟漪
        clicks = _manifest().get("clicks") or []
        cx_, cy_ = _keys_at(clicks, self.src0 + lt)
        px, py = to_screen(cx_, cy_)
        if px > 4 and py > 4:
            shape = [(px, py), (px, py + 30), (px + 8, py + 22), (px + 14, py + 36),
                     (px + 21, py + 33), (px + 15, py + 19), (px + 25, py + 18)]
            arrow = Layer()
            arrow.poly(shape, v=255)
            cv.paint(arrow, "ink", 0.88, mode="over")
            outline = Layer()
            outline.line(shape + [shape[0]], width=2, v=255)
            cv.paint(outline, "bone", 0.85, mode="over")

        for k in clicks:
            d = (self.src0 + lt) - k["t"]
            if not (0 <= d <= 0.7):
                continue
            sx, sy = to_screen(k["x"], k["y"])
            a = 1 - out_quint(clamp(d / 0.7))
            ring = Layer()
            ring.circle(sx, sy, 12 + 46 * out_expo(clamp(d / 0.7)), v=255, width=2)
            ring.circle(sx, sy, 6 + 20 * out_expo(clamp(d / 0.7)), v=255, width=2)
            cv.paint(ring, "signal", 0.90 * a, mode="over")
            if k.get("label") and d < 0.5:
                lay = Layer()
                lay.text_mixed((sx + 30, sy + 40), k["label"], size=22, v=255, tracking=1.0)
                cv.paint(lay, "signal", 0.95 * (1 - ramp(d, 0.34, 0.52)), mode="over")

        # ---- 底部字幕条：不透明压住应用自己的状态栏
        bar = Layer()
        bar.rect((0, HA, W, H), v=255)
        cv.paint(bar, "ink", 1.0, mode="over")
        rule = Layer()
        rule.line([(0, HA), (W, HA)], width=1, v=255)
        rule.line([(0, HA), (W * clamp(ctx["t"] / ctx["dur"]), HA)], width=3, v=255)
        cv.paint(rule, "signal", 0.9, mode="over")

        cur = None
        for t0, main, sub in self.caps:
            if lt >= t0:
                cur = (t0, main, sub)
        if cur is not None:
            t0, main, sub = cur
            a = ramp(lt, t0, t0 + 0.32) * (1 - ramp(lt, t0 + 3.7, t0 + 4.1))
            if a > 0.01:
                tick = Layer()
                tick.rect((44, HA + 16, 51, HA + 88), v=255, radius=2)
                cv.paint(tick, "signal", a, mode="over")
                lay = Layer()
                lay.text_mixed((76, HA + 50), main, size=36, v=255, tracking=1.0)
                cv.paint(lay, "bone", a, mode="over")
                if sub:
                    lay2 = Layer()
                    lay2.text_mixed((76, HA + 86), sub, size=22, v=255, tracking=0.8)
                    cv.paint(lay2, "ash", 0.92 * a, mode="over")

        # ---- 时间码：字幕条右端
        tc = (f"{int(ctx['t']) // 60}:{int(ctx['t']) % 60:02d}"
              f".{int(round((ctx['t'] % 1) * 10)) % 10}")
        lay = Layer()
        lay.text((W - 44, HA + 62), tc, font("mono_b", 24), v=255, anchor="rs", tracking=1.6)
        cv.paint(lay, "bone", 0.95, mode="over")
        tw = text_width(tc, font("mono_b", 24), tracking=1.6)
        lay2 = Layer()
        lay2.text_mixed((W - 44 - tw - 26, HA + 62), self.hud_name, size=21, v=255,
                        anchor="rs", tracking=1.2)
        cv.paint(lay2, "ash", 0.90, mode="over")

        return {"bloom": 0.0, "threshold": 3.0, "halation": 0.0, "grain": 0.007,
                "vig": 0.14, "ca": 0.3, "exposure": 1.0}