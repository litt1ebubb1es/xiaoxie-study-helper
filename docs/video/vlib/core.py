"""小蟹学习助手 · 介绍短片 —— 渲染核心

设计原则（照 TREATMENT.md 的规矩，换成橙色主题）：
  · 画面是时间的纯函数：同一帧永远渲染成同一张图，预览/导出逐像素一致；
  · 线性 HDR 合成，只有 signal 橙允许发光超过 0.85，骨白字永远不发光；
  · 调色板封闭：ink / bone / signal 三色 + ember/blood 做橙的层次，纸张版面用暖羊皮；
  · 运动：强缓动（outExpo / inOutCubic）、起步快收尾稳，切点落在拍子上。
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
SAFE = 96

# ---------------------------------------------------------------- 调色板


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: np.ndarray | float):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def hex_lin(h: str) -> np.ndarray:
    h = h.lstrip("#")
    v = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    return np.array([srgb_to_linear(float(x)) for x in v], np.float32)


HEX = {
    "ink": "#0A0A0B",        # 背景黑（微暖）
    "ink2": "#151517",       # 抬起来的黑：面板、暗纸
    "graphite": "#5E5B57",   # 次要线、压暗的字
    "ash": "#9C978F",        # 中间灰
    "bone": "#EEE9DF",       # 纸白 / 主字
    "signal": "#FF4D12",     # 危险橙：火花、卡点、关键数值
    "ember": "#FF8A3D",      # 更热的橙：核心、边缘光
    "blood": "#C21D0B",      # 橙的暗部
    "paper": "#F6EAD2",      # 羊皮纸（和 PPT 同一张纸）
    "paper2": "#FCF4E4",     # 纸面高光
    "paper_ink": "#2A1F16",  # 纸上的字
    "paper_ink2": "#5A4A38",
    "accent_ink": "#8F3407",  # 纸上小字用的深橙（对比度 6.6:1）
}
LIN = {k: hex_lin(v) for k, v in HEX.items()}
C = LIN

# ---------------------------------------------------------------- 缓动


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def out_expo(p: float) -> float:
    p = clamp(p)
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def out_quint(p: float) -> float:
    p = clamp(p)
    return 1 - (1 - p) ** 5


def in_out_cubic(p: float) -> float:
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def out_back(p: float, s: float = 1.7) -> float:
    p = clamp(p)
    return 1 + (s + 1) * (p - 1) ** 3 + s * (p - 1) ** 2


def smooth(p: float) -> float:
    p = clamp(p)
    return p * p * (3 - 2 * p)


def ramp(t: float, a: float, b: float, ease=out_expo) -> float:
    """t 在 [a,b] 上的进度，带缓动。"""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return ease((t - a) / (b - a))


def pulse(t: float, at: float, decay: float = 0.18) -> float:
    """at 这一下之后按指数衰减的脉冲（0..1），用来给"命中"做呼吸。"""
    d = t - at
    if d < 0:
        return 0.0
    return math.exp(-d / decay)


# ---------------------------------------------------------------- 字体

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    "cjk_b": "msyhbd.ttc",     # 中文粗体（主字）
    "cjk": "msyh.ttc",         # 中文常规
    "mono": "CascadiaMono.ttf",  # 机器：标签、时间码、注释
    "mono_b": "consolab.ttf",
    "din": "bahnschrift.ttf",  # 拉丁展示字（紧凑无衬线）
    "impact": "impact.ttf",
}


@functools.lru_cache(maxsize=256)
def font(key: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[key]), px)


def text_width(s: str, f: ImageFont.FreeTypeFont, tracking: float = 0.0) -> float:
    return sum(f.getlength(ch) for ch in s) + tracking * max(0, len(s) - 1)


def text_height(f: ImageFont.FreeTypeFont) -> float:
    a, d = f.getmetrics()
    return a + d


NO_BREAK_BEFORE = "，。、；：？！）】」』%,.;:?!)]}"
NO_BREAK_AFTER = "（【「『([{"


def wrap(s: str, f: ImageFont.FreeTypeFont, max_w: float, tracking: float = 0.0) -> list[str]:
    """按真实字宽折行；中文任意断，但不许把标点甩到行首。"""
    lines: list[str] = []
    cur = ""
    for ch in s:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        trial = cur + ch
        if text_width(trial, f, tracking) <= max_w or not cur:
            cur = trial
            continue
        if ch in NO_BREAK_BEFORE and cur:
            # 标点跟着上一行走（略超宽也认）
            cur = trial
            continue
        if cur[-1] in NO_BREAK_AFTER:
            lines.append(cur[:-1])
            cur = cur[-1] + ch
            continue
        lines.append(cur)
        cur = ch
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- 画布 / 图层


class Layer:
    """一张 L 掩膜，把同一颜色的图元攒在一起，最后一次性合成 —— 省 rasterize 次数。"""

    def __init__(self, w: int = W, h: int = H) -> None:
        self.img = Image.new("L", (w, h), 0)
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self._dirty = False

    # -- 图元 ------------------------------------------------------------
    def line(self, pts, width: float = 2.0, v: int = 255) -> None:
        if len(pts) >= 2:
            self.d.line([tuple(p) for p in pts], fill=v, width=max(1, int(round(width))), joint="curve")
            self._dirty = True

    def polyline_fade(self, pts, width: float = 2.0, v: int = 255) -> None:
        """逐段画，用来做"线条自己长出来"的效果（调用方自己截断 pts）。"""
        self.line(pts, width, v)

    def rect(self, box, v: int = 255, width: int = 0, radius: float = 0) -> None:
        x0, y0, x1, y1 = box
        if radius > 0:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=(v if width == 0 else None),
                                     outline=(v if width else None), width=width)
        else:
            self.d.rectangle([x0, y0, x1, y1], fill=(v if width == 0 else None),
                             outline=(v if width else None), width=width)
        self._dirty = True

    def circle(self, cx, cy, r, v: int = 255, width: int = 0) -> None:
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(v if width == 0 else None),
                       outline=(v if width else None), width=width)
        self._dirty = True

    def poly(self, pts, v: int = 255) -> None:
        self.d.polygon([tuple(p) for p in pts], fill=v)
        self._dirty = True

    def text(self, xy, s: str, f: ImageFont.FreeTypeFont, v: int = 255,
             anchor: str = "ls", tracking: float = 0.0, progress: float = 1.0,
             soft: float = 0.0) -> None:
        """逐字排（带字距），progress<1 时按字数"写出来"。"""
        if not s:
            return
        x, y = xy
        n = len(s)
        shown = progress * n
        if tracking == 0.0 and progress >= 1.0:
            self.d.text((x, y), s, font=f, fill=v, anchor=anchor)
            self._dirty = True
            return
        # 逐字排之前先把"整行宽度"算出来，否则右对齐/居中的锚点会逐字各对齐一次（糊成一团）
        total = sum(f.getlength(c) for c in s) + tracking * max(0, n - 1)
        if anchor[0] == "r":
            x -= total
        elif anchor[0] == "m":
            x -= total / 2
        for i, ch in enumerate(s):
            if i >= shown:
                break
            a = clamp(shown - i)
            self.d.text((x, y), ch, font=f, fill=int(v * a), anchor="l" + anchor[1])
            x += f.getlength(ch) + tracking
        self._dirty = True

    def text_mixed(self, xy, s: str, size: int = 24, v: int = 255, anchor: str = "ls",
                   tracking: float = 1.4, latin: str = "mono", cjk: str = "cjk",
                   progress: float = 1.0) -> None:
        """中英混排：CJK 走中文字体，其余走 mono —— 避免等宽字体缺字变豆腐块。"""
        if not s:
            return
        def f_of(ch):
            return font(cjk if ord(ch) > 0x2E7F else latin, size)
        total = sum(f_of(c).getlength(c) for c in s) + tracking * max(0, len(s) - 1)
        x, y = xy
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        shown = progress * len(s)
        for i, ch in enumerate(s):
            if i >= shown:
                break
            a = clamp(shown - i)
            f = f_of(ch)
            self.d.text((x, y), ch, font=f, fill=int(v * a), anchor="l" + anchor[1])
            x += f.getlength(ch) + tracking
        self._dirty = True

    # -- 输出 ------------------------------------------------------------
    def array(self) -> np.ndarray:
        if not self._dirty:
            return np.zeros((self.h, self.w), np.float32)
        return np.asarray(self.img, np.float32) / 255.0


# ---------------------------------------------------------------- 画布合成


class Canvas:
    def __init__(self, w: int = W, h: int = H, bg: str = "ink") -> None:
        self.w, self.h = w, h
        self.rgb = np.ones((h, w, 3), np.float32) * C[bg][None, None, :]

    def paint(self, layer: Layer, color: str | np.ndarray, opacity: float = 1.0,
              mode: str = "over", v: float = 1.0) -> None:
        if opacity <= 0.002:
            return
        a = layer.array() * (opacity * v)
        if not a.any():
            return
        col = C[color] if isinstance(color, str) else color
        a3 = a[..., None]
        if mode == "over":
            self.rgb *= 1 - a3
            self.rgb += col[None, None, :] * a3
        elif mode == "add":
            self.rgb += col[None, None, :] * a3
        elif mode == "mul":       # 压暗：纸上的墨线、阴影
            self.rgb *= 1 - a3 * (1 - col[None, None, :])
        else:
            raise ValueError(mode)

    def paint_mask(self, mask: np.ndarray, color: str | np.ndarray, opacity: float = 1.0,
                   mode: str = "add") -> None:
        a3 = (mask * opacity)[..., None]
        col = C[color] if isinstance(color, str) else color
        if mode == "add":
            self.rgb += col[None, None, :] * a3
        elif mode == "over":
            self.rgb *= 1 - a3
            self.rgb += col[None, None, :] * a3
        elif mode == "mul":
            self.rgb *= 1 - a3 * (1 - col[None, None, :])


# ---------------------------------------------------------------- 后处理

_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
_GRAIN_FIELD: np.ndarray | None = None
_VIG = None


def vignette_map(strength: float = 0.35) -> np.ndarray:
    global _VIG
    if _VIG is None:
        cx, cy = W / 2, H / 2
        r = np.sqrt(((_XX - cx) / cx) ** 2 + ((_YY - cy) / cy) ** 2) / math.sqrt(2)
        _VIG = np.clip(r, 0, 1)
    return 1 - (_VIG ** 2.2) * strength


def shoulder(x: np.ndarray, k: float = 0.72) -> np.ndarray:
    """0.72 以下线性，以上指数压；极亮处去饱和到白（跟 pdoom 同一条色调肩）。"""
    y = np.where(x <= k, x, k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k))))
    over = x.max(axis=2, keepdims=True)
    w = np.clip((over - 2.0) / 10.0, 0, 1) * 0.85
    return y * (1 - w) + w


def _down(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[:2]
    return a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3))


def _up(a: np.ndarray, size) -> np.ndarray:
    img = Image.fromarray(a.astype(np.float32), mode="F" if a.ndim == 2 else None) if a.ndim == 2 else None
    if img is not None:
        return np.asarray(img.resize(size, Image.BILINEAR), np.float32)
    chans = [np.asarray(Image.fromarray(a[..., i].astype(np.float32)).resize(size, Image.BILINEAR), np.float32)
             for i in range(a.shape[2])]
    return np.stack(chans, -1)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if a.ndim == 2:
        return gaussian_filter(a, sigma, mode="nearest")
    return np.stack([gaussian_filter(a[..., i], sigma, mode="nearest") for i in range(a.shape[2])], -1)


def finish(cv: Canvas, frame_idx: int, *, bloom: float = 0.55, threshold: float = 0.85,
           halation: float = 0.25, grain: float = 0.020, vig: float = 0.35,
           ca: float = 1.2, flash: float = 0.0, fade: float = 0.0,
           exposure: float = 1.0, shake=(0, 0), zoom: float = 1.0) -> np.ndarray:
    """线性 HDR → 8bit sRGB。全部是确定性操作（噪声用帧号做种子）。"""
    x = cv.rgb * exposure

    # -- bloom：只让亮处发光 -------------------------------------------------
    if bloom > 0:
        small = _down(x, 4)
        lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        m = np.clip((lum - threshold) / max(1e-3, 1.0 - threshold), 0, None)[..., None]
        glow = _blur(small * m, 6.0)
        rgb_g = _up(glow, (W, H))
        x = x + rgb_g * bloom
        if halation > 0:
            warm = _blur(small * m, 16.0) * np.array([1.0, 0.30, 0.10], np.float32)[None, None, :]
            x = x + _up(warm, (W, H)) * halation

    x = shoulder(x)
    x = x + flash

    # -- 色彩分叉（整体近似径向色散，便宜够看） ------------------------------
    if ca > 0.05:
        d = int(round(ca))
        if d >= 1:
            x[..., 0] = np.roll(x[..., 0], d, axis=1)
            x[..., 2] = np.roll(x[..., 2], -d, axis=1)

    img = linear_to_srgb(x) if False else linear_to_srgb(np.clip(x, 0, 1))
    out = (img * 255.0)

    # -- 暗角 / 抖动颗粒 / 褪黑 -------------------------------------------------
    if vig > 0:
        out *= vignette_map(vig)[..., None]
    if grain > 0:
        global _GRAIN_FIELD
        if _GRAIN_FIELD is None:
            _GRAIN_FIELD = np.random.default_rng(0xC0FFEE).standard_normal((H, W), dtype=np.float32)
        out += (_GRAIN_FIELD * grain * 255.0)[..., None]
    if fade > 0:
        out *= max(0.0, 1.0 - fade)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # -- 镜头：先缩放再抖，边缘外扩用黑（安全的 crop） ---------------------------
    if abs(zoom - 1.0) > 1e-4 or shake != (0, 0):
        im = Image.fromarray(out)
        if abs(zoom - 1.0) > 1e-4:
            nw, nh = int(W / zoom), int(H / zoom)
            x0, y0 = (W - nw) // 2, (H - nh) // 2
            im = im.crop((x0, y0, x0 + nw, y0 + nh)).resize((W, H), Image.BICUBIC)
        sx, sy = int(round(shake[0])), int(round(shake[1]))
        if sx or sy:
            im = Image.fromarray(np.roll(np.asarray(im), (sy, sx), axis=(0, 1)))
        out = np.asarray(im)
    return out