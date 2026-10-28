"""八个版面（plate）。每个 plate 是"时间的纯函数"：render(cv, lt, ctx) 只依赖 lt。

共用语法：ink 场 / 纸场交替；骨白大字做判断；signal 橙只给"最要紧的那一下"；
mono 中英混排小字做机器口吻的注脚；页脚一条细进度线。
"""
from __future__ import annotations

import math

import numpy as np

from .core import (C, H, SAFE, W, Canvas, Layer, clamp, font, in_out_cubic, out_back, out_expo,
                   out_quint, pulse, ramp, smooth, text_width, wrap)

MARGIN = SAFE


# ---------------------------------------------------------------- 共用小件


def _gauss(x: float, mu: float, sig: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sig * sig))


def spark(cv: Canvas, x: float, y: float, r: float = 26.0, inten: float = 1.0,
          color: str = "ember") -> None:
    """一个橙色光点：核心炽白、外层橙晕，加法混合，允许过曝。"""
    if inten <= 0.004:
        return
    x0, x1 = int(max(0, x - r * 2.2)), int(min(W, x + r * 2.2))
    y0, y1 = int(max(0, y - r * 2.2)), int(min(H, y + r * 2.2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
    core = np.clip(1 - d, 0, 1) ** 3
    halo = np.clip(1 - d / 2.2, 0, 1) ** 2.2
    cv.rgb[y0:y1, x0:x1] += (C[color] * halo[..., None] * 0.5 + np.float32(1.7) * core[..., None]) * inten


def hairline(layer: Layer, x0: float, x1: float, y: float, v: int = 255, w: float = 1.0) -> None:
    layer.line([(x0, y), (x1, y)], width=w, v=v)


def hatch(layer: Layer, box, spacing: float = 7.0, angle: float = 45.0, v: int = 255,
          w: float = 1.0) -> None:
    """蚀刻排线；线距细到 3px 以下自动减淡，避免摩尔纹。"""
    x0, y0, x1, y1 = box
    fade = clamp((spacing - 2.2) / 2.0, 0.0, 1.0)
    if fade <= 0.02:
        return
    vv = int(v * (0.35 + 0.65 * fade))
    diag = math.hypot(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = math.radians(angle)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    k = 0
    while k * spacing < diag:
        off = k * spacing - diag / 2
        px, py = cx + nx * off, cy + ny * off
        layer.line([(px - dx * diag / 2, py - dy * diag / 2), (px + dx * diag / 2, py + dy * diag / 2)],
                   width=w, v=vv)
        k += 1


def mono(cv: Canvas, s: str, xy, size: int = 22, color: str = "ash", opacity: float = 1.0,
         tracking: float = 1.2, anchor: str = "ls") -> None:
    lay = Layer()
    lay.text_mixed(xy, s, size=size, v=255, anchor=anchor, tracking=tracking)
    cv.paint(lay, color, opacity, mode="over")


def scrim(cv: Canvas, box, color: str = "ink", opacity: float = 0.82, radius: float = 6) -> None:
    lay = Layer()
    lay.rect(box, v=255, radius=radius)
    cv.paint(lay, color, opacity, mode="over")


def hud(cv: Canvas, t_abs: float, dur: float, plate: str, *, dark: bool = True,
        opacity: float = 0.6) -> None:
    """t_abs 是整片时间（不是本版面的局部时间），右下角时间码和底部进度线都按它走。"""
    ink = C["bone"] if dark else C["paper_ink"]
    lay = Layer()
    lay.text_mixed((MARGIN, H - 56), "小蟹学习助手  /  本地运行", size=20, v=255, tracking=1.4)
    lay.text_mixed((MARGIN, H - 30), plate, size=20, v=255, tracking=1.4)
    mm, ss = divmod(int(t_abs), 60)
    dm, ds = divmod(int(dur), 60)
    tc = f"{mm}:{ss:02d}.{int(t_abs % 1 * 10)}  /  {dm}:{ds:02d}.0"
    lay.text((W - MARGIN, H - 30), tc, font("mono", 20), v=255, anchor="rs", tracking=1.2)
    cv.paint(lay, ink, opacity, mode="over")
    rule = Layer()
    hairline(rule, MARGIN, W - MARGIN, H - 74, v=90, w=1.0)
    hairline(rule, MARGIN, MARGIN + (W - 2 * MARGIN) * clamp(t_abs / dur), H - 74, v=255, w=1.6)
    cv.paint(rule, ink, opacity * 0.5, mode="over")


def big_block(cv: Canvas, lines, x: float, y: float, size: int, color: str = "bone",
              opacity: float = 1.0, progress: float = 1.0, lh: float = 1.16,
              tracking: float = 0.0, fkey: str = "cjk_b") -> None:
    if not lines:
        return
    f = font(fkey, size)
    shown = progress * len(lines)
    lay = Layer()
    for i, s in enumerate(lines):
        a = clamp(shown - i)
        if a <= 0:
            break
        lay.text((x, y + i * size * lh), s, f, v=255, anchor="ls", tracking=tracking, progress=a)
    cv.paint(lay, color, opacity, mode="over")


def title_block(cv: Canvas, text: str, *, y: float = 190, size: int = 92, max_w: float = 1500,
                progress: float = 1.0, color: str = "bone", t: float = 0.0, sub: str = "",
                sub_size: int = 26, sub_color: str = "ash", sub_at: float = 0.45,
                sub_op: float = 1.0, sub_gap: float = 1.5) -> float:
    """一行判断 + 机器口吻的副标。标题折成两行时，副标自己往下让，不用手改坐标。"""
    lines, size = fit_lines(text, size, max_w)
    big_block(cv, lines, MARGIN, y, size, color, 1.0, progress)
    last_base = y + (len(lines) - 1) * size * 1.16
    if sub:
        mono(cv, sub, (MARGIN + 4, last_base + size * sub_gap), size=sub_size, color=sub_color,
             opacity=ramp(t, sub_at, sub_at + 0.55) * sub_op)
    return last_base


def fit_lines(s: str, size: int, max_w: float, max_lines: int = 3, tracking: float = 0.0):
    while size > 24:
        ls = wrap(s, font("cjk_b", size), max_w, tracking)
        if len(ls) <= max_lines:
            return ls, size
        size -= 4
    return wrap(s, font("cjk_b", size), max_w, tracking), size


# ---------------------------------------------------------------- 纸场

_PAPER_BG = None


def inv_shoulder(x, k: float = 0.72):
    """色调肩的反函数：纸色先反向抬高，过完肩正好落回目标色（否则纸会发灰）。"""
    x = np.asarray(x, np.float32)
    return np.where(x <= k, x, k - (1 - k) * np.log(np.clip(1 - (x - k) / (1 - k), 1e-4, 1.0)))


def _up2(a: np.ndarray, size) -> np.ndarray:
    from PIL import Image as _I
    return np.asarray(_I.fromarray(a.astype(np.float32)).resize(size, _I.BICUBIC), np.float32)


def paper_field() -> np.ndarray:
    global _PAPER_BG
    if _PAPER_BG is None:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(7)
        fiber = _up2(gaussian_filter(rng.standard_normal((H // 4, W // 4), dtype=np.float32), 1.2), (W, H))
        stain = _up2(gaussian_filter(rng.standard_normal((H // 16, W // 16), dtype=np.float32), 6.0), (W, H))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
        base = inv_shoulder(C["paper"])[None, None, :] * (1 - 0.055 * r[..., None] ** 1.6)
        img = base * (1 + fiber[..., None] * 0.010 + stain[..., None] * 0.014)
        _PAPER_BG = img.astype(np.float32)
    return _PAPER_BG


PAPER_POST = {"bloom": 0.0, "threshold": 4.0, "halation": 0.0, "grain": 0.010,
              "vig": 0.10, "ca": 0.0, "exposure": 1.0}


# ---------------------------------------------------------------- P1 冷开场


class ColdOpen:
    """黑场里，一个橙色火花把标题写出来；写完，机器口吻的副标落下去。"""

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        sy = H * 0.46
        sx = MARGIN + (W - 2 * MARGIN) * 0.985 * out_expo(clamp(lt / 1.55))

        trail = Layer()
        trail.line([(MARGIN, sy), (sx, sy)], width=2, v=255)
        cv.paint(trail, "bone", 0.5 * (1 - ramp(lt, 1.7, 2.4)), mode="over")

        ticks = Layer()
        for i in range(0, 97):
            x = MARGIN + (W - 2 * MARGIN) * i / 96
            if x > sx:
                break
            hgt = 26 if i % 4 == 0 else 13
            ticks.line([(x, sy - hgt), (x, sy + hgt)], width=1.5, v=255)
        cv.paint(ticks, "graphite", 0.85, mode="over")

        lay = Layer()
        lay.text((MARGIN, sy - 92), "小蟹学习助手", font("cjk_b", 168), v=255,
                 tracking=8, progress=ramp(lt, 0.35, 1.55, out_quint))
        cv.paint(lay, "bone", 1.0, mode="over")

        if lt > 1.7:
            mono(cv, "不录屏 · 可本地 · 记得住你卡在哪一秒", (MARGIN + 6, sy + 96), size=28,
                 color="ash", opacity=ramp(lt, 1.7, 2.35))

        # 收尾一拍：标题下的橙线自己长出来
        if lt > 2.0:
            ul = Layer()
            w = 380 * ramp(lt, 2.0, 2.7, out_expo)
            ul.line([(MARGIN + 6, sy + 150), (MARGIN + 6 + w, sy + 150)], width=4, v=255)
            cv.paint(ul, "signal", 0.95, mode="add")

        spark(cv, sx, sy, r=24, inten=1.0 * (1 - ramp(lt, 2.3, 2.9)) + 0.18)
        hud(cv, ctx["t"], ctx["dur"], "开场", dark=True, opacity=0.32 * ramp(lt, 1.2, 2.0))
        return {"zoom": 1.0 + 0.05 * (1 - out_expo(clamp(lt / 1.6))), "bloom": 0.62}


# ---------------------------------------------------------------- P2 问题


def _level(u: float) -> float:
    base = 0.84 + 0.05 * math.sin(u * 5.0 + 0.4) + 0.03 * math.sin(u * 13.0 + 0.8)
    return clamp(base - 0.50 * _gauss(u, 0.235, 0.050) - 0.29 * _gauss(u, 0.615, 0.042), 0.06, 1.0)


class Problem:
    """时间轴成为主体：理解度曲线画出来，两处掉线被钉住。"""

    X0, X1 = MARGIN, W - MARGIN
    Y_TOP, Y_BOT, AXIS_Y = 330.0, 660.0, 716.0

    def _pt(self, u: float):
        return (self.X0 + (self.X1 - self.X0) * u,
                self.Y_TOP + (1 - _level(u)) * (self.Y_BOT - self.Y_TOP))

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        # 轴 + 刻度（0/5/10/15/20/25 分钟）
        axis = Layer()
        hairline(axis, self.X0, self.X1, self.AXIS_Y, v=150, w=1.4)
        for i in range(0, 151):
            x = self.X0 + (self.X1 - self.X0) * i / 150
            hgt = 24 if i % 30 == 0 else (13 if i % 10 == 0 else 6)
            axis.line([(x, self.AXIS_Y), (x, self.AXIS_Y + hgt)], width=1.2, v=125)
        cv.paint(axis, "graphite", 0.8, mode="over")

        marks = Layer()
        for k in range(6):
            x = self.X0 + (self.X1 - self.X0) * k / 5
            lab = "0:00" if k == 0 else f"{k * 5:02d}:00"
            anch = "ls" if k == 0 else ("rs" if k == 5 else "ms")
            marks.text((x, self.AXIS_Y + 58), lab, font("mono", 21), v=255, anchor=anch, tracking=0.8)
        cv.paint(marks, "graphite", 0.95, mode="over")

        # 曲线 + 面积
        p = ramp(lt, 0.1, 1.35, in_out_cubic)
        n = 480
        pts = [self._pt(i / (n - 1)) for i in range(max(2, int(n * p)))]
        fill = Layer()
        fill.poly(pts + [(pts[-1][0], self.AXIS_Y), (pts[0][0], self.AXIS_Y)], v=255)
        cv.paint(fill, "graphite", 0.10, mode="add")
        line = Layer()
        line.line(pts, width=3.2, v=255)
        cv.paint(line, "bone", 0.95, mode="over")

        # 两处掉线
        spots = [(0.240, "06:00", "掉了 3 分钟"), (0.613, "15:20", "掉了 8 分钟")]
        for si, (u, tc, note) in enumerate(spots):
            a = ramp(lt, 1.1 + 0.5 * si, 1.45 + 0.5 * si, out_expo)
            if a <= 0.01:
                continue
            x, y = self._pt(u)
            dash = Layer()
            k = 0.0
            while y + 14 + k < self.AXIS_Y - 4:
                y0 = y + 14 + k
                dash.line([(x, y0), (x, min(y0 + 9, self.AXIS_Y - 4))], width=1.5, v=255)
                k += 18
            cv.paint(dash, "signal", 0.40 * a, mode="add")
            pin = Layer()
            pin.circle(x, y, 9, v=255)
            cv.paint(pin, "signal", a, mode="add")
            ring = Layer()
            ring.circle(x, y, 20 + 22 * (1 - a), v=255, width=2)
            cv.paint(ring, "signal", 0.5 * a, mode="add")

            # 标签块：给个暗底，别和曲线打架
            bx0, bx1 = x - 108, x + 108
            scrim(cv, (bx0, y - 96, bx1, y - 30), opacity=0.72 * a, radius=6)
            lab = Layer()
            lab.text_mixed((x, y - 70), tc, size=25, v=255, anchor="ms", tracking=1.0)
            cv.paint(lab, "bone", a, mode="over")
            lab2 = Layer()
            lab2.text_mixed((x, y - 42), note, size=20, v=255, anchor="ms", tracking=0.8)
            cv.paint(lab2, "signal", 0.95 * a, mode="over")

        # 大字判断：两段
        if lt < 3.0:
            lines, size = fit_lines("看网课真正让人放弃的，往往不是难", 92, 1500)
            big_block(cv, lines, MARGIN, 190, size, "bone", 1.0, ramp(lt, 0.15, 1.1))
        else:
            lines, size = fit_lines("而是不知道从哪一秒开始没跟上。", 92, 1500)
            big_block(cv, lines, MARGIN, 190, size, "bone", 1.0, ramp(lt, 3.05, 3.55))

        mono(cv, "横轴：一集 25 分钟的网课    ｜    纵轴：还跟得上的程度", (MARGIN + 4, 838),
             size=24, color="ash", opacity=ramp(lt, 0.9, 1.6))
        mono(cv, "橙点 = 掉线的那一秒，点一下就能跳回那里", (MARGIN + 4, 884),
             size=24, color="signal", opacity=ramp(lt, 3.5, 4.1) * 0.95)

        su = clamp(out_expo(clamp((lt - 0.15) / 3.6))) * 0.80 + 0.04
        sx, sy = self._pt(su)
        spark(cv, sx, sy, r=16, inten=0.7 + 0.3 * ctx["kick"](lt))

        hud(cv, ctx["t"], ctx["dur"], "问题 · 时间轴", dark=True)
        z = 1.0 + 0.05 * pulse(lt, 3.0, 0.32) + 0.03 * smooth(clamp((lt - 3.0) / 2.0))
        return {"zoom": z, "bloom": 0.5 + 0.2 * ctx["kick"](lt)}


# ---------------------------------------------------------------- P3 笔记（纸场）


class Notes:
    """纸场：按 Enter 把这一秒攒下来，卡片落成一列；下课一次性答完。"""

    ITEMS = [("01:58", "为什么梯度为零不一定是最优解？"),
             ("04:46", "边界条件和约束条件是一回事吗？"),
             ("11:12", "这里为什么要用一阶近似？")]

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        cv.rgb = paper_field().copy()
        title_block(cv, "按一下，就把这一秒记下来", y=190, size=92, max_w=1180, color="paper_ink",
                    progress=ramp(lt, 0.05, 0.7), t=lt, sub="ENTER  →  时间戳 + 当时那句原话",
                    sub_color="paper_ink2")

        # 键帽
        press = 1 - 0.10 * (pulse(lt, 1.0, 0.10) + pulse(lt, 2.5, 0.10) + pulse(lt, 4.0, 0.10))
        kx, ky, kw, kh = MARGIN + 6, 560, 250, 96
        cap = Layer()
        cap.rect((kx, ky, kx + kw, ky + kh), v=255, width=3, radius=10)
        hatch(cap, (kx + 10, ky + 10, kx + kw - 10, ky + kh - 10), spacing=7.0, angle=45, v=100)
        cv.paint(cap, "paper_ink", 0.8, mode="mul")
        lab = Layer()
        lab.text((kx + kw / 2, ky + kh * 0.66), "ENTER", font("mono_b", int(34 * press)), v=255,
                 anchor="ms", tracking=4)
        cv.paint(lab, "paper_ink", 0.95, mode="mul")

        # 卡片
        for i, (tc, q) in enumerate(self.ITEMS):
            t0 = 0.95 + i * 1.5
            a = ramp(lt, t0, t0 + 0.45, out_expo)
            if a <= 0.01:
                continue
            cx = 760 + 34 * (1 - a)
            cy = 452 + i * 128 - 26 * (1 - out_back(clamp((lt - t0) / 0.5)))
            cw, chh = 1000, 104
            card = Layer()
            card.rect((cx, cy, cx + cw, cy + chh), v=255, width=2, radius=8)
            card.rect((cx, cy, cx + 9, cy + chh), v=255)
            cv.paint(card, "paper_ink", 0.9 * a, mode="mul")
            lay = Layer()
            lay.text((cx + 34, cy + 66), tc, font("mono", 26), v=255, tracking=1.5)
            lay.text((cx + 150, cy + 66), q, font("cjk_b", 32), v=255)
            cv.paint(lay, "paper_ink", 0.95 * a, mode="mul")

        mono(cv, "已记 3 条 · 都在这一节课里", (MARGIN + 6, 730), size=24, color="paper_ink2",
             opacity=ramp(lt, 4.4, 5.0))
        mono(cv, "不打断听课 / 不手动抄时间", (MARGIN + 6, 776), size=24, color="paper_ink2",
             opacity=ramp(lt, 1.6, 2.2))

        # 全部回答
        if lt > 5.2:
            a = ramp(lt, 5.2, 5.75)
            bx, by, bw2, bh2 = 760, 906, 300, 72
            btn = Layer()
            btn.rect((bx, by, bx + bw2, by + bh2), v=255, radius=36)
            cv.paint(btn, "signal", 0.95 * a, mode="mul")
            lay = Layer()
            lay.text((bx + bw2 / 2, by + 48), "全部回答", font("cjk_b", 34), v=255, anchor="ms",
                     tracking=3)
            cv.paint(lay, "paper", 1.0 * a, mode="over")
            if lt > 6.1:
                mono(cv, "→ 用这一节的上下文逐条答，并标出处", (bx + bw2 + 34, by + 46), size=24,
                     color="accent_ink", opacity=ramp(lt, 6.1, 6.7))

        hud(cv, ctx["t"], ctx["dur"], "笔记 · 问题队列", dark=False)
        return dict(PAPER_POST, zoom=1.0 + 0.02 * pulse(lt, 5.2, 0.3))


# ---------------------------------------------------------------- P4 换讲法


class Retell:
    """五种讲法：讲不通就换一种，不是再听一遍。"""

    CARDS = [("换类比", "它像什么？"), ("补前置", "先补哪块？"), ("数值化", "换个数字？"),
             ("反向验证", "反过来还成立？"), ("分层复述", "一句话说完？")]

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        title_block(cv, "同一个概念，五种讲法", y=200, size=96, max_w=1300,
                    progress=ramp(lt, 0.05, 0.6), t=lt, sub="讲不通就换一种 —— 不是「再听一遍」",
                    sub_gap=1.55)

        cw, gap, chh = 300, 44, 300
        x0 = (W - (cw * 5 + gap * 4)) / 2
        cy = 470
        active = int(lt / 0.5) % 5 if lt > 1.1 else -1
        for i, (lab, cap) in enumerate(self.CARDS):
            t0 = 0.3 + i * 0.15
            a = ramp(lt, t0, t0 + 0.5, out_back)
            if a <= 0.01:
                continue
            x = x0 + i * (cw + gap)
            y = cy - 40 * (1 - a)
            hot = i == active
            if hot:
                fill = Layer()
                fill.rect((x, y, x + cw, y + chh), v=255, radius=14)
                cv.paint(fill, "signal", 0.95 * a, mode="over")
            else:
                box = Layer()
                box.rect((x, y, x + cw, y + chh), v=255, width=2, radius=14)
                cv.paint(box, "graphite", 0.95 * a, mode="over")
            lay = Layer()
            lay.text((x + 26, y + 56), f"0{i + 1}", font("mono", 30), v=255, tracking=2)
            cv.paint(lay, "ink" if hot else "signal", a, mode="over")
            lay2 = Layer()
            lay2.text((x + 26, y + 156), lab, font("cjk_b", 42), v=255)
            cv.paint(lay2, "ink" if hot else "bone", a, mode="over")
            rule = Layer()
            hairline(rule, x + 26, x + 146, y + 200, v=255, w=2)
            cv.paint(rule, "ink" if hot else "graphite", 0.9 * a, mode="over")
            lay3 = Layer()
            lay3.text_mixed((x + 26, y + 244), cap, size=22, v=255, tracking=0.6)
            cv.paint(lay3, "ink" if hot else "ash", 0.9 * a, mode="over")

        mono(cv, "听不懂就换一种，换到讲通为止", (W - MARGIN, 900), size=24, color="ash",
             opacity=ramp(lt, 1.4, 2.0), anchor="rs")

        hud(cv, ctx["t"], ctx["dur"], "重讲 · 五种讲法", dark=True)
        return {"zoom": 1.0 + 0.03 * ctx["kick"](lt), "bloom": 0.55}


# ---------------------------------------------------------------- P5 出题与判分


class Quiz:
    """出题有据：一道 4 选 1 亮起来，一道因为出处查不到被划掉，掌握度随之上升。"""

    OPTS = ["A. 把样本空间映射到实数轴的一个函数", "B. 全部可能结果的集合",
            "C. 随机变量的加权平均", "D. 偏离均值的平方的期望"]

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        title_block(cv, "出题有据，答对才改掌握度", y=190, size=92, max_w=1300,
                    progress=ramp(lt, 0.05, 0.6), t=lt, sub="题干只用这一段原文 / 出处查不到就作废",
                    sub_gap=1.55)

        cx, cy, cw, chh = MARGIN + 4, 430, 1010, 340
        card = Layer()
        card.rect((cx, cy, cx + cw, cy + chh), v=255, width=2, radius=12)
        cv.paint(card, "graphite", 0.9, mode="over")

        q = Layer()
        q.text((cx + 40, cy + 70), "1. 按原文，随机变量是什么？", font("cjk_b", 38), v=255,
               progress=ramp(lt, 0.3, 1.05))
        cv.paint(q, "bone", 0.95, mode="over")

        picked = 0
        for i, o in enumerate(self.OPTS):
            t0 = 0.85 + i * 0.2
            a = ramp(lt, t0, t0 + 0.4)
            if a <= 0.01:
                continue
            y = cy + 136 + i * 58
            on = (i == picked) and lt > 2.5
            if on:
                hi = Layer()
                hi.rect((cx + 34, y - 30, cx + 34 + 830, y + 16), v=255, radius=6)
                cv.paint(hi, "signal", 0.16 * ramp(lt, 2.5, 2.85), mode="add")
            lay = Layer()
            lay.text((cx + 44, y), o, font("cjk", 30), v=255)
            cv.paint(lay, "bone" if on else "ash", a, mode="over")
            if on:
                ul = Layer()
                ul.line([(cx + 44, y + 10), (cx + 44 + 560, y + 10)], width=3, v=255)
                cv.paint(ul, "signal", 0.85, mode="add")

        # 答对：勾
        if lt > 2.9:
            a = ramp(lt, 2.9, 3.2, out_back)
            gx, gy = cx + cw - 84, cy + 136 + picked * 58 - 6
            circ = Layer()
            circ.circle(gx + 6, gy - 6, 24, v=255, width=2)
            cv.paint(circ, "signal", 0.7 * a, mode="add")
            chk = Layer()
            chk.line([(gx - 8, gy - 6), (gx + 2, gy + 5), (gx + 20, gy - 18)], width=5, v=255)
            cv.paint(chk, "signal", a, mode="add")

        # 出处卡（右栏）
        if lt > 1.15:
            a = ramp(lt, 1.15, 1.6)
            sx0, sy0 = W - MARGIN - 520, 300
            box = Layer()
            box.rect((sx0, sy0, sx0 + 500, sy0 + 128), v=255, width=2, radius=10)
            box.rect((sx0, sy0, sx0 + 6, sy0 + 128), v=255)
            cv.paint(box, "graphite", 0.9 * a, mode="over")
            lab = Layer()
            lab.text_mixed((sx0 + 30, sy0 + 46), "出处 11:12", size=24, v=255, tracking=1.2)
            cv.paint(lab, "signal", 0.95 * a, mode="over")
            lab2 = Layer()
            lab2.text((sx0 + 30, sy0 + 96), "「随机变量是…把样本空间映到实数」", font("cjk", 24), v=255)
            cv.paint(lab2, "bone", 0.85 * a, mode="over")

        # 作废卡
        if lt > 3.4:
            a = ramp(lt, 3.4, 3.8)
            bx, by = MARGIN + 4, 804
            box = Layer()
            box.rect((bx, by, bx + 1010, by + 88), v=255, width=2, radius=10)
            cv.paint(box, "graphite", 0.85 * a, mode="over")
            lay3 = Layer()
            lay3.text((bx + 36, by + 56), "3. 引文在这一节原文里找不到", font("cjk", 30), v=255)
            cv.paint(lay3, "ash", 0.85 * a, mode="over")
            if lt > 3.75:
                strike = Layer()
                sp = ramp(lt, 3.75, 4.2, in_out_cubic)
                strike.line([(bx + 30, by + 44), (bx + 30 + 470 * sp, by + 44)], width=4, v=255)
                cv.paint(strike, "blood", 0.9, mode="add")
            if lt > 4.0:
                mono(cv, "作废 · 不计分 · 不进掌握度", (bx + 540, by + 56), size=25,
                     color="signal", opacity=ramp(lt, 4.0, 4.4) * 0.95)

        # 掌握度
        if lt > 4.3:
            a = ramp(lt, 4.3, 4.8)
            p = out_quint(clamp((lt - 4.6) / 1.8))
            bx, by, bw2 = W - MARGIN - 520, 520, 500
            mono(cv, "掌握度", (bx, by - 26), size=24, color="ash", opacity=a, tracking=2.0)
            bar = Layer()
            bar.rect((bx, by, bx + bw2, by + 44), v=255, width=2, radius=6)
            cv.paint(bar, "graphite", 0.75 * a, mode="over")
            fill = Layer()
            fill.rect((bx + 3, by + 3, bx + 3 + (bw2 - 6) * 0.86 * p, by + 41), v=255, radius=4)
            cv.paint(fill, "signal", 0.95, mode="add")
            num = Layer()
            num.text((bx, by + 136), f"{0.86 * p:.2f}", font("din", 64), v=255, tracking=1)
            cv.paint(num, "bone", a, mode="over")
            mono(cv, "只在答对时上升；答错只换讲法", (bx, by + 196), size=22, color="ash",
                 opacity=a * 0.9)

        hud(cv, ctx["t"], ctx["dur"], "小测 · 判分", dark=True)
        return {"zoom": 1.0 + 0.02 * pulse(lt, 2.9, 0.25) + 0.03 * pulse(lt, 4.6, 0.3), "bloom": 0.52}


# ---------------------------------------------------------------- P6 本地运行（纸场）


class Edge:
    """纸场：一台机器，锁，三个数字 —— 2.6MB / 317KB / 0 台服务器。"""

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        cv.rgb = paper_field().copy()
        title_block(cv, "全部在你这台机器上跑", y=180, size=92, max_w=1150, color="paper_ink",
                    progress=ramp(lt, 0.05, 0.6), t=lt, sub="不录屏 / 不传云 / 不装插件",
                    sub_color="paper_ink2", sub_at=0.4)

        # 笔记本线稿 + 屏幕里的迷你界面
        x0, y0, x1, y1 = W - MARGIN - 560, 250, W - MARGIN - 60, 520
        a_lap = ramp(lt, 0.25, 0.85)
        lap = Layer()
        lap.rect((x0, y0, x1, y1), v=255, width=3, radius=10)
        lap.line([(x0 - 34, y1 + 30), (x1 + 34, y1 + 30)], width=3, v=255)
        lap.line([(x0 - 34, y1 + 30), (x0 - 6, y1)], width=2, v=255)
        lap.line([(x1 + 34, y1 + 30), (x1 + 6, y1)], width=2, v=255)
        cv.paint(lap, "paper_ink", 0.75 * a_lap, mode="mul")

        ui = ramp(lt, 0.8, 1.4)
        if ui > 0.01:
            # 屏幕里的视频区：一个播放三角 + 两行字幕占位
            inner = Layer()
            inner.rect((x0 + 26, y0 + 28, x1 - 26, y0 + 186), v=255, width=2, radius=4)
            cv.paint(inner, "paper_ink", 0.42 * ui, mode="mul")
            play = Layer()
            pcx, pcy = (x0 + x1) / 2, y0 + 100
            play.poly([(pcx - 14, pcy - 20), (pcx - 14, pcy + 20), (pcx + 18, pcy)], v=255)
            cv.paint(play, "paper_ink", 0.30 * ui, mode="mul")
            sub = Layer()
            for k, wd in enumerate([300, 210]):
                cx0 = pcx - wd / 2
                sub.line([(cx0, y0 + 148 + k * 16), (cx0 + wd, y0 + 148 + k * 16)], width=3, v=150)
            cv.paint(sub, "paper_ink", 0.35 * ui, mode="mul")
            # 时间轴 + 三个橙钉
            tl = Layer()
            tl.line([(x0 + 40, y0 + 236), (x1 - 40, y0 + 236)], width=2, v=255)
            for k in range(1, 13):
                xx = x0 + 40 + (x1 - x0 - 80) * k / 13
                tl.line([(xx, y0 + 236), (xx, y0 + 246)], width=1.2, v=255)
            cv.paint(tl, "paper_ink", 0.5 * ui, mode="mul")
            for k, xx in enumerate([0.22, 0.47, 0.71, 0.88]):
                pa = ramp(lt, 1.15 + 0.26 * k, 1.45 + 0.26 * k, out_back)
                if pa <= 0:
                    continue
                px = x0 + 40 + (x1 - x0 - 80) * xx
                pin = Layer()
                pin.circle(px, y0 + 236 - 22 * pa, 8 * pa, v=255)
                pin.line([(px, y0 + 236), (px, y0 + 236 - 18 * pa)], width=2, v=255)
                cv.paint(pin, "signal", 0.95 * pa, mode="mul")

        # 锁
        if lt > 1.1:
            a = ramp(lt, 1.1, 1.5, out_back)
            lx, ly = x1 - 44, y1 + 44
            body = Layer()
            body.rect((lx - 30, ly - 6, lx + 30, ly + 44), v=255, radius=8)
            body.circle(lx, ly + 18, 8, v=0)
            cv.paint(body, "signal", 0.92 * a, mode="mul")
            shackle = Layer()
            shackle.line([(lx - 18, ly - 8), (lx - 18, ly - 24), (lx + 18, ly - 24), (lx + 18, ly - 8)],
                         width=6, v=255)
            cv.paint(shackle, "signal", 0.92 * a, mode="mul")

        # 三个数字
        nums = [("2.6", "MB", "157 秒音频的转轨", 0.0), ("317", "KB", "一集微积分的字幕", 0.45),
                ("0", "台", "要联网的服务器", 0.9)]
        for i, (val, unit, cap, t0) in enumerate(nums):
            a = ramp(lt, t0 + 0.15, t0 + 0.6, out_expo)
            if a <= 0.01:
                continue
            bx, by = MARGIN + 6 + i * 420, 700
            k = out_quint(clamp((lt - t0 - 0.15) / 1.1))
            shown = f"{float(val) * k:.1f}" if val == "2.6" else f"{float(val) * k:.0f}"
            num = Layer()
            num.text((bx, by), shown, font("din", 96), v=255, tracking=0)
            wnum = text_width(shown, font("din", 96))
            num.text_mixed((bx + wnum + 8, by), unit, size=44, v=255, tracking=1.0, latin="din")
            cv.paint(num, "accent_ink" if a > 0.5 else "paper_ink", a, mode="mul")
            lay = Layer()
            lay.text_mixed((bx, by + 54), cap, size=28, v=255, tracking=0.8)
            cv.paint(lay, "paper_ink", 0.85 * a, mode="mul")

        mono(cv, "模型可以本地，也可以云端 —— 笔记永远在你磁盘上", (MARGIN + 6, 880), size=24,
             color="paper_ink2", opacity=ramp(lt, 2.2, 2.9))

        hud(cv, ctx["t"], ctx["dur"], "代价 · 本地", dark=False)
        return dict(PAPER_POST, zoom=1.0 + 0.02 * pulse(lt, 1.1, 0.3))


# ---------------------------------------------------------------- P7 三步开始


class Steps:
    """三步：双击图标 → 生成演示会话 → 点时间轴。右侧出现一个窗口线稿。"""

    STEPS = [("01", "双击图标", "一键部署，ai小白也能轻松上手"),
             ("02", "生成演示会话", "一节随机变量与分布的课，8 分钟"),
             ("03", "点时间轴上的点", "直接跳回那一秒，看它为什么标这里")]

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        lines, size = fit_lines("三步，先进去看一眼", 96, 900)
        big_block(cv, lines, MARGIN, 180, size, "bone", 1.0, ramp(lt, 0.05, 0.55))

        y0 = 460
        for i, (n, head, sub) in enumerate(self.STEPS):
            t0 = 0.4 + i * 0.7
            a = ramp(lt, t0, t0 + 0.5, out_expo)
            if a <= 0.01:
                continue
            yy = y0 + i * 140
            num = Layer()
            num.text((MARGIN + 4, yy + 44), n, font("din", 70), v=255, tracking=1)
            cv.paint(num, "signal" if i == 0 else "graphite", a, mode="add" if i == 0 else "over")
            lay = Layer()
            lay.text((MARGIN + 120, yy + 42), head, font("cjk_b", 44))
            lay.text_mixed((MARGIN + 120, yy + 92), sub, size=27, tracking=0.8)
            cv.paint(lay, "bone", a, mode="over")

        # 右侧：应用窗口线稿
        wx0, wy0, wx1, wy1 = 1000, 330, W - MARGIN, 830
        wa = ramp(lt, 0.2, 0.9)
        win = Layer()
        win.rect((wx0, wy0, wx1, wy1), v=255, width=2, radius=12)
        win.rect((wx0, wy0, wx1, wy0 + 44), v=255, width=2, radius=12)
        for k in range(3):
            win.circle(wx0 + 26 + k * 22, wy0 + 22, 5, v=110)
        cv.paint(win, "graphite", 0.9 * wa, mode="over")
        # 视频区
        vid = Layer()
        vid.rect((wx0 + 28, wy0 + 74, wx1 - 28, wy0 + 330), v=255, width=2, radius=6)
        hatch(vid, (wx0 + 30, wy0 + 76, wx1 - 30, wy0 + 328), spacing=9, angle=45, v=70)
        cv.paint(vid, "graphite", 0.55 * ramp(lt, 0.6, 1.2), mode="over")
        # 时间轴 + 钉
        tl = Layer()
        tl.line([(wx0 + 40, wy0 + 400), (wx1 - 40, wy0 + 400)], width=2, v=255)
        for k in range(1, 13):
            xx = wx0 + 40 + (wx1 - wx0 - 80) * k / 13
            tl.line([(xx, wy0 + 400), (xx, wy0 + 410)], width=1.2, v=255)
        cv.paint(tl, "graphite", 0.8 * ramp(lt, 0.9, 1.5), mode="over")
        for k, xx in enumerate([0.24, 0.47, 0.72, 0.88]):
            pa = ramp(lt, 1.6 + 0.42 * k, 1.9 + 0.42 * k, out_back)
            if pa <= 0:
                continue
            px = wx0 + 40 + (wx1 - wx0 - 80) * xx
            pin = Layer()
            pin.line([(px, wy0 + 400), (px, wy0 + 400 - 22 * pa)], width=2, v=255)
            pin.circle(px, wy0 + 400 - 22 * pa, 7 * pa, v=255)
            cv.paint(pin, "signal", 0.95 * pa, mode="add")
        # 笔记卡
        for k in range(2):
            ca = ramp(lt, 2.6 + 0.4 * k, 2.95 + 0.4 * k)
            if ca <= 0:
                continue
            cy2 = wy0 + 448 + k * 66
            c = Layer()
            c.rect((wx0 + 40, cy2, wx1 - 40, cy2 + 52), v=255, width=2, radius=5)
            cv.paint(c, "graphite", 0.75 * ca, mode="over")
            t = Layer()
            t.text_mixed((wx0 + 58, cy2 + 34), ["01:58 为什么梯度为零？",
                                                "04:46 边界条件？"][k], size=22, v=255, tracking=0.8)
            cv.paint(t, "ash", 0.9 * ca, mode="over")

        # 鼠标点第一步
        if lt > 1.4:
            a = ramp(lt, 1.4, 1.7)
            mx = MARGIN + 30 - 22
            my = y0 + 40 + 4 * pulse(lt, 1.7, 0.12)
            cur = Layer()
            cur.poly([(mx, my), (mx + 2, my + 34), (mx + 11, my + 26), (mx + 17, my + 40),
                      (mx + 24, my + 36), (mx + 18, my + 23), (mx + 29, my + 21)], v=255)
            cv.paint(cur, "bone", a, mode="over")
            rip = Layer()
            rip.circle(mx + 26, my + 34, 8 + 44 * pulse(lt, 1.7, 0.4), v=255, width=2)
            cv.paint(rip, "signal", 0.65 * pulse(lt, 1.7, 0.4), mode="add")

        mono(cv, "Windows · 不用登录、不需要服务器", (MARGIN + 6, 900), size=24, color="ash",
             opacity=ramp(lt, 2.0, 2.6))

        hud(cv, ctx["t"], ctx["dur"], "上手 · 三步", dark=True)
        return {"bloom": 0.5, "zoom": 1.0 + 0.03 * pulse(lt, 1.7, 0.3)}


# ---------------------------------------------------------------- P8 落版


class Outro:
    """把火花收成一粒句号：谢谢观看。"""

    def render(self, cv: Canvas, lt: float, ctx: dict) -> dict:
        yy = H * 0.30
        line_p = ramp(lt, 0.05, 0.8, in_out_cubic)
        trail = Layer()
        trail.line([(MARGIN, yy), (MARGIN + (W - 2 * MARGIN) * line_p, yy)], width=2, v=255)
        cv.paint(trail, "graphite", 0.5, mode="over")

        if lt > 0.3:
            big_block(cv, ["谢谢观看"], MARGIN, H * 0.55, 168, "bone", 1.0,
                      ramp(lt, 0.3, 1.0, out_quint), tracking=12)
            mono(cv, "小蟹学习助手 · 边听课边把不懂的钉在时间轴上", (MARGIN + 8, H * 0.55 + 92),
                 size=30, color="ash", opacity=ramp(lt, 0.85, 1.45))

        if lt > 2.3:
            mono(cv, "想试：那就来吧", (MARGIN + 8, H * 0.80),
                 size=28, color="signal", opacity=ramp(lt, 2.3, 2.9) * 0.95)

        rule = Layer()
        rw = 520 * ramp(lt, 1.4, 2.2, out_expo)
        rule.line([(MARGIN + 8, H * 0.66), (MARGIN + 8 + rw, H * 0.66)], width=3, v=255)
        cv.paint(rule, "signal", 0.85, mode="add")

        sx = MARGIN + 8 + 18 * (1 - ramp(lt, 0.2, 1.2))
        spark(cv, sx, H * 0.55 - 62, r=15 * (1 - 0.35 * ramp(lt, 3.0, 4.4)),
              inten=0.85 * (1 - 0.45 * ramp(lt, 3.4, 4.6)))

        hud(cv, ctx["t"], ctx["dur"], "谢谢", dark=True, opacity=0.35)
        return {"bloom": 0.55, "fade": ramp(lt, 6.5, 7.7) * 0.9, "zoom": 1.0 + 0.02 * ramp(lt, 0, 4)}
