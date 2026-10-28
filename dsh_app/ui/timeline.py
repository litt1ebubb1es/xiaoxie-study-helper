"""时间轴视图（方案 3.6 / 8.2 阶段一"时间轴可视化"）。

同一张图上叠三样东西：
- 认知负荷曲线 L（内容本身有多难，与用户行为无关）
- 行为信号 B（用户在这段的真实反应）
- 回看点区间（融合判定结果，颜色区分强弱，选中态高亮）

刻意不画"滚动字幕"：听课过程中不打断（ADR-004），这里只做事后回放。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPolygonF)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from . import theme as T

# 颜色全部来自主题模块：图表里最显眼的应该是回看点，不是背景
BG = QColor(T.TL_BG)
GRID = QColor(T.TL_GRID)
TEXT = QColor(T.TL_TEXT)
TEXT_HI = QColor(T.TL_TEXT_HI)
LOAD = QColor(T.TL_LOAD)
BEHAV = QColor(T.TL_BEHAV)
KNOT_STRONG = QColor(T.TL_KNOT_STRONG)
KNOT_MED = QColor(T.TL_KNOT_MED)
PLAYHEAD = QColor(T.TL_PLAYHEAD)
PLOT_SURFACE = T.TL_PLOT
BORDER_LINE = T.BORDER
GRID_H = T.TL_GRID_H
T_MONO = T.FONT_MONO
T_FONT = T.FONT_STACK[0]


@dataclass
class KnotBand:
    id: int
    t_start: float
    t_end: float
    score: float
    grade: str
    level: str
    evidence: str = ""


class TimelineWidget(QWidget):
    knotSelected = Signal(int)
    viewChanged = Signal(float, float)   # 视野起止（秒），给外面的"当前看哪一段"用

    def __init__(self, parent=None):
        super().__init__(parent)
        self.windows: list[dict] = []
        self.knots: list[KnotBand] = []
        self.duration: float = 0.0
        self.playhead: float = 0.0
        self.selected_id: int = -1
        self.hover_t: float = -1.0
        # 视野：view_span=0 表示"整节都看得见"。3 小时的课全挤一屏时，
        # 一个回看点只有几个像素宽，缎带叠在一起——所以必须能放大看局部。
        self.view_t0: float = 0.0
        self.view_span: float = 0.0
        self._press_x: float | None = None
        self._press_t0: float = 0.0
        self._dragging: bool = False
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # -- 数据 -----------------------------------------------------------
    def set_data(self, windows: Sequence[dict], knots: Sequence[KnotBand], duration: float) -> None:
        self.windows = list(windows)
        self.knots = list(knots)
        self.duration = max(float(duration or 0.0),
                            max((k.t_end for k in self.knots), default=0.0), 1.0)
        # 换了一节课就回到"看全貌"：留着上一节的缩放位置会让人找不到自己在哪
        self.view_t0, self.view_span = 0.0, 0.0
        self._emit_view()
        self.update()

    def set_playhead(self, t: float) -> None:
        self.playhead = max(0.0, float(t))
        self.update()

    def set_selected(self, knot_id: int) -> None:
        self.selected_id = int(knot_id)
        self.update()

    # -- 几何 -----------------------------------------------------------
    def _plot_rect(self) -> QRectF:
        return QRectF(46, 30, max(10.0, self.width() - 64), max(10.0, self.height() - 64))

    def _span(self) -> float:
        """当前视野宽度（秒）。view_span=0 表示没放大，等于整节时长。"""
        return self.view_span if self.view_span > 0 else max(self.duration, 1e-6)

    def view_range(self) -> tuple[float, float]:
        """视野的 [起, 止]，并把起点夹在合法范围内（别让用户拖到时间轴外面）。"""
        span = self._span()
        a = min(max(0.0, self.view_t0), max(0.0, self.duration - span))
        return a, a + span

    def _x(self, t: float) -> float:
        r = self._plot_rect()
        a, b = self.view_range()
        return r.left() + ((float(t) - a) / max(1e-6, b - a)) * r.width()

    def _t(self, x: float) -> float:
        r = self._plot_rect()
        a, b = self.view_range()
        if r.width() <= 0:
            return a
        t = a + (x - r.left()) / r.width() * (b - a)
        return max(a, min(b, t))

    # -- 视野控制 -------------------------------------------------------
    def reset_view(self) -> None:
        self.view_t0, self.view_span = 0.0, 0.0
        self._emit_view()
        self.update()

    def _emit_view(self) -> None:
        a, b = self.view_range()
        self.viewChanged.emit(a, b)

    def zoom_at(self, factor: float, anchor_t: float | None = None) -> None:
        """以某个时间点为锚缩放。锚点不动，视野绕它收/放。"""
        if self.duration <= 0:
            return
        a, b = self.view_range()
        anchor = (a + b) / 2.0 if anchor_t is None else max(a, min(b, float(anchor_t)))
        span = max(30.0, min(self.duration, (b - a) / max(0.05, float(factor))))
        ratio = (anchor - a) / max(1e-6, b - a)
        self.view_span = span
        self.view_t0 = anchor - ratio * span
        lo = min(max(0.0, self.view_t0), max(0.0, self.duration - span))
        self.view_t0 = lo
        if span >= self.duration - 1e-6:
            self.view_t0, self.view_span = 0.0, 0.0
        self._emit_view()
        self.update()

    # -- 绘制 -----------------------------------------------------------
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()

        # 绘图底：白底上再铺一层更浅的面，曲线才有"落在纸上"的感觉
        p.fillRect(r, QBrush(QColor(PLOT_SURFACE)))
        p.setPen(QPen(QColor(BORDER_LINE), 1))
        p.drawRect(r)

        # 水平参考线（25/50/75%），比竖线更淡：它只用来读高度，不该抢视线
        p.setPen(QPen(QColor(GRID_H), 1, Qt.SolidLine))
        p.setFont(QFont(T_MONO, 7))
        for frac in (0.25, 0.5, 0.75):
            y = r.bottom() - frac * r.height() * 0.92
            p.drawLine(QPointF(r.left() + 1, y), QPointF(r.right() - 1, y))
            # 左侧百分比：图很高时不标刻度，曲线就只能看出"起伏"、读不出"多难"
            p.setPen(QPen(QColor(TEXT), 1))
            p.drawText(QRectF(4, y - 7, r.left() - 9, 14),
                       Qt.AlignRight | Qt.AlignVCenter, f"{int(frac * 100)}")
            p.setPen(QPen(QColor(GRID_H), 1, Qt.SolidLine))
        p.setPen(QPen(QColor(GRID_H), 1, Qt.SolidLine))

        step = self._tick_step()
        a, b = self.view_range()
        t = math.floor(a / step) * step
        p.setFont(QFont(T_MONO, 8))
        while t <= b + 1e-6:
            x = self._x(t)
            if x >= r.left() - 1:
                p.setPen(QPen(QColor(GRID), 1, Qt.DotLine))
                p.drawLine(QPointF(x, r.top() + 1), QPointF(x, r.bottom() - 1))
                p.setPen(QPen(TEXT))
                p.drawText(QRectF(x - 30, r.bottom() + 6, 60, 14),
                           Qt.AlignHCenter | Qt.AlignTop, _mmss(t))
            t += step

        if not self.windows:
            p.setPen(QPen(TEXT))
            p.setFont(QFont(T_FONT, 10))
            p.drawText(r, Qt.AlignCenter, "还没有分析结果。录制一节课，或导入一段本地视频。")
            return

        # 缩小/放大之后，超出视野的回看点带、曲线和播放头都得裁掉：
        # 否则它们会画到左边的刻度栏和上下的留白里。
        p.save()
        p.setClipRect(r.adjusted(-2, -10, 2, 2))

        # 回看点带先画：作为背景层，不能盖住曲线（否则带内曲线读不出来）
        for k in self.knots:
            x0, x1 = self._x(k.t_start), self._x(k.t_end)
            col = QColor(KNOT_STRONG if k.grade == "strong" else KNOT_MED)
            sel = k.id == self.selected_id
            w_band = max(3.0, x1 - x0)
            # 顶部缎带：一眼看到"哪几段卡住了"，又不糊住曲线
            rib = QRectF(x0, r.top() + 1, w_band, 5)
            p.fillRect(rib, QBrush(col))
            # 底部刻度上的短横标记：不占高度，也能对齐时间
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(x0, r.bottom() - 5), QPointF(x0 + w_band, r.bottom() - 5))
            if sel:
                # 选中：缎带加高 + 两侧实线。不做通栏染色——白底上一大片淡橙会显脏
                p.fillRect(QRectF(x0, r.top() + 1, w_band, 9), QBrush(col))
                edge = QColor(col); edge.setAlpha(140)
                p.setPen(QPen(edge, 1.1))
                # 只在边缘真的落在视野里时才画：放大之后一个远处的回看点会被裁到
                # 视野边线上，那里画一条通栏竖线会被误读成"这个时刻有回看点"。
                if r.left() + 1 <= x0 <= r.right() - 1:
                    p.drawLine(QPointF(x0, r.top() + 10), QPointF(x0, r.bottom() - 6))
                if r.left() + 1 <= x1 <= r.right() - 1:
                    p.drawLine(QPointF(x1, r.top() + 10), QPointF(x1, r.bottom() - 6))

        # 负荷曲线（填充）
        path = QPainterPath()
        first = True
        for w in self.windows:
            x = self._x(float(w["t_start"]))
            y = r.bottom() - float(w["l_score"]) * r.height() * 0.92
            if first:
                path.moveTo(x, r.bottom())
                path.lineTo(x, y)
                first = False
            else:
                path.lineTo(x, y)
        if not first:
            path.lineTo(self._x(float(self.windows[-1]["t_end"])), r.bottom())
            path.closeSubpath()
            grad = QLinearGradient(0, r.top(), 0, r.bottom())
            c1 = QColor(LOAD); c1.setAlpha(26)
            c2 = QColor(LOAD); c2.setAlpha(3)
            grad.setColorAt(0.0, c1)
            grad.setColorAt(1.0, c2)
            p.fillPath(path, QBrush(grad))

        # L 折线
        p.setPen(QPen(LOAD, 1.7))
        poly = QPolygonF()
        for w in self.windows:
            poly.append(QPointF(self._x((float(w["t_start"]) + float(w["t_end"])) / 2),
                                r.bottom() - float(w["l_score"]) * r.height() * 0.92))
        p.drawPolyline(poly)

        # B 折线
        p.setPen(QPen(BEHAV, 1.8, Qt.DashLine))
        polyb = QPolygonF()
        for w in self.windows:
            polyb.append(QPointF(self._x((float(w["t_start"]) + float(w["t_end"])) / 2),
                                 r.bottom() - float(w["b_score"]) * r.height() * 0.92))
        p.drawPolyline(polyb)

        # 播放头
        if self.playhead > 0:
            x = self._x(self.playhead)
            p.setPen(QPen(PLAYHEAD, 1.2))
            p.drawLine(QPointF(x, r.top() - 6), QPointF(x, r.bottom()))
            p.setBrush(QBrush(PLAYHEAD))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(x - 4, r.top() - 6), QPointF(x + 4, r.top() - 6),
                                     QPointF(x, r.top() + 1)]))
        p.restore()

        # 图例
        p.setFont(QFont(T_FONT, 8))
        lx = r.left()
        for label, col, dashed in (("认知负荷 L", LOAD, False), ("行为信号 B", BEHAV, True),
                                   ("重点回看", KNOT_STRONG, False), ("一般回看", KNOT_MED, False)):
            p.setPen(QPen(col, 2, Qt.DashLine if dashed else Qt.SolidLine))
            p.drawLine(QPointF(lx, 14), QPointF(lx + 18, 14))
            p.setPen(QPen(TEXT))
            p.drawText(QRectF(lx + 24, 6, 88, 16), Qt.AlignLeft | Qt.AlignVCenter, label)
            lx += 108

    def _tick_step(self) -> float:
        target = 8
        raw = max(1.0, self._span() / max(1, target))
        for cand in (10, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if raw <= cand:
                return cand
        return 3600

    # -- 交互 -----------------------------------------------------------
    def wheelEvent(self, ev) -> None:
        """滚轮缩放：以光标所在时间为锚点，最远放到 30 秒视野。"""
        if self.duration <= 0:
            return
        delta = ev.angleDelta().y()
        if not delta:
            return
        self.zoom_at(1.25 if delta > 0 else 0.8, self._t(ev.position().x()))
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        """双击回到"整节都看得见"。"""
        self.reset_view()
        ev.accept()

    def mousePressEvent(self, ev) -> None:
        self._press_x = ev.position().x()
        self._press_t0 = self.view_range()[0]
        self._dragging = False

    def mouseReleaseEvent(self, ev) -> None:
        moved = (self._press_x is not None
                 and abs(ev.position().x() - self._press_x) > 4)
        self._press_x = None
        if moved:
            # 拖动=平移，不该顺手把回看点选中（手一松就跳走很难受）
            self._dragging = False
            return
        self._drag_select(ev.position().x())

    def _drag_select(self, x: float) -> None:
        t = self._t(x)
        best = None
        for k in self.knots:
            # 屏幕距离比时间距离准：放大后 5 秒可能是半屏，缩小后不到 1 像素
            pad = max(4.0, (self.view_range()[1] - self.view_range()[0]) * 0.01)
            if k.t_start - pad <= t <= k.t_end + pad:
                if best is None or abs((k.t_start + k.t_end) / 2 - t) < abs((best.t_start + best.t_end) / 2 - t):
                    best = k
        if best is not None:
            self.selected_id = best.id
            self.update()
            self.knotSelected.emit(best.id)

    def mouseMoveEvent(self, ev) -> None:
        x = ev.position().x()
        if self._press_x is not None:
            dx = x - self._press_x
            if abs(dx) > 4:
                self._dragging = True
                r = self._plot_rect()
                span = self._span()
                # 位移换算成时间：拖多少像素就挪多少秒，方向是"抓着纸往左拉"
                shift = -(dx / max(1.0, r.width())) * span
                self.view_span = span
                self.view_t0 = max(0.0, min(self.duration - span,
                                            self._press_t0 + shift)) if self.duration > span else 0.0
                self._emit_view()
                self.update()
                return
        self.hover_t = self._t(x)
        for k in self.knots:
            if k.t_start <= self.hover_t <= k.t_end:
                QToolTip.showText(
                    ev.globalPosition().toPoint(),
                    f"[{_mmss(k.t_start)}] {k.evidence or k.level}｜score {k.score:.2f}",
                    self)
                return
        QToolTip.hideText()


def _mmss(sec: float) -> str:
    sec = max(0, int(float(sec or 0)))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"