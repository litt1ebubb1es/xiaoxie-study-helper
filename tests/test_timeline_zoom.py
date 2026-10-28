"""时间轴缩放：3 小时的课要能放大到看清一个回看点。

为什么这件事值得有测试：缩放的坐标换算只要差一点，画出来的回看点就会和刻度对不上，
而这种错"看起来只是稍微偏了"，不会报异常。所以这里断言的是**往返一致**：
时间→像素→时间 必须回到原值，视野边界必须夹在 [0, duration] 内。
"""
from __future__ import annotations

import pytest

from dsh_app.ui.timeline import KnotBand, TimelineWidget


@pytest.fixture()
def tl(qapp):
    w = TimelineWidget()
    w.resize(1000, 400)                  # 给个确定宽度，像素换算才有唯一答案
    bands = [
        KnotBand(id=1, t_start=600.0, t_end=630.0, grade="strong", score=0.8, level="explicit"),
        # 注意让所有回看点都落在时长之内：set_data 会把 duration 撑到最晚的回看点，
        # 这条测试要的是"3 小时的课"，所以回看点不能越界
        KnotBand(id=2, t_start=3000.0, t_end=3020.0, grade="medium", score=0.5, level="implicit"),
    ]
    wins = [{"t_start": float(i * 60), "t_end": float(i * 60 + 60),
             "l_score": 0.5, "b_score": 0.4} for i in range(60)]
    w.set_data(wins, bands, 3600.0)
    return w


def test_default_view_is_the_whole_lesson(tl):
    a, b = tl.view_range()
    assert (a, b) == (0.0, 3600.0)


def test_time_to_pixel_roundtrip(tl):
    for t in (0.0, 12.5, 600.0, 1799.0, 3600.0):
        assert tl._t(tl._x(t)) == pytest.approx(t, abs=0.5)


def test_zoom_keeps_the_anchor_under_the_cursor(tl):
    anchor = 1200.0
    tl.zoom_at(4.0, anchor)
    a, b = tl.view_range()
    assert b - a == pytest.approx(900.0, rel=0.02)
    assert a <= anchor <= b
    # 锚点在视野里的相对位置没变：这才是"缩放不跑偏"
    assert (anchor - a) / (b - a) == pytest.approx(1200.0 / 3600.0, abs=0.02)


def test_zoom_is_clamped_to_the_lesson(tl):
    tl.zoom_at(100.0, 3599.0)             # 一路放大到最右端
    a, b = tl.view_range()
    assert a >= 0.0 and b <= 3600.0 + 1e-6


def test_zoom_out_returns_to_full_view(tl):
    tl.zoom_at(8.0, 1800.0)
    assert tl.view_span > 0
    tl.zoom_at(0.01, 1800.0)
    assert tl.view_span == 0.0            # span>=duration 就归零＝整节可见


def test_zoom_never_goes_below_the_minimum_window(tl):
    for _ in range(20):
        tl.zoom_at(4.0, 1000.0)
    a, b = tl.view_range()
    assert b - a >= 30.0 - 1e-6


def test_reset_view_restores_everything(tl):
    tl.zoom_at(6.0, 600.0)
    tl.reset_view()
    assert tl.view_range() == (0.0, 3600.0)


def test_view_changed_emits_on_zoom_and_reset(qapp, tl):
    seen = []
    tl.viewChanged.connect(lambda a, b: seen.append((a, b)))
    tl.zoom_at(4.0, 600.0)
    tl.reset_view()
    assert len(seen) == 2
    assert seen[-1] == (0.0, 3600.0)


def test_tick_step_follows_the_view_span(tl):
    full = tl._tick_step()
    tl.zoom_at(10.0, 600.0)
    zoomed = tl._tick_step()
    assert zoomed <= full, "放大之后刻度必须更密，否则放大了也读不出时间"


def test_t_is_clamped_inside_the_view(tl):
    tl.zoom_at(4.0, 1800.0)
    a, b = tl.view_range()
    assert tl._t(0) == pytest.approx(a)            # 点在左边留白
    assert tl._t(1e6) == pytest.approx(b)          # 点在右边外面


def test_drag_pans_the_window(tl):
    tl.zoom_at(4.0, 1800.0)
    a0, _b0 = tl.view_range()
    r = tl._plot_rect()
    tl._press_x = r.left() + r.width() / 2
    tl._press_t0 = a0

    class Ev:
        def position(self):
            class P:
                def __init__(self, x):
                    self._x = x

                def x(self):
                    return self._x

            return P(tl._press_x - 100)          # 往左拖 100 像素

    tl.mouseMoveEvent(Ev())
    a1, _b1 = tl.view_range()
    assert a1 > a0, "往左拖=看后面一段"
    assert tl._dragging is True


def test_click_still_selects_a_knot_but_drag_does_not(qapp, tl):
    """拖动是平移，手一松不该顺手把回看点选中——那会让人莫名其妙跳到别的段。"""
    picked = []
    tl.knotSelected.connect(picked.append)
    x = tl._x(615.0)

    class Ev:
        def __init__(self, x):
            self._x = x

        def position(self):
            class P:
                def __init__(self, v):
                    self._v = v

                def x(self):
                    return self._v

            return P(self._x)

    # 纯点击：选中
    tl.mousePressEvent(Ev(x))
    tl.mouseReleaseEvent(Ev(x))
    assert picked == [1]

    # 拖动：不选中
    picked.clear()
    tl.mousePressEvent(Ev(x))
    tl.mouseMoveEvent(Ev(x + 60))
    tl.mouseReleaseEvent(Ev(x + 60))
    assert picked == []


def test_double_click_resets(qapp, tl):
    tl.zoom_at(4.0, 600.0)

    class Ev:
        def accept(self):
            pass

    tl.mouseDoubleClickEvent(Ev())
    assert tl.view_range() == (0.0, 3600.0)


def test_new_data_resets_the_zoom(tl):
    """换一节课就回到全貌：留着上一节的缩放位置会让人找不到自己在哪。"""
    tl.zoom_at(6.0, 600.0)
    tl.set_data([], [], 1800.0)
    assert tl.view_span == 0.0
    assert tl.view_range() == (0.0, 1800.0)