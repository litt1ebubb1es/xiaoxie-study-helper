"""主题与对话气泡的回归测试。

为什么值得测：换肤是"全局改一处"的操作，最容易出的问题不是"难看"而是
① 某个控件还残留旧色值（在浅色底上直接看不见）；② 对话里的用户输入
被当 HTML 执行；③ 颜色 token 拼错导致整块变透明。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from dsh_app.ui import chat_html as CH
from dsh_app.ui import theme as T

UI_DIR = Path(__file__).resolve().parent.parent / "dsh_app" / "ui"


# ------------------------------------------------------------------ 主题
def test_theme_is_light():
    """橙白主题的前提：画布是浅色，强调色是橙。"""
    def rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    bg = rgb(T.BG)
    assert min(bg) > 240, "画布必须是近白（浅色主题）"
    r, g, b = rgb(T.ACCENT)
    assert r > 200 and g < 140 and b < 90, "强调色必须是橙，不能又变回蓝"


@pytest.mark.parametrize("token", ["ACCENT", "TEXT", "SURFACE", "BORDER",
                                   "TL_LOAD", "TL_BEHAV", "TL_KNOT_STRONG"])
def test_tokens_are_valid_hex(token):
    v = getattr(T, token)
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", v), f"{token}={v} 不是合法色值"


def test_qss_uses_tokens_and_has_no_old_dark_colors():
    qss = T.qss()
    assert T.ACCENT in qss and T.BG in qss
    for old in ("#0f1319", "#12151c", "#1d2534", "#4cc2ff", "#e6edf7"):
        assert old not in qss.lower(), f"样式表里还有旧深色 {old}"


def test_radius_scale_is_ordered():
    """圆角要有节奏：卡片比控件更圆，胶囊是极限值。混着 4/6/10 会显得东一块西一块。"""
    assert T.R_CTRL < T.R_CARD < T.R_PILL
    qss = T.qss()
    assert f"border-radius:{T.R_CARD}px" in qss
    assert f"border-radius:{T.R_CTRL}px" in qss


def test_spacing_scale_is_ordered():
    steps = [T.GAP_XS, T.GAP_S, T.GAP_M, T.GAP_L, T.GAP_XL]
    assert steps == sorted(steps) and len(set(steps)) == 5


def test_no_hardcoded_hex_left_in_ui_code():
    """UI 代码里不该再出现裸色值：颜色只允许来自 theme 模块。"""
    bad = {}
    for f in UI_DIR.glob("*.py"):
        if f.name in ("theme.py",):
            continue
        hits = re.findall(r"#[0-9A-Fa-f]{6}\b", f.read_text(encoding="utf-8"))
        # chat_html 里为了 Qt 表格 bgcolor 需要写 #RRGGBB 吗？不 —— 都是从 theme 插进去的
        if hits:
            bad[f.name] = sorted(set(hits))
    assert not bad, f"发现裸色值：{bad}"


# ------------------------------------------------------------------ 气泡
def test_user_input_is_escaped():
    """用户会问 `<script>` 这类东西：必须转义，不能当 HTML 执行。"""
    html = CH.chat_html([{"role": "user", "content": "<script>alert(1)</script> & x"}], None)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp;" in html


def test_bubbles_use_tables_because_qt_ignores_div_background():
    """Qt 富文本对 div 的 background/border-left 无效（实测见 data/probe/bubble_test.png），
    所以气泡必须靠 table bgcolor 上色——这条约束写死在测试里，防止以后被"优化"回去。"""
    html = CH.chat_html([{"role": "user", "content": "问题"},
                         {"role": "assistant", "content": "回答"}], "m1")
    assert html.count("<table") >= 2
    assert f"bgcolor='{T.ACCENT}'" in html      # 你：橙色条
    assert f"bgcolor='{T.SURFACE}'" in html     # 助教：白底


def test_sources_show_where_the_answer_came_from():
    html = CH.chat_html([{
        "role": "assistant", "content": "因为右连续",
        "sources": [{"doc": "讲义.pdf", "source_ref": "第2章 · 14页", "score": "0.82",
                     "snippet": "分布函数右连续", "ord": 1}]}], None)
    assert "依据" in html and "讲义.pdf" in html and "第2章 · 14页" in html
    assert "0.82" in html


def test_empty_chat_says_what_to_ask():
    html = CH.chat_html([], None)
    assert "还没有对话" in html