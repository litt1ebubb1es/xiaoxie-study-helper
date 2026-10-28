"""对话气泡的 HTML（从 main_window 抽出来单独放，方便调版式和截图对照）。

两条硬约束决定了这里的写法：

1. **Qt 的富文本不支持 `background` / `border-left` / `border-radius`**。
   实测（data/probe/bubble_test.png）：div 上写 background-color 完全不生效，
   整块仍是白的。只有 `<table bgcolor=...>` 真的会涂色。
   所以气泡一律用表格拼：左边一列 4px 当色条，右边一列上底色。

2. **不做左右对齐的气泡**。桌面窗口宽度变化很大，右侧对齐在窄窗口里会把正文
   挤成一列字；统一左对齐 + 底色区分说话人，正文永远吃满宽度。

说话人靠"色条 + 小标签"区分：你 = 橙，助教 = 暖灰。
"""
from __future__ import annotations

from . import theme


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _gap(px: int = 10) -> str:
    """气泡之间的空隙。Qt 里 div 的 margin/height 都不可靠（会被折叠成 0），
    用一张只有一行、字号极小的表来撑开最稳。"""
    return (f"<table width='100%' cellspacing='0' cellpadding='0'>"
            f"<tr><td style='font-size:{max(1, px // 2)}px'>&nbsp;</td></tr></table>")


def chat_html(msgs: list[dict], model: str | None) -> str:
    if not msgs:
        hint = ("还没有对话。可以问：这段没懂、这个定理怎么用、帮我对比 A 和 B。<br>"
                "左侧导入你自己的讲义/论文后，回答会优先依据这些资料，并标明出处。")
        return (f"<div style='color:{theme.TEXT_3};padding:4px 6px;line-height:1.8'>"
                f"{hint}</div>")

    out: list[str] = []
    for i, m in enumerate(msgs):
        body = _esc(m["content"]).replace("\n", "<br>")
        if m["role"] == "user":
            out.append(_bubble("你", body, bar=theme.ACCENT, fill=theme.ACCENT_SOFT,
                               label_color=theme.ACCENT_PRESS))
        else:
            who = f"助教 · {model}" if model else "助教"
            extra = _sources(m.get("sources") or [])
            out.append(_bubble(who, body + extra, bar=theme.BORDER_STRONG,
                               fill=theme.SURFACE, label_color=theme.TEXT_3))
        if i < len(msgs) - 1:
            out.append(_gap(12))
    return "".join(out)


def _bubble(who: str, body_html: str, *, bar: str, fill: str, label_color: str) -> str:
    return (
        f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        f"<td width='4' bgcolor='{bar}'>&nbsp;</td>"
        f"<td bgcolor='{fill}' style='padding-left:12px;padding-right:12px;"
        f"padding-top:8px;padding-bottom:9px'>"
        f"<span style='color:{label_color};font-size:11px'>{who}</span><br>"
        f"<span style='color:{theme.TEXT}'>{body_html}</span>"
        f"</td></tr></table>")


def _sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    rows = "".join(
        f"<tr><td style='padding-top:3px;color:{theme.TEXT_2};font-size:12px'>"
        f"《{_esc(x.get('doc', ''))}》{_where(x)}"
        f"<span style='color:{theme.TEXT_3}'>（相关度 {x.get('score', '')}）</span>"
        f"　{_esc(x.get('snippet', ''))}</td></tr>" for x in sources)
    return (
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-top:9px'>"
        f"<tr><td bgcolor='{theme.SURFACE_ALT}' style='padding:8px 10px'>"
        f"<span style='color:{theme.TEXT_3};font-size:11px'>依据</span>"
        f"<table width='100%' cellspacing='0' cellpadding='0'>{rows}</table>"
        f"</td></tr></table>")


def _where(x: dict) -> str:
    """出处：有页码/时间码就显示它，让人能回原文；否则退回「第 N 段」。"""
    ref = (x.get("source_ref") or "").strip()
    if ref:
        return f"<span style='color:{theme.ACCENT_PRESS}'>{_esc(ref)}</span>"
    sec = (x.get("section") or "").strip()
    if sec:
        return f"<span style='color:{theme.ACCENT_PRESS}'>·{_esc(sec)}</span>"
    return f"<span style='color:{theme.ACCENT_PRESS}'>第 {x.get('ord', 0)} 段</span>"