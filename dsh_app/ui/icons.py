"""图标：手写 SVG 路径 → QSvgRenderer 渲染成 QIcon（不下载图标包）。

为什么不下载一套 iconfont/SVG 包：
1. 这个项目要求"下载物不落 C 盘"，而且图标包还带授权问题；
2. 图标颜色必须跟着主题走（侧栏灰、选中橙、主按钮上白），位图要备好几套；
3. 统一 24×24 网格、1.7 线宽、圆头圆角，整套看起来才是一个体系。

第一版用 QPainterPath 逐笔画，结果弧线接不对（多个子路径没 moveTo 就从原点拉线，
渲染出来是一堆"星芒"）。改成手写 SVG 的 M/L/C/A 路径后既准又好读——见
data/probe/icons_sheet.png 的对照图。

用法：
    btn.setIcon(icons.get("play", theme.TEXT_ON_ACCENT))
    nav.addItem(QListWidgetItem(icons.get("timeline"), "时间轴与回看点"))
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from . import theme

# 描边型图标（fill=none, stroke=当前色）
_STROKE: dict[str, str] = {
    # 用书签而不是旗子：旗子无论怎么画，15px 下都容易被读成字母 P；
    # 书签本身就带"在这一刻做个记号"的意思，和「标记回看点」也是同一个隐喻。
    "flag": '<path d="M7 3.6h10v16.8l-5-3.8-5 3.8z"/>',
    "upload": '<path d="M12 16V4m-4.5 4.5L12 4l4.5 4.5"/>'
              '<path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "link": '<path d="M10 13.5l4-4"/><path d="M8.5 15.5a4 4 0 0 1 0-5.6l2.4-2.4"/>'
            '<path d="M15.5 8.5a4 4 0 0 1 0 5.6L13.1 16.5"/>',
    "eye": '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/>'
           '<circle cx="12" cy="12" r="2.7"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20 4v5h-5"/>',
    "demo": '<path d="M5 19v-6M10 19V6M15 19v-8M20 19V3"/>',
    "export": '<path d="M12 3v12m-4.5-4.5L12 15l4.5-4.5"/>'
              '<path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "chat": '<path d="M21 11.6a8.4 8.4 0 0 1-12.3 7.5L4 20.4l1.4-4.5A8.4 8.4 0 1 1 21 11.6z"/>',
    "settings": '<path d="M4 7h5M14 7h6M4 12h8M17 12h3M4 17h3M12 17h8"/>'
                '<circle cx="11.5" cy="7" r="2.2"/><circle cx="14.5" cy="12" r="2.2"/>'
                '<circle cx="9.5" cy="17" r="2.2"/>',
    "timeline": '<path d="M3 12h3.5l2.2-6 3.2 12 2.5-8.2 1.8 2.2H21"/>',
    "reword": '<path d="M17 3l3.5 3.5L17 10"/><path d="M3.5 11V9.4A3.4 3.4 0 0 1 6.9 6h13"/>'
              '<path d="M7 21l-3.5-3.5L7 14"/><path d="M20.5 13v1.6A3.4 3.4 0 0 1 17.1 18h-13"/>',
    "quiz": '<rect x="3.5" y="3.5" width="17" height="17" rx="4"/><path d="M8 12.2l2.8 2.8L16 9.4"/>',
    "map": '<path d="M9 4L3.5 6.4v13.2L9 17.2l6 2.4 5.5-2.4V4.6L15 7 9 4z"/>'
           '<path d="M9 4v13.2M15 7v12.6"/>',
    "attach": '<path d="M20.5 11.2l-8.2 8.2a5 5 0 0 1-7.1-7.1L13.4 4a3.4 3.4 0 0 1 4.8 4.8'
              'l-8.3 8.2a1.9 1.9 0 0 1-2.7-2.7l7.6-7.5"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M16.2 16.2L21 21"/>',
    "chevron": '<path d="M6 9.5l6 6 6-6"/>',
    "book": '<path d="M4 19.2A2.4 2.4 0 0 1 6.4 16.8H20"/>'
            '<path d="M6.4 2.5H20v19.5H6.4A2.4 2.4 0 0 1 4 19.6V4.9A2.4 2.4 0 0 1 6.4 2.5z"/>',
    "spark": '<path d="M12 3l2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2z"/>',
    "monitor": '<rect x="2.8" y="4" width="18.4" height="12.6" rx="2.4"/>'
               '<path d="M9 20.6h6M12 16.6v4"/>',
    "mic": '<rect x="9" y="2.6" width="6" height="11.4" rx="3"/>'
           '<path d="M5.4 11.4a6.6 6.6 0 0 0 13.2 0M12 18v3.2"/>',
    # 吉祥物：小螃蟹。造型只借"大眼 + 宽壳 + 两个张开的夹子"这个意思，
    # 线条全部自己画（参考图是别人画的，不能照抄）。
    # 两种密度：小于 44px 时只留壳和眼，钳子和腿在小尺寸上只会糊成噪点。
    "crab": '<path d="M4.5 14.2a7.5 5.4 0 1 0 15 0a7.5 5.4 0 1 0-15 0"/>'
            '<circle cx="8.9" cy="7.9" r="2.1"/><circle cx="15.1" cy="7.9" r="2.1"/>'
            '<circle cx="8.9" cy="8.3" r=".7" fill="currentColor"/>'
            '<circle cx="15.1" cy="8.3" r=".7" fill="currentColor"/>'
            '<path d="M4.9 12c-1.6-1.5-3.4-1.8-3.9-.7"/>'
            '<path d="M0.9 8.6l1.5 1.2M3.4 8.1l-.2 1.9"/>'
            '<path d="M19.1 12c1.6-1.5 3.4-1.8 3.9-.7"/>'
            '<path d="M23.1 8.6l-1.5 1.2M20.6 8.1l.2 1.9"/>'
            '<path d="M8.6 18.4l-.9 2M15.4 18.4l.9 2"/>',
    "crab-min": '<path d="M4.2 13.8a7.8 5.6 0 1 0 15.6 0a7.8 5.6 0 1 0-15.6 0"/>'
                '<circle cx="9" cy="9.2" r="2.2"/><circle cx="15" cy="9.2" r="2.2"/>'
                '<circle cx="9" cy="9.6" r=".75" fill="currentColor"/>'
                '<circle cx="15" cy="9.6" r=".75" fill="currentColor"/>',
    "check": '<path d="M4.5 12.5l5 5L20 7"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
}

# 填充型图标（fill=当前色, stroke=none）
_FILL: dict[str, str] = {
    "play": '<path d="M8 5.2c0-.8.9-1.3 1.6-.9l9 6.1c.6.4.6 1.3 0 1.7l-9 6.1c-.7.5-1.6 0-1.6-.9V5.2z"/>',
    "send": '<path d="M20.7 3.3a1 1 0 0 0-1.1-.2L3.7 10.2c-.9.3-.9 1.6 0 1.9l6 2.2 2.2 6c.3.9 1.6.9 1.9 0'
            'L20.9 4.4c.2-.4.1-.8-.2-1.1zM11 13.4l7.3-7.3-9.2 5.2 1.9 2.1z"/>',
    "stop": '<rect x="7" y="7" width="10" height="10" rx="2.6"/>',
    "more": '<circle cx="5.5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/>'
            '<circle cx="18.5" cy="12" r="1.7"/>',
}


def _svg(name: str, color: str, width: float) -> bytes:
    if name in _FILL:
        body = _FILL[name]
        attrs = f'fill="{color}" stroke="none"'
    else:
        # currentColor：瞳孔这种"实心小块"要跟描边同色，模板里只写 currentColor，
        # 这里按当次请求的颜色替换（不能改 _STROKE 本体，否则下次换色就串味）
        body = _STROKE[name].replace("currentColor", color)
        attrs = (f'fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linecap="round" stroke-linejoin="round"')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'{attrs}>{body}</svg>').encode("utf-8")


def _render(name: str, color: str, size: int, width: float) -> QPixmap:
    # 没有 QGuiApplication 时构造 QPixmap 会让 Qt 直接 abort（不是抛异常、抓不住），
    # 所以先自己挡一下，报个能看懂的错。
    if QGuiApplication.instance() is None:
        raise RuntimeError("取图标前必须先创建 QApplication")
    renderer = QSvgRenderer(QByteArray(_svg(name, color, width)))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(p)
    p.end()
    return pm


@lru_cache(maxsize=512)
def _cached(name: str, color: str, size: int, width: float) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_render(name, color, size, width))
    ic.addPixmap(_render(name, color, size * 2, width))   # 高分屏不糊
    return ic


def get(name: str, color: str | None = None, size: int = 18,
        width: float = 1.7) -> QIcon:
    """取图标。默认次要文字色；主按钮上请显式传白色。"""
    if name not in _STROKE and name not in _FILL:
        raise KeyError(f"没有这个图标：{name}")
    return _cached(name, color or theme.TEXT_2, int(size), float(width))


def app_icon(size: int = 256) -> QIcon:
    """窗口图标 / 托盘图标：同一只小螃蟹，多备几档尺寸免得起毛边。"""
    ic = QIcon()
    for px in (16, 24, 32, 48, 64, 128, int(size)):
        ic.addPixmap(app_badge(px))
    return ic


def names() -> list[str]:
    return sorted(set(_STROKE) | set(_FILL))


def app_badge(size: int = 30, glyph: str | None = None, width: float = 1.9) -> QPixmap:
    """应用标：橙色圆角方块 + 白色简笔吉祥物。品牌区用它，不占图片资源。"""
    from PySide6.QtGui import QColor, QPainterPath

    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    box = QPainterPath()
    box.addRoundedRect(0, 0, size, size, size * 0.29, size * 0.29)
    p.fillPath(box, QColor(theme.ACCENT))
    # 小尺寸换简化版：任务栏/标题栏只有 16~32px，钳子和腿只会糊成一团
    if glyph is None:
        glyph = "crab" if size >= 44 else "crab-min"
    inner = size * (0.74 if glyph == "crab" else 0.66)
    star = _render(glyph, theme.TEXT_ON_ACCENT, int(round(inner)), width)
    off = (size - inner) / 2.0
    p.drawPixmap(int(round(off)), int(round(off)), star)
    p.end()
    return pm