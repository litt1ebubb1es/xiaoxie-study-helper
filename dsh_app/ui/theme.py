"""主题：橙 + 白，圆润极简（白纸 + 一点橙）。

为什么单独一个模块而不是继续把颜色写在控件里：
上一版是深色，颜色散在 100 多处内联样式里，换肤等于全文件重写。
现在颜色、圆角、间距都只有这一份，控件里只引用 `theme.xxx`。

四条设计原则（照 draw-ui：层级靠留白和字重，不靠描边堆叠）：
1. 只有一种强调色（橙），只用在**主操作**和**当前状态**上，别到处点。
2. 背景三级：画布 #FBFAF8 → 卡片 #FFFFFF → 淡橙 #FFF3EA。
3. 层次靠"卡片 + 很浅的投影"，不靠加深描边：描边一多，界面就显脏。
4. 圆角有节奏：卡片 14、控件 10、小标签 999（胶囊）。圆角大小一致才"圆滑"，
   混着 4/6/10 会显得东一块西一块。
"""
from __future__ import annotations

from pathlib import Path

# ---- 背景与描边 -------------------------------------------------------
BG = "#FBFAF8"            # 画布（暖白）
SURFACE = "#FFFFFF"       # 卡片/输入框
SURFACE_ALT = "#F7F4F0"   # 侧栏、表头、次级面板
SURFACE_TINT = "#FFF3EA"  # 橙色淡底（选中、hover）
BORDER = "#EFEAE3"        # 常规描边（很浅：层次主要靠投影）
BORDER_STRONG = "#E2D9CE"

# ---- 文字 -------------------------------------------------------------
TEXT = "#191C1F"
TEXT_2 = "#5B646E"
TEXT_3 = "#8C949C"
TEXT_ON_ACCENT = "#FFFFFF"
ON_ACCENT = TEXT_ON_ACCENT          # 图标用的别名：白图标压在橙底上

# ---- 强调色（唯一的橙） ----------------------------------------------
ACCENT = "#F2621B"
ACCENT_HOVER = "#DE5512"
ACCENT_PRESS = "#C2470B"
ACCENT_SOFT = "#FFF1E7"
ACCENT_LINE = "#F9CDB0"

# ---- 语义色（克制使用，只在状态标签上） ------------------------------
OK = "#0E7A4E"
OK_SOFT = "#E8F6EF"
WARN = "#A05C09"
WARN_SOFT = "#FDF4E6"
BAD = "#C13A2C"
BAD_SOFT = "#FBEDEB"

# ---- 圆角与间距（统一节奏） ------------------------------------------
R_CARD = 14
R_CTRL = 10
R_PILL = 999
GAP_XS = 4
GAP_S = 8
GAP_M = 12
GAP_L = 16
GAP_XL = 22

# ---- 图表（时间轴） ---------------------------------------------------
TL_BG = "#FFFFFF"
TL_PLOT = "#FCFBF9"
TL_GRID = "#EFEAE3"
TL_GRID_H = "#F4EFE9"
TL_TEXT = "#6B7280"
TL_TEXT_HI = "#191C1F"
TL_LOAD = "#64748B"        # 认知负荷：冷灰蓝，退到后面去
TL_BEHAV = "#F59E0B"       # 行为信号：琥珀
TL_KNOT_STRONG = "#EA580C"  # 显性回看点：砖橙
TL_KNOT_MED = "#FBBF24"     # 中性回看点：浅琥珀
TL_PLAYHEAD = "#1F2937"     # 播放头：近黑（不要用橙色抢戏）
TL_SELECT = "#0F172A"

# 字体：QSS 里的 font-family 只认第一个能用的，所以只用单个确定存在的字族；
# 真正的回退交给 apply_font()（QFont.setFamilies 才支持字体族回退列表）。
FONT_UI = "Microsoft YaHei UI"
FONT_STACK = ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
              "Noto Sans CJK SC", "Segoe UI"]
FONT_MONO = "Cascadia Mono"
FONT_MONO_STACK = ["Cascadia Mono", "Consolas", "JetBrains Mono", "Courier New"]
FONT_SIZE = 13


def apply_font(app) -> None:
    """给整个应用设字体：带字族回退列表，且明确指定中文覆盖字体。

    之前 QSS 里写 `font-family:"a","b",sans-serif` 在 offscreen 下会挑到没有中文字形的
    字体，整屏变成「豆腐块」。所以字体在这里设一次，QSS 只管颜色和尺寸。
    """
    from PySide6.QtGui import QFont

    f = QFont()
    f.setFamilies(FONT_STACK)
    f.setPixelSize(FONT_SIZE)
    f.setHintingPreference(QFont.PreferFullHinting)
    f.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(f)


def soft_shadow(widget, radius: int = 26, dy: int = 3, alpha: int = 16):
    """给卡片挂一层很浅的投影。

    QSS 没有 box-shadow，只能走 QGraphicsDropShadowEffect。投影要**浅到几乎看不见**：
    它的作用是让白卡片从暖白画布上"浮"起来 1mm，不是做立体感。
    """
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QGraphicsDropShadowEffect

    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setOffset(0, dy)
    eff.setColor(QColor(25, 22, 18, alpha))
    widget.setGraphicsEffect(eff)
    return eff


def _check_icon() -> str:
    """勾选标记的 PNG 路径：QSS 能 url() 引用图片，但不能画图形。"""
    return (Path(__file__).with_name("assets") / "check.png").as_posix()


def qss() -> str:
    """全局样式表。控件的 objectName 约定见 README「界面约定」。"""
    _CHECK_ICON = _check_icon()
    return f"""
QWidget {{
    background:{BG}; color:{TEXT};
    font-size:{FONT_SIZE}px;
}}
QMainWindow, QDialog {{ background:{BG}; }}
QToolTip {{
    background:{SURFACE}; color:{TEXT}; border:1px solid {BORDER_STRONG};
    border-radius:{R_CTRL}px; padding:7px 10px;
}}

/* ---- 页签：下划线式（像目录，不像一排按钮） ---- */
QTabWidget::pane {{ border:0; background:{BG}; }}
QTabBar {{ background:transparent; }}
QTabBar::tab {{
    background:transparent; color:{TEXT_3};
    padding:10px 16px; margin-right:4px;
    border:0; border-bottom:2px solid transparent;
}}
QTabBar::tab:hover {{ color:{TEXT}; background:{SURFACE_TINT};
                     border-top-left-radius:{R_CTRL}px; border-top-right-radius:{R_CTRL}px; }}
QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; font-weight:600; }}

/* ---- 按钮：胶囊感，主操作实心橙 ---- */
QPushButton {{
    background:{SURFACE}; color:{TEXT};
    border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL}px;
    padding:7px 14px; min-height:20px;
}}
QPushButton:hover {{ background:{SURFACE_TINT}; border-color:{ACCENT_LINE}; }}
QPushButton:pressed {{ background:{ACCENT_SOFT}; }}
QPushButton:checked {{ background:{ACCENT_SOFT}; border-color:{ACCENT_LINE}; color:{ACCENT_PRESS}; }}
QPushButton:disabled {{ color:{TEXT_3}; border-color:{BORDER}; background:{SURFACE_ALT}; }}
QPushButton#primary {{
    background:{ACCENT}; color:{TEXT_ON_ACCENT};
    border:1px solid {ACCENT}; font-weight:600;
}}
QPushButton#primary:hover {{ background:{ACCENT_HOVER}; border-color:{ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background:{ACCENT_PRESS}; border-color:{ACCENT_PRESS}; }}
QPushButton#primary:disabled {{
    background:{ACCENT_SOFT}; color:{ACCENT_LINE}; border-color:{ACCENT_LINE};
}}
/* 危险操作不用实心红：白底红字更克制，也避免主界面出现两个色块 */
QPushButton#danger {{
    background:{SURFACE}; color:{BAD}; border:1px solid #EDC3BC;
}}
QPushButton#danger:hover {{ background:{BAD_SOFT}; border-color:{BAD}; }}
QPushButton#danger:disabled {{ color:{TEXT_3}; border-color:{BORDER}; background:{SURFACE_ALT}; }}
QPushButton#ghost {{
    background:transparent; border:1px solid transparent; color:{TEXT_2}; padding:5px 9px;
}}
QPushButton#ghost:hover {{ background:{SURFACE_TINT}; color:{TEXT}; }}

/* ---- 输入 ---- */
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox {{
    background:{SURFACE}; border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL}px;
    selection-background-color:{ACCENT_SOFT}; selection-color:{TEXT};
}}
QLineEdit {{ padding:7px 11px; }}
QPlainTextEdit, QTextEdit, QTextBrowser {{ padding:10px; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus {{ border:1px solid {ACCENT}; }}
QComboBox {{
    background:{SURFACE}; border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL}px;
    padding:6px 11px; min-height:20px;
}}
QComboBox:hover {{ border-color:{ACCENT_LINE}; }}
QComboBox::drop-down {{ border:0; width:22px; }}
QComboBox::down-arrow {{
    image:none; border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid {TEXT_3}; width:0; height:0; margin-right:8px;
}}
QComboBox QAbstractItemView {{
    background:{SURFACE}; border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL}px;
    selection-background-color:{ACCENT_SOFT}; selection-color:{TEXT}; padding:6px;
}}

/* ---- 复选框/单选框 ---- */
QCheckBox, QRadioButton {{ spacing:8px; background:transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; }}
QCheckBox::indicator {{ border:1px solid {BORDER_STRONG}; border-radius:5px;
                        background:{SURFACE}; }}
QCheckBox::indicator:hover {{ border-color:{ACCENT}; }}
QCheckBox::indicator:checked {{
    background:{ACCENT}; border-color:{ACCENT};
    image:url("{_CHECK_ICON}");
}}
QRadioButton::indicator {{ border:1px solid {BORDER_STRONG}; border-radius:9px;
                           background:{SURFACE}; }}
QRadioButton::indicator:checked {{ border:5px solid {ACCENT}; background:{SURFACE}; }}

/* ---- 表格：表格本身无框，靠外层卡片包出圆角 ---- */
QTableWidget, QTreeWidget {{ background:transparent; border:0; gridline-color:{BORDER}; }}
QListWidget {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:{R_CARD}px; }}
QTableWidget::item {{ padding:7px 8px; border-bottom:1px solid {BORDER}; }}
QTableWidget::item:hover {{ background:{SURFACE_TINT}; }}
QTableWidget::item:selected, QListWidget::item:selected {{
    background:{ACCENT_SOFT}; color:{TEXT};
}}
QListWidget#plain {{ background:transparent; border:0; border-radius:0; }}
QListWidget::item {{ padding:8px 10px; border-radius:{R_CTRL}px; margin:2px 4px; }}
QListWidget::item:hover {{ background:{SURFACE_TINT}; }}
QHeaderView {{ background:transparent; }}
QHeaderView::section {{
    background:{SURFACE_ALT}; color:{TEXT_2};
    border:0; border-bottom:1px solid {BORDER};
    padding:9px 8px; font-weight:600;
}}
QHeaderView::section:first {{ border-top-left-radius:{R_CARD - 3}px; }}
QHeaderView::section:last {{ border-top-right-radius:{R_CARD - 3}px; }}
QTableCornerButton::section {{ background:{SURFACE_ALT}; border:0; }}

/* ---- 分组框：卡片式 ---- */
QGroupBox {{
    border:1px solid {BORDER}; border-radius:{R_CARD}px; margin-top:14px;
    padding:14px 12px 12px 12px; background:{SURFACE};
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:14px; padding:0 6px; color:{TEXT_2}; font-weight:600;
}}

/* ---- 进度条 ---- */
QProgressBar {{
    background:{SURFACE_ALT}; border:0; border-radius:5px; text-align:center;
    color:{TEXT_2}; min-height:8px;
}}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:5px; }}

/* ---- 滚动条：细、留白、不抢视线 ---- */
QScrollBar:vertical {{ background:transparent; width:11px; margin:3px 2px; }}
QScrollBar::handle:vertical {{
    background:{BORDER_STRONG}; border-radius:6px; min-height:30px;
}}
QScrollBar::handle:vertical:hover {{ background:{TEXT_3}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
QScrollBar:horizontal {{ background:transparent; height:11px; margin:2px 3px; }}
QScrollBar::handle:horizontal {{
    background:{BORDER_STRONG}; border-radius:6px; min-width:30px;
}}

/* ---- 状态栏 ---- */
QStatusBar {{ background:{SURFACE_ALT}; color:{TEXT_2}; border-top:1px solid {BORDER}; }}
QStatusBar::item {{ border:0; }}

/* ---- 分割器 ---- */
QSplitter::handle {{ background:transparent; width:10px; }}
QSplitter::handle:hover {{ background:{ACCENT_LINE}; border-radius:5px; }}

/* ---- 卡片与文字角色（objectName） ---- */
QFrame#card {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:{R_CARD}px;
}}
QFrame#quizCard {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:{R_CARD}px;
}}
QFrame#quizCard:hover {{ border-color:{ACCENT_LINE}; }}
QFrame#panel {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:{R_CARD}px;
}}
QFrame#toolbar {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:{R_CARD}px;
}}
QFrame#divider {{ background:{BORDER}; border:0; }}
/* 左栏：一整条，不是浮起来的卡片——浮卡片会让右栏显得"贴在另一层"上 */
QFrame#sidebar {{
    background:{SURFACE}; border:0; border-right:1px solid {BORDER};
    border-radius:0;
}}
QFrame#recordCard {{
    background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:{R_CARD}px;
}}
/* 导航：竖排图标 + 文字，选中只染淡橙，不用整块实心 */
QListWidget#nav {{ background:transparent; border:0; border-radius:0; outline:0; }}
QListWidget#nav::item {{
    padding:6px 9px; margin:1px 0; border:0; outline:0;
    border-radius:{R_CTRL}px; color:{TEXT_2};
}}
QListWidget#nav::item:hover {{ background:{SURFACE_ALT}; color:{TEXT}; }}
QListWidget#nav::item:selected {{
    background:{ACCENT_SOFT}; color:{ACCENT_PRESS}; border:0; outline:0;
    font-weight:600;
}}
QLabel#brand {{ font-size:13px; font-weight:700; color:{TEXT}; }}
QLabel#pageTitle {{ font-size:19px; font-weight:700; color:{TEXT}; }}
/* 更多菜单：图标 + 文字，行高 30，分隔线细一条 */
QMenu {{
    background:{SURFACE}; border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL}px;
    padding:6px;
}}
QMenu::item {{ padding:7px 26px 7px 30px; border-radius:{R_CTRL - 4}px; color:{TEXT}; }}
QMenu::item:selected {{ background:{ACCENT_SOFT}; color:{ACCENT_PRESS}; }}
QMenu::icon {{ left:10px; }}
QMenu::separator {{ height:1px; background:{BORDER}; margin:5px 8px; }}
QLabel#h1 {{ font-size:18px; font-weight:700; color:{TEXT}; }}
QLabel#h2 {{ font-size:14px; font-weight:600; color:{TEXT}; }}
QLabel#muted {{ color:{TEXT_3}; font-size:12px; }}
QLabel#thumb {{
    border:1px solid {BORDER_STRONG}; border-radius:{R_CTRL - 2}px;
    background:{SURFACE_ALT};
}}
QLabel#section {{ color:{TEXT_3}; font-size:11px; font-weight:700; letter-spacing:1px; }}
QLabel#pill {{
    background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:{R_PILL}px;
    padding:3px 11px; color:{TEXT_2}; font-size:12px;
}}
QLabel#pillAccent {{
    background:{ACCENT_SOFT}; border:1px solid {ACCENT_LINE}; border-radius:{R_PILL}px;
    padding:3px 11px; color:{ACCENT_PRESS}; font-size:12px; font-weight:600;
}}
"""


def web_qss_extra() -> str:
    """给内嵌 HTML（对话、漏洞地图、导出）用的配色说明，方便 markdown 里引用。"""
    return (f"正文 {TEXT}／次要 {TEXT_2}／强调 {ACCENT}／卡片 {SURFACE}／描边 {BORDER}")


# ---- 兼容旧名（有些地方还在按旧名引用） ------------------------------
def pill(text: str, accent: bool = False):
    """做一个小胶囊标签：状态、计数、站点名都用它，比裸文字整齐。"""
    from PySide6.QtWidgets import QLabel

    lb = QLabel(text)
    lb.setObjectName("pillAccent" if accent else "pill")
    return lb