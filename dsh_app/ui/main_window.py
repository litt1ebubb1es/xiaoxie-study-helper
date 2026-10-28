"""桌面主窗口（PySide6）。一个软件，双击即开，不用起 Web 服务。

界面分区对应方案的四个功能：
- 时间轴页：F1 回看点识别结果 + 负荷/行为曲线 + 回看点清单
- 重讲页：  F2 划段重讲（五种策略）
- 地图页：  F4 漏洞地图（分布、关联、复习顺序、根因）
- 设置页：  统一接入层（云端/本地切换、能力探测）+ 本地模型检测与一键部署
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QIcon,
                           QKeySequence, QShortcut, QTextCursor)
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QBoxLayout, QScrollArea, QSizePolicy, QSplitter, QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from .. import bili
from .. import capture as cap
from .. import maintenance
from ..cognitive import EVENT_KIND_CN, QUADRANT_CN
from ..config import (APP_NAME, AUDIO_DIR, CLOUD_PRESETS, DATA_DIR, Settings,
                      ensure_dirs, provider, vision_of)
from ..db import DB
from ..local_deploy import detect_all, one_click_deploy, build_deploy_script
from ..llm import build_llm
from ..pipeline import Engine, _hms
from ..review import STRATEGIES
from ..cognitive.mastery import DEFAULT_GATE, STATUS_CN
from ..whisper_setup import (SIZES as WHISPER_SIZES,
                             DEFAULT_SIZE as DEFAULT_WHISPER_SIZE,
                             status as whisper_status,
                             install_deps as install_whisper_deps,
                             download_model as download_whisper,
                             transcribe_file as transcribe_audio)
from . import icons, theme
from .chat_html import chat_html
from .timeline import KnotBand, TimelineWidget, _mmss

STYLE = theme.qss()

# 自动抓取短于这个秒数就当误触发丢掉（浏览器提示音、广告前奏、误点播放）。
# 真人讲课时，8 秒连一句话都说不完整，留着也没有分析价值。
AUTO_CAPTURE_MIN_KEEP_S = 8.0

# 六个页面：标题、副标题、图标。导航、页头、窗口标题都从这一张表来，
# 以前三处各写一遍，改一个页签名得改三个地方。
PAGES = [
    ("时间轴与回看点", "认知负荷和行为信号叠在一起看，哪一段卡住了一眼可见", "timeline"),
    ("划段重讲", "只重讲卡住的那一段，讲法跟着证据选，不从头上课", "reword"),
    ("小测与待答", "先问再讲：答对了才算过，答错进待答队列", "quiz"),
    ("漏洞地图", "知识点掌握度和前置关系，复习先补没补过的前置", "map"),
    ("问模型", "回答带知识库依据；课里没讲的会明说是通用知识", "chat"),
    ("设置与模型部署", "端点、模型、抓取方式、本地模型一键部署", "settings"),
]


def _section(title: str) -> QLabel:
    """小节标题：小号灰字 + 字距，比加粗大字更安静。"""
    lb = QLabel(title)
    lb.setObjectName("section")
    return lb


def _page() -> tuple[QWidget, QVBoxLayout]:
    """每个页签统一的版心：边距和间距只有一套，界面看起来才"匀"。"""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(theme.GAP_L, theme.GAP_M, theme.GAP_L, theme.GAP_L)
    v.setSpacing(theme.GAP_M)
    return w, v


def _card(pad: int = theme.GAP_M, spacing: int = theme.GAP_S,
          shadow: bool = False) -> tuple[QFrame, QVBoxLayout]:
    """白卡片容器：圆角 + 很浅的描边（可选投影）。

    表格/图表都放进卡片里，而不是各自画边框——一个界面只该有一种"框"。
    """
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(pad, pad, pad, pad)
    lay.setSpacing(spacing)
    if shadow:
        theme.soft_shadow(f)
    return f, lay


def _vruler() -> QFrame:
    """工具条上的分组竖线：比多留白更能说明"这几组不是一回事"。"""
    f = QFrame()
    f.setObjectName("divider")
    f.setFixedWidth(1)
    f.setFixedHeight(22)
    return f


def _chat_html(msgs: list[dict], model: str | None) -> str:
    """薄封装：真正的模板在 chat_html.py，方便单独调版式和做截图对比。"""
    return chat_html(msgs, model)


class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")


class WebCaptureBridge(QObject):
    """后台监听线程 → 主线程的通道。

    监听跑在普通 threading.Thread 里（它不碰界面），回调只 emit 信号；
    真正的起停录制、刷新界面都在主线程的槽里做——Qt 控件不能跨线程碰。
    """
    start = Signal(object)
    stop = Signal(object)
    state = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, db: DB):
        super().__init__()
        self.settings = settings
        self.db = db
        self._web_last_proc = ""
        self._web_sig = WebCaptureBridge()
        self._web_sig.start.connect(self._web_on_start)
        self._web_sig.stop.connect(self._web_on_stop)
        self._web_sig.state.connect(self._web_on_state)
        self._web_watch = None
        self.engine = Engine(db, settings)
        self.current_session = ""
        self.current_knots: list[Any] = []
        self._level = 0.0
        self.engine.on_level = self._on_capture_level
        self._workers: list[Worker] = []
        self._live_count = 0
        self._record_started_at = 0.0
        self._chat_msgs: list[dict] = []
        self._quiz_session = ""      # 题目面板属于哪一节：换节要清空，别显示别人的题
        self._tray: QSystemTrayIcon | None = None

        self.setWindowTitle(f"{APP_NAME} v0.2")
        self.setWindowIcon(icons.app_icon())
        self.resize(1440, 900)
        theme.apply_font(QApplication.instance() or QApplication([]))
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._build_shortcuts()
        self._refresh_sessions()
        self._update_backend_badge()
        self._build_tray()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._on_tick)
        self.ui_timer.start(250)

        self._start_autocap()
        QTimer.singleShot(400, self._startup_probe)
        # 上次是开着的话，这次接着盯（录制是用户明确打开过的动作，不重复问）
        QTimer.singleShot(700, self._maybe_autostart_web)
        QTimer.singleShot(900, self._safe_refresh_kp)
        QTimer.singleShot(500, self._refresh_kb)
        # 启动后收拾磁盘：清过期音频、当天第一次自动备份
        QTimer.singleShot(1200, self._startup_maintenance)
        QTimer.singleShot(1600, self._maybe_daily_backup)
        QTimer.singleShot(2200, self._check_review_notice)
        self.notice_timer = QTimer(self)
        self.notice_timer.timeout.connect(self._check_review_notice)
        self.notice_timer.start(30 * 60 * 1000)      # 每半小时看一眼要不要提醒

    # ------------------------------------------------------------------
    # 轻量状态文件：data/ui_state.json（和问模型小窗共用同一个文件）
    # ------------------------------------------------------------------
    def _ui_state_path(self) -> Path:
        """和小窗共用同一个文件：data/ui_state.json（两处各存一份必然打架）"""
        p = DATA_DIR / "ui_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_ui_state(self) -> dict:
        try:
            p = self._ui_state_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:                  # noqa: BLE001
            pass
        return {}

    def _save_ui_state(self, state: dict) -> None:
        try:
            p = self._ui_state_path()
            # 合并而不是覆盖：小窗那一半的键不能被主窗口写丢
            cur = self._load_ui_state()
            cur.update(state)
            p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:                  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 界面搭建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """骨架：左栏（品牌/导航/课程/录制）+ 右侧（页头 + 页面）。

        上一版把十个按钮全塞在顶部一条里，屏幕一窄就挤成两行、还和页签抢注意力。
        现在按"在哪用"分层：
        - 导航是唯一入口（顶部页签栏藏起来），竖排六项、带图标；
        - 录制三件套放左栏底部卡片里（它们和"正在上课"绑在一起，不在页面之间跳）；
        - 导入/补看/分析/导出这类低频操作收进页头「更多」菜单；
        - 页头只留标题、一句话说明、后端状态和「问模型小窗」。
        """
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = self._build_sidebar()
        root.addWidget(side)

        mainw = QWidget()
        mv = QVBoxLayout(mainw)
        mv.setContentsMargins(theme.GAP_L, theme.GAP_M, theme.GAP_L, theme.GAP_S)
        mv.setSpacing(theme.GAP_M)
        mv.addLayout(self._build_page_header())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_timeline_tab(), PAGES[0][0])
        self.tabs.addTab(self._build_reword_tab(), PAGES[1][0])
        self.tabs.addTab(self._build_quiz_tab(), PAGES[2][0])
        self.tabs.addTab(self._build_map_tab(), PAGES[3][0])
        self.tabs.addTab(self._build_chat_tab(), PAGES[4][0])
        # 设置页越加越长（设备/转写/生成层/部署/存储/数据），必须能滚：
        # 否则小屏幕上一裁，最后的「备份」就永远点不到。
        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QFrame.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._settings_scroll.setWidget(self._build_settings_tab())
        self.tabs.addTab(self._settings_scroll, PAGES[5][0])
        # 页签栏藏掉：导航在左栏，顶部再来一排页签就是两套导航抢同一件事
        self.tabs.tabBar().hide()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        mv.addWidget(self.tabs, 1)
        root.addWidget(mainw, 1)

        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.status.showMessage("就绪")

        self.nav.setCurrentRow(0)

    # -- 左栏 ------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(248)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(theme.GAP_M, theme.GAP_M, theme.GAP_M, theme.GAP_M)
        sv.setSpacing(theme.GAP_S)

        brand = QHBoxLayout()
        brand.setSpacing(theme.GAP_S)
        logo = QLabel()
        logo.setPixmap(icons.app_badge(30))
        brand.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(0)
        title = QLabel("小蟹学习助手")
        title.setObjectName("brand")
        names.addWidget(title)
        ver = QLabel("v0.2 · 端侧")
        ver.setObjectName("muted")
        names.addWidget(ver)
        brand.addLayout(names)
        brand.addStretch(1)
        sv.addLayout(brand)
        sv.addSpacing(theme.GAP_XS)

        sv.addWidget(_section("功能"))
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFrameShape(QFrame.NoFrame)
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for text, _sub, ic in PAGES:
            it = QListWidgetItem(icons.get(ic), text)
            it.setSizeHint(QSize(0, 34))
            self.nav.addItem(it)
        self.nav.setFixedHeight(len(PAGES) * 34 + theme.GAP_XS)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        sv.addWidget(self.nav)

        sv.addSpacing(theme.GAP_XS)
        # 入口要能被发现：躲在「更多」菜单里等于没有
        head_course = QHBoxLayout()
        head_course.setContentsMargins(0, 0, 0, 0)
        head_course.setSpacing(theme.GAP_XS)
        head_course.addWidget(_section("课程 / 节次"))
        head_course.addStretch(1)
        self.btn_bili_side = QPushButton("B 站登录")
        self.btn_bili_side.setObjectName("ghost")
        self.btn_bili_side.setIcon(icons.get("link"))
        self.btn_bili_side.setToolTip(
            "从 B 站导入整门课：扫码登录或借用浏览器登录，只搬标题、时长、链接。\n"
            "导入后点一节右键「看这一节」，播放就会自动开始录。")
        self.btn_bili_side.clicked.connect(self._on_bili_import)
        head_course.addWidget(self.btn_bili_side)
        sv.addLayout(head_course)
        self.session_list = QListWidget()
        self.session_list.setObjectName("plain")
        self.session_list.setWordWrap(True)
        self.session_list.setFrameShape(QFrame.NoFrame)
        self.session_list.itemActivated.connect(self._on_session_activated)
        self.session_list.itemClicked.connect(self._on_session_activated)
        # 用久了课程/节次一定会乱：右键就能整理，不用去翻数据库
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._on_session_menu)
        sv.addWidget(self.session_list, 1)
        self.lbl_side_stats = QLabel("")
        self.lbl_side_stats.setObjectName("muted")
        self.lbl_side_stats.setWordWrap(True)
        sv.addWidget(self.lbl_side_stats)

        # -- 录制卡：三个和"正在上课"绑定的动作 + 状态
        rec = QFrame()
        rec.setObjectName("recordCard")
        rv = QVBoxLayout(rec)
        rv.setContentsMargins(theme.GAP_S, theme.GAP_S, theme.GAP_S, theme.GAP_S)
        rv.setSpacing(theme.GAP_XS)

        self.btn_record = QPushButton("开始录制")
        self.btn_record.setObjectName("primary")
        self.btn_record.setIcon(icons.get("play", theme.ON_ACCENT, 15))
        self.btn_record.setMinimumHeight(32)
        rv.addWidget(self.btn_record)

        row = QHBoxLayout()
        row.setSpacing(theme.GAP_XS)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setIcon(icons.get("stop", theme.BAD, 15))
        self.btn_stop.setEnabled(False)
        self.btn_mark = QPushButton("标记回看点")
        self.btn_mark.setIcon(icons.get("flag", theme.ACCENT, 15))
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_mark)
        rv.addLayout(row)

        self.chk_autoweb = QCheckBox("看网页视频自动录")
        self.chk_autoweb.setToolTip(
            "勾上后：检测到浏览器在放声音就自动开始录，安静了自动停并分析。\n"
            "只认浏览器进程（Chrome/Edge/Firefox/国产壳），别的程序放声音不会触发。\n"
            "不区分站点：B站、学堂在线、学校自建平台、YouTube 都一样。")
        self.chk_autoweb.setChecked(bool(getattr(self.settings.capture, "auto_web_capture", False)))
        self.chk_autoweb.toggled.connect(self._on_autoweb_toggled)
        rv.addWidget(self.chk_autoweb)

        stat = QHBoxLayout()
        stat.setSpacing(theme.GAP_XS)
        self.lbl_rec = QLabel("未录制")
        self.lbl_rec.setObjectName("pill")
        stat.addWidget(self.lbl_rec)
        self.level = QProgressBar()
        self.level.setRange(0, 1000)
        self.level.setFixedHeight(6)
        self.level.setTextVisible(False)
        stat.addWidget(self.level, 1)
        rv.addLayout(stat)
        sv.addWidget(rec)

        self.btn_record.clicked.connect(self._on_record)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_mark.clicked.connect(self._on_mark)
        return side

    # -- 页头 ------------------------------------------------------------
    def _build_page_header(self) -> QHBoxLayout:
        head = QHBoxLayout()
        head.setSpacing(theme.GAP_S)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.lbl_page_title = QLabel(PAGES[0][0])
        self.lbl_page_title.setObjectName("pageTitle")
        titles.addWidget(self.lbl_page_title)
        self.lbl_page_sub = QLabel(PAGES[0][1])
        self.lbl_page_sub.setObjectName("muted")
        titles.addWidget(self.lbl_page_sub)
        head.addLayout(titles)
        head.addStretch(1)

        self.badge = QLabel("后端：未探测")
        self.badge.setObjectName("pill")
        head.addWidget(self.badge, 0, Qt.AlignVCenter)

        self.btn_chat_pop = QPushButton("问模型小窗")
        self.btn_chat_pop.setIcon(icons.get("chat"))
        self.btn_chat_pop.setToolTip("开一个能置顶的小对话窗，边看网课边问。\n"
                                     "截图 Ctrl+V 直接粘、图片/PDF 拖进去就能发给模型。")
        self.btn_chat_pop.clicked.connect(self._open_chat_window)
        head.addWidget(self.btn_chat_pop, 0, Qt.AlignVCenter)

        # 低频操作全收进这里：顶部只留"现在这一页要用的"
        self.btn_more = QPushButton("更多")
        self.btn_more.setIcon(icons.get("more"))
        self.menu_more = QMenu(self)
        self.btn_bili = QPushButton("从 B 站导入课程…")
        self.btn_bili.setToolTip("借用浏览器/扫码的登录态，把整门课的分集清单拉进来："
                                 "只搬标题、时长、链接，视频还在 B 站")
        self.btn_import = QPushButton("导入视频/音频…")
        self.btn_url = QPushButton("从链接导入…")
        self.btn_url.setToolTip("贴 B站/YouTube 等链接：有字幕只取字幕（不下载视频），\n"
                                "没字幕只下音轨再转写")
        self.btn_vision = QPushButton("补看回看点画面")
        self.btn_vision.setToolTip("对每个回看点，取那一刻的视频画面交给视觉模型识别，"
                                   "看当时屏幕上写了什么（只取故事板帧，不下载视频）")
        self.btn_analysis = QPushButton("重新分析")
        self.btn_demo = QPushButton("生成演示会话")
        self.btn_export = QPushButton("导出笔记")
        self.btn_export.setToolTip("把这一节课的回看点、掌握度、屏幕笔记、累积的问题"
                                   "装订成一份 markdown，存到 data/exports/")
        for b in (self.btn_bili, self.btn_import, self.btn_url, self.btn_vision,
                  self.btn_analysis, self.btn_demo, self.btn_export):
            b.hide()                       # 只当动作载体用：菜单点了就是点它
        entries = [
            ("从 B 站导入课程…", "link", self.btn_bili),
            ("导入视频/音频…", "upload", self.btn_import),
            ("从链接导入…", "link", self.btn_url),
            ("补看回看点画面", "eye", self.btn_vision),
            (None, None, None),
            ("重新分析", "refresh", self.btn_analysis),
            ("生成演示会话", "demo", self.btn_demo),
            ("导出笔记", "export", self.btn_export),
        ]
        for text, ic, btn in entries:
            if text is None:
                self.menu_more.addSeparator()
                continue
            act = self.menu_more.addAction(icons.get(ic), text)
            act.triggered.connect(btn.click)
        self.btn_more.setMenu(self.menu_more)
        head.addWidget(self.btn_more, 0, Qt.AlignVCenter)

        self.btn_bili.clicked.connect(self._on_bili_import)
        self.btn_import.clicked.connect(self._on_import)
        self.btn_analysis.clicked.connect(self._on_analyze)
        self.btn_demo.clicked.connect(self._on_demo)
        self.btn_export.clicked.connect(self._on_export_notes)
        return head

    # -- 导航联动 --------------------------------------------------------
    def _on_nav_changed(self, row: int) -> None:
        if row < 0 or row >= self.tabs.count():
            return
        if self.tabs.currentIndex() != row:
            self.tabs.setCurrentIndex(row)
        self._update_page_header(row)

    def _on_tab_changed(self, idx: int) -> None:
        # 代码里 setCurrentIndex 切页时也要把左侧导航和页头跟上
        if idx >= 0 and self.nav.currentRow() != idx:
            self.nav.blockSignals(True)
            self.nav.setCurrentRow(idx)
            self.nav.blockSignals(False)
        self._update_page_header(idx)

    def _update_page_header(self, idx: int) -> None:
        if not (0 <= idx < len(PAGES)):
            return
        text, sub, _ic = PAGES[idx]
        self.lbl_page_title.setText(text)
        self.lbl_page_sub.setText(sub)
        self.setWindowTitle(f"{APP_NAME} · {text}")
        self._sync_nav_icons(idx)

    def _sync_nav_icons(self, current: int) -> None:
        """当前项图标染橙、其余灰：QSS 管不到图标颜色，只能自己换一遍。"""
        for i, (_t, _s, ic) in enumerate(PAGES):
            color = theme.ACCENT if i == current else theme.TEXT_2
            item = self.nav.item(i)
            if item is not None:
                item.setIcon(icons.get(ic, color))

    def _build_timeline_tab(self) -> QWidget:
        w, v = _page()
        # 图表一张卡、清单一张卡：同一页只有两个"框"，看着才整齐
        chart_card, cl = _card(pad=theme.GAP_M, spacing=theme.GAP_S, shadow=True)
        # 图表卡里加一行小标题：长课（3 小时）全挤一屏时回看点只有几像素宽，
        # 得让人知道能放大，而不是以为"这条曲线就这样"。
        ch = QHBoxLayout()
        ch.setSpacing(theme.GAP_S)
        ch.addWidget(_section("认知负荷 / 行为信号"))
        self.lbl_zoom = QLabel("滚动缩放 · 拖动平移 · 双击复位")
        self.lbl_zoom.setObjectName("muted")
        ch.addWidget(self.lbl_zoom)
        ch.addStretch(1)
        self.lbl_zoom_state = QLabel("全貌")
        self.lbl_zoom_state.setObjectName("pill")
        ch.addWidget(self.lbl_zoom_state)
        self.btn_zoom_fit = QPushButton("适应窗口")
        self.btn_zoom_fit.setIcon(icons.get("monitor"))
        self.btn_zoom_fit.setToolTip("回到整节都看得见")
        ch.addWidget(self.btn_zoom_fit)
        cl.addLayout(ch)
        self.timeline = TimelineWidget()
        self.timeline.knotSelected.connect(self._on_knot_selected)
        self.timeline.viewChanged.connect(self._on_view_changed)
        cl.addWidget(self.timeline, 1)
        v.addWidget(chart_card, 1)
        self.btn_zoom_fit.clicked.connect(self.timeline.reset_view)

        list_card, tl = _card(pad=theme.GAP_M, spacing=theme.GAP_XS)
        self.knot_table = QTableWidget(0, 6)
        self.knot_table.setHorizontalHeaderLabels(["时间", "等级", "分数", "象限", "为什么标记这里", "状态"])
        self.knot_table.setFrameShape(QFrame.NoFrame)
        self.knot_table.setShowGrid(False)
        hdr = self.knot_table.horizontalHeader()
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        # 注意：循环变量不能叫 w —— 外层的 w 是正在建的 QWidget，覆盖掉会被回收
        for col, px in ((0, 78), (1, 68), (2, 62), (3, 96), (5, 78)):
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.knot_table.setColumnWidth(col, px)
        self.knot_table.verticalHeader().setVisible(False)
        self.knot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.knot_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.knot_table.cellDoubleClicked.connect(
            lambda r, c: self._open_reword_for_row(r))
        self.knot_table.setMinimumHeight(96)
        tl.addWidget(self.knot_table, 0)
        # 回看点操作条：选中一行之后能做的事，都摆在一处。
        # 双击是"进重讲"（已在表上），这里补上"这一刻是第几秒"和"我不同意"。
        #
        # 为什么不做内嵌回听/回放：软件手里只有音频，没有画面。听 20 秒原话
        # 解不了"这一步推导怎么来的"——那恰恰是卡住的地方。所以这里不装作
        # 能回看，只把时间点抄给用户，让他回原来的播放器拖过去。
        ops = QHBoxLayout()
        ops.setSpacing(theme.GAP_S)
        self.btn_copy_ts = QPushButton("复制时间点")
        self.btn_copy_ts.setIcon(icons.get("link"))
        self.btn_copy_ts.setToolTip("把这一行的时刻复制到剪贴板"
                                    "（视频还在原播放器里，自己拖到那一刻看）")
        self.btn_confirm_knot = QPushButton("确实卡住")
        self.btn_confirm_knot.setIcon(icons.get("check"))
        self.btn_confirm_knot.setToolTip("记一条正反馈，进校准样本")
        self.btn_dismiss_knot = QPushButton("不算回看点")
        self.btn_dismiss_knot.setIcon(icons.get("close"))
        self.btn_dismiss_knot.setToolTip("误判就标掉：它不再进重讲选项、统计和复习建议，"
                                         "但记录留着（以后调阈值要用）")
        self.btn_undo_knot = QPushButton("撤销")
        self.btn_undo_knot.setToolTip("把当前这行恢复成未被忽略")
        for b in (self.btn_copy_ts, self.btn_confirm_knot,
                  self.btn_dismiss_knot, self.btn_undo_knot):
            ops.addWidget(b)
        self.lbl_ts = QLabel("选中一行，把时间点抄下来回视频里看")
        self.lbl_ts.setObjectName("muted")
        ops.addWidget(self.lbl_ts, 1)
        tl.addLayout(ops)
        self.btn_copy_ts.clicked.connect(self._on_copy_timestamp)
        self.btn_confirm_knot.clicked.connect(self._on_confirm_knot)
        self.btn_dismiss_knot.clicked.connect(self._on_dismiss_knot)
        self.btn_undo_knot.clicked.connect(self._on_restore_knot)
        self.knot_table.itemSelectionChanged.connect(self._refresh_ts_hint)
        self.timeline.knotSelected.connect(lambda _id: self._refresh_ts_hint())
        self.knot_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.knot_table.customContextMenuRequested.connect(self._on_knot_menu)
        hint = QLabel("双击一行进入重讲。曲线：灰蓝=认知负荷 L，琥珀虚线=行为信号 B；"
                      "顶部缎带=回看点区间（砖橙=重点回看，浅琥珀=一般回看）。"
                      "要看那段画面就点「复制时间点」，回视频里拖过去。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        tl.addWidget(hint)
        self.lbl_coverage = QLabel("")
        self.lbl_coverage.setObjectName("muted")
        self.lbl_coverage.setWordWrap(True)
        tl.addWidget(self.lbl_coverage)
        v.addWidget(list_card, 0)
        return w

    def _build_reword_tab(self) -> QWidget:
        w, v = _page()
        # 一行到底、左对齐：回看点 → 讲法 → 生成 → 提问。不要中间插弹性空间，
        # 否则「策略」会被推到最右边，和回看点隔出一条空地，看着像两回事。
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_S)
        row.addWidget(QLabel("回看点"))
        self.reword_knot = QComboBox()
        self.reword_knot.setMinimumWidth(260)
        self.reword_knot.setMaximumWidth(340)
        row.addWidget(self.reword_knot, 0)
        row.addSpacing(theme.GAP_M)
        row.addWidget(QLabel("讲法"))
        self.reword_strategy = QComboBox()
        self.reword_strategy.setMinimumWidth(220)
        for s in STRATEGIES:
            self.reword_strategy.addItem(f"{s.name}　{s.fit}", s.key)
        row.addWidget(self.reword_strategy)
        self.btn_reword = QPushButton("生成重讲")
        self.btn_reword.setObjectName("primary")
        self.btn_reword.setIcon(icons.get("spark", theme.ON_ACCENT, 16))
        row.addWidget(self.btn_reword)
        self.btn_ask_knot = QPushButton("就这个回看点提问")
        self.btn_ask_knot.setToolTip("带着这个回看点的原文和依据切到「问模型」，"
                                     "模型会优先用你的知识库回答")
        row.addWidget(self.btn_ask_knot)
        row.addStretch(1)
        v.addLayout(row)

        self.reword_out = QTextBrowser()
        self.reword_out.setOpenExternalLinks(True)
        self.reword_out.setPlaceholderText(
            "选一个回看点和一种讲法，点「生成重讲」。\n"
            "重讲只用这一段的原文和你的知识库，讲完给一道小测验收——听懂了不算，答对才算。")
        v.addWidget(self.reword_out, 1)

        fb = QHBoxLayout()
        self.btn_useful = QPushButton("这个讲明白了（记为有用）")
        self.btn_mastered = QPushButton("这段我本来就会（排除）")
        fb.addWidget(self.btn_useful)
        fb.addWidget(self.btn_mastered)
        fb.addStretch(1)
        v.addLayout(fb)
        self.btn_reword.clicked.connect(self._on_reword)
        self.btn_ask_knot.clicked.connect(self._on_ask_current_knot)
        self.btn_useful.clicked.connect(lambda: self._feedback("confirm_useful"))
        self.btn_mastered.clicked.connect(lambda: self._feedback("mastered"))
        return w

    def _build_quiz_tab(self) -> QWidget:
        """小测 + 待答问题。

        为什么单开一页而不是塞进重讲页：小测是**证据生产**（会改掌握度），
        待答问题是**时机管理**（听课攒、课后答），两件事都不属于"重讲"。
        """
        w, v = _page()

        top = QHBoxLayout()
        top.addWidget(QLabel("回看点"))
        self.quiz_knot = QComboBox()
        self.quiz_knot.setMinimumWidth(300)
        self.quiz_knot.setMaximumWidth(430)
        top.addWidget(self.quiz_knot, 1)
        self.btn_quiz = QPushButton("出 3 道小测")
        self.btn_quiz.setIcon(icons.get("quiz", theme.ON_ACCENT, 16))
        self.btn_quiz.setObjectName("primary")
        self.btn_quiz.setToolTip("题只从这一段原文出；每题都带「支持答案的原句」，"
                                "出处查不到的题当场作废，不写掌握度")
        top.addWidget(self.btn_quiz)
        self.btn_import_ex = QPushButton("导入练习…")
        self.btn_import_ex.setToolTip("导入老师发的 xlsx / csv 题目；"
                                      "导入题的答案以文件为准，不查原文（模型题才查原文）")
        self.btn_quiz_from_ex = QPushButton("用练习出 3 道")
        self.btn_quiz_from_ex.setToolTip("从导入的练习里抽还没做过的题，判分与模型题同一套")
        self.btn_ex_template = QPushButton("练习模板")
        self.btn_ex_template.setToolTip("生成一份能直接填的模板（有 openpyxl 就是 xlsx）")
        top.addWidget(self.btn_import_ex)
        top.addWidget(self.btn_quiz_from_ex)
        top.addWidget(self.btn_ex_template)
        v.addLayout(top)

        exrow = QHBoxLayout()
        self.lbl_ex = QLabel("导入的练习：0 条")
        self.lbl_ex.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        self.lbl_ex.setWordWrap(True)
        exrow.addWidget(self.lbl_ex, 1)
        v.addLayout(exrow)

        nxt = QHBoxLayout()
        self.btn_next = QPushButton("下一步做什么")
        self.btn_next.setToolTip("按「该复习 > 没过门槛 > 开新节」的顺序挑一件事，"
                                 "并说明为什么是它")
        nxt.addWidget(self.btn_next)
        self.lbl_next = QLabel("")
        self.lbl_next.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        self.lbl_next.setWordWrap(True)
        self.lbl_next.setTextInteractionFlags(Qt.TextSelectableByMouse)
        nxt.addWidget(self.lbl_next, 1)
        v.addLayout(nxt)

        self.lbl_quiz = QLabel("还没测过。掌握度靠答题实测，不靠自己点「讲明白了」。")
        self.lbl_quiz.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        self.lbl_quiz.setWordWrap(True)
        v.addWidget(self.lbl_quiz)

        self.quiz_area = QScrollArea()
        self.quiz_area.setWidgetResizable(True)
        # 滚动区自己不要再画一层框：外层已经是卡片，双框会显得"嵌了两层"
        self.quiz_area.setFrameShape(QFrame.NoFrame)
        self.quiz_area.setStyleSheet("QScrollArea{background:transparent;}")
        self.quiz_holder = QWidget()
        self.quiz_holder.setObjectName("quizHolder")
        self.quiz_holder.setStyleSheet("QWidget#quizHolder{background:transparent;}")
        self.quiz_layout = QVBoxLayout(self.quiz_holder)
        self.quiz_layout.setContentsMargins(0, 0, 0, 0)
        self.quiz_layout.setSpacing(theme.GAP_M)
        self.quiz_layout.addStretch(1)
        self.quiz_area.setWidget(self.quiz_holder)
        v.addWidget(self.quiz_area, 2)

        qh = QHBoxLayout()
        ql = QLabel("待答问题（听课时先攒着，下课集中答）")
        ql.setStyleSheet("font-weight:600;")
        qh.addWidget(ql)
        self.lbl_q_count = QLabel("")
        self.lbl_q_count.setStyleSheet(f"color:{theme.TEXT_3};")
        qh.addWidget(self.lbl_q_count)
        qh.addStretch(1)
        self.btn_q_ans = QPushButton("回答选中")
        self.btn_q_all = QPushButton("全部回答")
        self.btn_q_del = QPushButton("删除选中")
        for b in (self.btn_q_ans, self.btn_q_all, self.btn_q_del):
            qh.addWidget(b)
        v.addLayout(qh)

        row = QHBoxLayout()
        self.edit_q = QLineEdit()
        self.edit_q.setPlaceholderText("现在想不通的点，先记下来（Enter 攒进队列，不打断听课）")
        row.addWidget(self.edit_q, 1)
        self.btn_q_add = QPushButton("攒着")
        row.addWidget(self.btn_q_add)
        v.addLayout(row)

        self.q_list = QListWidget()
        # 待答清单给够高度：空题目区占着一大半屏，下面两张卡却只有两行，页面重心就飘了
        self.q_list.setMinimumHeight(120)
        self.q_list.setMaximumHeight(230)
        v.addWidget(self.q_list, 1)
        self.q_out = QTextBrowser()
        self.q_out.setMinimumHeight(96)
        self.q_out.setMaximumHeight(180)
        self.q_out.setPlaceholderText("选中一条点「回答」看答案")
        v.addWidget(self.q_out, 0)

        self.btn_quiz.clicked.connect(self._on_make_quiz)
        self.btn_next.clicked.connect(self._on_next_action)
        self.btn_import_ex.clicked.connect(self._on_import_exercises)
        self.btn_quiz_from_ex.clicked.connect(self._on_quiz_from_exercises)
        self.btn_ex_template.clicked.connect(self._on_exercise_template)
        self.btn_q_add.clicked.connect(self._on_queue_question)
        self.edit_q.returnPressed.connect(self._on_queue_question)
        self.btn_q_ans.clicked.connect(self._on_answer_question)
        self.btn_q_all.clicked.connect(self._on_answer_all)
        self.btn_q_del.clicked.connect(self._on_drop_question)
        self.q_list.currentItemChanged.connect(self._on_question_selected)
        return w

    # -- 小测 -----------------------------------------------------------
    def refresh_quiz_knots(self) -> None:
        """回看点下拉与「划段重讲」页共用 current_knots，但要显示小节号。"""
        keep = self.quiz_knot.currentData()
        self.quiz_knot.clear()
        knots = self.current_knots or []
        if not knots and self.current_session:
            try:
                knots = self.db.get_knots(self.current_session)
            except Exception:  # noqa: BLE001
                knots = []
        sess_idx = {}
        try:
            for s in self.db.list_sessions():
                sess_idx[str(s["id"])] = int(s["idx"] or 0)
        except Exception:  # noqa: BLE001
            pass
        for k in sorted(knots, key=lambda x: x.t_start):
            n = sess_idx.get(str(k.session_id), 0)
            gname = {"strong": "强", "medium": "中", "mid": "中"}.get(str(k.grade), "候选")
            label = f"第 {n} 节 {_hms(k.t_start)}　{gname}回看点　分数 {k.score:.2f}"
            self.quiz_knot.addItem(label, int(k.id))
            self.quiz_knot.setItemData(self.quiz_knot.count() - 1,
                                       (k.excerpt or "")[:300], Qt.ToolTipRole)
        if keep is not None:
            i = self.quiz_knot.findData(keep)
            if i >= 0:
                self.quiz_knot.setCurrentIndex(i)
        self.refresh_quiz_hint()

    def refresh_quiz_hint(self) -> None:
        sid = self.current_session
        if not sid:
            self.lbl_quiz.setText("先选一节课。")
            return
        try:
            ov = self.engine.quiz_overview(sid)
        except Exception as e:  # noqa: BLE001
            self.lbl_quiz.setText(f"小测状态读不出来：{e}")
            return
        st = ov["stats"]
        if not st["total"]:
            self.lbl_quiz.setText("这一节还没测过。出 3 道题，答对才算证据。")
        # 没有题目时补一张空状态卡：空白屏看不出"这里该有什么"
        self._ensure_quiz_placeholder()
        if not st["total"]:
            pass
        else:
            self.lbl_quiz.setText(
                f"这一节共 {st['total']} 道题，已答 {st['answered']} 道，"
                f"答对 {st['correct']} 道。作废的题（出处查不到）不计入掌握度。")

    def _on_next_action(self) -> None:
        """下一步做什么：复习优先，其次没过门槛的，最后才建议开新节。"""
        cid = self._current_course_id()
        if not cid:
            self.lbl_next.setText("还没有课程数据。")
            return
        try:
            res = self.engine.next_action(cid)
        except Exception as e:  # noqa: BLE001
            self.lbl_next.setText(f"算不出来：{e}")
            return
        self.lbl_next.setText(res.get("why") or "")
        kid = res.get("knot_id")
        act = res.get("action")
        strat = (res.get("strategy") or {}).get("name") or ""
        hint = ""
        if kid is not None:
            i = self.quiz_knot.findData(int(kid))
            if i >= 0:
                self.quiz_knot.setCurrentIndex(i)
            if act == "replay":
                hint = (f"　→ 已选好回看点：先去「划段重讲」按「{strat}」重讲一遍，"
                        f"再回来出题验证")
            else:
                hint = "　→ 已选好回看点，点「出 3 道小测」"
            if res.get("locked"):
                hint += "（门槛未过：这条是闸门锁定的，做题不算过，得有证据）"
        self.lbl_next.setText((res.get("why") or "") + hint)

    def _clear_quiz_area(self, with_placeholder: bool = True) -> None:
        """清空题目区。

        ``with_placeholder=False`` 用于"马上要放真题"的场景：空状态卡不要再加进去，
        否则它会占住列表第一位，把新出的题顶到看不见的地方（表现为"出题出不来"）。
        """
        while self.quiz_layout.count():
            item = self.quiz_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # 空状态紧跟操作行往下排：上留 1 段小空、下留大空，页面才有"从上往下读"的顺序。
        # 上下各 1 段等分弹性时，卡片会浮在页面中间，和上面那排控件断开。
        if with_placeholder:
            self.quiz_layout.addSpacing(theme.GAP_S)
            self.quiz_layout.addWidget(self._quiz_empty_card())
        self.quiz_layout.addStretch(1)

    def _ensure_quiz_placeholder(self) -> None:
        """题目区只有 stretch 时补一张说明卡。

        插在**最前面**（index 0）：布局里那条 stretch 是留给下方空白用的，
        卡片插到它后面就会被顶到页面底部，上面空出一大片。
        """
        if not hasattr(self, "quiz_layout"):
            return
        cards = [self.quiz_layout.itemAt(i).widget() for i in range(self.quiz_layout.count())]
        if any(c is not None for c in cards):
            return
        self.quiz_layout.insertWidget(0, self._quiz_empty_card())

    def _quiz_empty_card(self) -> QWidget:
        """没有题目时的空状态。

        不只放一句"还没有小测"——那会在页面上留一大片白。这里排成三步引导，
        顺便把"为什么非要答题不可"讲清楚（点「讲明白了」不算证据）。
        """
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(theme.GAP_XL, theme.GAP_L, theme.GAP_XL, theme.GAP_L)
        lay.setSpacing(theme.GAP_M)

        title = QLabel("还没有小测")
        title.setObjectName("h2")
        lay.addWidget(title)

        steps = QHBoxLayout()
        steps.setSpacing(theme.GAP_L)
        # 手工折行而不是 setWordWrap：滚动区里 wordWrap 标签的高度会被算矮，
        # 三行文字只画一行、压在卡片边框上。显式换行后高度是确定的。
        guide = (
            ("1", "选回看点", "上面下拉里挑一段——\n回看点就是这节课\n卡住你的地方"),
            ("2", "出 3 道题", "题只用那一段原文出，\n每题都带\n「支持答案的原句」"),
            ("3", "答对才算过", "出处查不到的题当场作废；\n答对才改掌握度，\n自评不算"),
        )
        for num, head, body in guide:
            col = QVBoxLayout()
            col.setSpacing(theme.GAP_XS)
            badge = QLabel(num)
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background:{theme.ACCENT_SOFT}; color:{theme.ACCENT_PRESS};"
                f"border-radius:11px; font-weight:700; font-size:12px;")
            col.addWidget(badge)
            h = QLabel(head)
            h.setStyleSheet(f"color:{theme.TEXT}; font-weight:600; font-size:13px;")
            col.addWidget(h)
            b = QLabel(body)
            b.setObjectName("muted")
            b.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            col.addWidget(b)
            col.addStretch(1)
            steps.addLayout(col, 1)
        lay.addLayout(steps)
        card.setMaximumWidth(980)
        card.setMinimumHeight(150)
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return card

    def _on_make_quiz(self) -> None:
        kid = self.quiz_knot.currentData()
        if kid is None:
            QMessageBox.information(self, "提示", "这一节还没有回看点，先做分析")
            return
        self.btn_quiz.setEnabled(False)
        self.lbl_quiz.setText("正在出题…（题目必须带得出处的原句，模型给的题会被逐条校验）")
        w = Worker(self.engine.make_quiz, int(kid))
        w.done.connect(self._quiz_done)
        w.failed.connect(self._quiz_failed)
        self._spawn(w)

    def _quiz_failed(self, err: str) -> None:
        self.btn_quiz.setEnabled(True)
        self.lbl_quiz.setText(f"出题失败：{err.splitlines()[0][:180]}")
        self.status.showMessage("出题失败", 8000)

    # -- 练习导入 --------------------------------------------------------
    def _on_exercise_template(self) -> None:
        from .. import exercises as ex
        from ..config import EXPORT_DIR

        try:
            p = ex.write_template(EXPORT_DIR / "练习模板.xlsx")
        except Exception as e:                      # noqa: BLE001
            QMessageBox.warning(self, "生成模板失败", str(e))
            return
        self.status.showMessage(f"模板已生成：{p}", 12000)
        QMessageBox.information(
            self, "练习模板",
            f"已经写到：\n{p}\n\n"
            "列：知识点 / 题干 / 选项A-D / 答案 / 解析。\n"
            "答案可以写 A-D，也可以写 1-4 或选项原文。")

    def _on_import_exercises(self) -> None:
        cid = self._current_course_id()
        if not cid:
            self.status.showMessage("先选一门课", 6000)
            return
        from ..config import EXPORT_DIR

        path, _ = QFileDialog.getOpenFileName(
            self, "导入练习（xlsx / csv）", str(EXPORT_DIR),
            "练习文件 (*.xlsx *.xlsm *.csv *.txt);;所有文件 (*)")
        if not path:
            return
        self.status.showMessage("正在导入练习…")
        w = Worker(self.engine.import_exercises, path, cid)
        w.done.connect(self._import_exercises_done)
        w.failed.connect(lambda e: QMessageBox.warning(self, "导入失败", str(e)))
        self._spawn(w)

    def _import_exercises_done(self, res: dict) -> None:
        self.refresh_exercise_label()
        if not res.get("ok"):
            msg = res.get("error") or "导入失败"
            lines = [f"第 {e['row_no']} 行：{e['reason']}" for e in (res.get("errors") or [])[:8]]
            QMessageBox.warning(self, "导入失败", msg + ("\n" + "\n".join(lines) if lines else ""))
            return
        msg = f"导入 {res['imported']} 条"
        if res.get("skipped"):
            msg += f"，跳过 {res['skipped']} 条"
        if res.get("unmatched"):
            msg += f"；{len(res['unmatched'])} 个知识点名没对上（{('、'.join(res['unmatched'][:3]))}）"
        self.status.showMessage(msg, 15000)
        if res.get("errors"):
            lines = [f"第 {e['row_no']} 行：{e['reason']}" for e in res["errors"][:8]]
            QMessageBox.information(self, "导入结果", msg + "\n\n跳过原因：\n" + "\n".join(lines))
        else:
            QMessageBox.information(self, "导入结果", msg)

    def _on_quiz_from_exercises(self) -> None:
        if not self.current_session:
            return
        cid = self._current_course_id()
        self.btn_quiz_from_ex.setEnabled(False)
        self.lbl_quiz.setText("正在从导入的练习里抽题…")
        w = Worker(lambda: self.engine.quiz_from_exercises(cid, self.current_session, k=3))
        w.done.connect(self._quiz_from_ex_done)
        w.failed.connect(self._quiz_failed)
        self._spawn(w)

    def _quiz_from_ex_done(self, res: dict) -> None:
        self.btn_quiz_from_ex.setEnabled(True)
        if not res.get("ok"):
            self.lbl_quiz.setText(res.get("error") or "没有可用的导入练习")
            self.status.showMessage(res.get("error") or "没有可用的导入练习", 9000)
            return
        items = res.get("items") or []
        self._clear_quiz_area(with_placeholder=False)
        self._quiz_session = self.current_session
        header = QLabel("这组题来自你导入的练习：答案以文件为准（不做原文回查）。"
                        "判分规则和模型题完全一样。")
        header.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        header.setWordWrap(True)
        self.quiz_layout.insertWidget(self.quiz_layout.count() - 1, header)
        for i, item in enumerate(items, 1):
            self.quiz_layout.insertWidget(self.quiz_layout.count() - 1,
                                          self._quiz_block(i, dict(item, void=False)))
        self.lbl_quiz.setText(f"从导入的练习出了 {len(items)} 道题。选完点每题的「提交」。")
        self.refresh_quiz_hint()

    def refresh_exercise_label(self) -> None:
        cid = self._current_course_id()
        if not cid:
            self.lbl_ex.setText("导入的练习：0 条")
            return
        try:
            ov = self.engine.exercises_overview(cid)
        except Exception:                            # noqa: BLE001
            self.lbl_ex.setText("导入的练习：读取失败")
            return
        txt = f"导入的练习：{ov['total']} 条（{ov['answered_kp']} 条能对上知识点）"
        if ov["unmatched"]:
            txt += "；没对上的知识点名：" + "、".join(ov["unmatched"][:4])
        self.lbl_ex.setText(txt)

    def _quiz_done(self, res: dict) -> None:
        self.btn_quiz.setEnabled(True)
        if not res.get("ok"):
            msg = res.get("error") or "出题失败"
            if res.get("dropped"):
                msg += "；丢弃原因：" + "，".join(res["dropped"][:3])
            self.lbl_quiz.setText(msg)
            self.status.showMessage(msg[:160], 12000)
            return
        items = res.get("items") or []
        self._clear_quiz_area(with_placeholder=False)
        self._quiz_session = self.current_session
        self.quiz_layout.insertWidget(self.quiz_layout.count() - 1,
                                      self._quiz_header(res))
        for i, item in enumerate(items, 1):
            self.quiz_layout.insertWidget(self.quiz_layout.count() - 1,
                                          self._quiz_block(i, item))
        note = f"出了 {len(items)} 道题"
        if res.get("void"):
            note += f"（其中 {len(res['void'])} 道出处查不到，已作废、不计分）"
        if res.get("degraded"):
            note += "；这次是降档重问才拿到的"
        if res.get("dropped"):
            note += f"；丢弃 {len(res['dropped'])} 道不合规的题"
        self.lbl_quiz.setText(note + "。选完点每题的「提交」。")
        self.refresh_quiz_hint()

    def _quiz_header(self, res: dict) -> QWidget:
        btn = QPushButton("重出这一组")
        btn.setObjectName("ghost")
        btn.clicked.connect(self._on_make_quiz)
        lab = QLabel("出处逐题校验：引文必须能在本节原文里查到，查不到就作废。")
        lab.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 4)
        h.addWidget(lab, 1)
        h.addWidget(btn)
        return row

    def _quiz_block(self, idx: int, item: dict) -> QWidget:
        box = QFrame()
        box.setObjectName("quizCard")
        box.setStyleSheet(
            f"#quizCard {{background:{theme.SURFACE}; border:1px solid {theme.BORDER};"
            f" border-radius:8px;}}")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 10, 12, 10)
        q = QLabel(f"{idx}. {item['prompt']}")
        q.setWordWrap(True)
        q.setStyleSheet("font-weight:600;")
        v.addWidget(q)
        group = QButtonGroup(box)
        for i, ch in enumerate(item.get("choices") or []):
            rb = QRadioButton(f"{chr(65 + i)}. {ch}")
            rb.setProperty("qitem", int(item["item_id"]))
            rb.setProperty("qindex", i)
            group.addButton(rb, i)
            v.addWidget(rb)
        foot = QHBoxLayout()
        sub = QPushButton("提交")
        sub.setObjectName("primary")
        fb = QLabel("")
        fb.setWordWrap(True)
        foot.addWidget(sub)
        foot.addWidget(fb, 1)
        v.addLayout(foot)
        if item.get("void"):
            fb.setText(f"⚠️ 这道题作废：{item.get('void_reason') or '出处查不到'}（不写掌握度）")
            fb.setStyleSheet(f"color:{theme.WARN};")
            sub.setEnabled(False)
        else:
            def _grade(_=False, it=item, g=group, lab=fb, btn=sub) -> None:
                chosen = g.checkedId()
                if chosen is None or chosen < 0:
                    lab.setText("先选一个选项。")
                    return
                res = self.engine.answer_quiz(int(it["item_id"]), int(chosen))
                btn.setEnabled(False)
                if res.get("void"):
                    lab.setText(f"⚠️ 作废：{res.get('reason')}")
                    lab.setStyleSheet(f"color:{theme.WARN};")
                    return
                if res.get("correct"):
                    lab.setText(f"✅ 对了。依据：{res.get('evidence', '')[:120]}")
                    lab.setStyleSheet(f"color:{theme.OK};")
                else:
                    lab.setText(f"❌ {res.get('reason')}。依据：{res.get('evidence', '')[:120]}")
                    lab.setStyleSheet(f"color:{theme.BAD};")
                kp = res.get("kp") or {}
                if kp:
                    self.status.showMessage(
                        f"知识点「{kp.get('name')}」掌握度 {kp.get('mastery', 0) * 100:.0f}%"
                        f"（{STATUS_CN.get(kp.get('status'), kp.get('status'))}）", 12000)
                self.refresh_quiz_hint()
                try:
                    self.refresh_kp_table()
                except Exception:  # noqa: BLE001
                    pass

            sub.clicked.connect(_grade)
        return box

    # -- 待答问题 -------------------------------------------------------
    def refresh_questions(self) -> None:
        self.q_list.clear()
        sid = self.current_session
        if not sid:
            self.lbl_q_count.setText("")
            return
        try:
            rows = self.db.get_questions(sid, None)
        except Exception:  # noqa: BLE001
            rows = []
        pending = 0
        for r in rows:
            status = str(r["status"] or "pending")
            tag = {"pending": "待答", "answered": "已回答", "dropped": "已丢弃"}.get(status, status)
            if status == "pending":
                pending += 1
            it = QListWidgetItem(f"{_hms(float(r['t_sec'] or 0))}　[{tag}]　{r['text'][:70]}")
            it.setData(Qt.UserRole, int(r["id"]))
            it.setToolTip(str(r["excerpt"] or ""))
            if status != "pending":
                it.setForeground(QBrush(QColor(f"{theme.TEXT_3}")))
            self.q_list.addItem(it)
        self.lbl_q_count.setText(f"（待答 {pending} 条）" if rows else "（还没有）")

    def _on_queue_question(self) -> None:
        text = self.edit_q.text().strip()
        if not text:
            return
        sid = self.current_session
        if not sid:
            QMessageBox.information(self, "提示", "先选一节课")
            return
        # 录音中就用录音时长，否则用这一节"最后一个事件"的位置
        t = None
        try:
            if self.engine.recorder is not None:
                t = float(self.engine.recorder.duration)
        except Exception:  # noqa: BLE001
            t = None
        res = self.engine.queue_question(sid, text, t)
        if not res.get("ok"):
            self.status.showMessage(res.get("error") or "攒不进去", 8000)
            return
        self.edit_q.clear()
        self.refresh_questions()
        self.status.showMessage(f"已攒下（第 {_hms(res['t_sec'])} 处），下课后一起答", 8000)

    def _on_question_selected(self, cur, _prev=None) -> None:
        if cur is None:
            return
        qid = cur.data(Qt.UserRole)
        rows = self.db.query("SELECT * FROM pending_questions WHERE id=?", (int(qid),))
        if not rows:
            return
        r = rows[0]
        body = f"问题：{r['text']}"
        if r["answer"]:
            body += f"\n\n回答：{r['answer']}"
        self.q_out.setPlainText(body)

    def _on_answer_question(self) -> None:
        it = self.q_list.currentItem()
        if it is None:
            self.status.showMessage("先选一条问题", 5000)
            return
        self.q_out.setPlainText("正在回答…")
        w = Worker(self.engine.answer_question, int(it.data(Qt.UserRole)))
        w.done.connect(self._answer_done)
        w.failed.connect(lambda e: self.q_out.setPlainText(f"回答失败：{e}"))
        self._spawn(w)

    def _answer_done(self, res: dict) -> None:
        if not res.get("ok"):
            self.q_out.setPlainText(f"没答出来：{res.get('error') or '未知原因'}")
        else:
            self.q_out.setPlainText(f"回答：{res.get('answer')}")
        self.refresh_questions()

    def _on_answer_all(self) -> None:
        sid = self.current_session
        if not sid:
            return
        self.q_out.setPlainText("正在把待答问题逐条答完…")
        w = Worker(self.engine.answer_all_questions, sid)
        w.done.connect(self._answer_all_done)
        w.failed.connect(lambda e: self.q_out.setPlainText(f"批量回答失败：{e}"))
        self._spawn(w)

    def _answer_all_done(self, res: dict) -> None:
        msg = f"答了 {res.get('answered', 0)} 条，还剩 {res.get('remaining', 0)} 条。"
        if res.get("failed"):
            msg += " 失败 " + "、".join(str(f["id"]) for f in res["failed"][:5])
        self.q_out.setPlainText(msg)
        self.status.showMessage(msg, 10000)
        self.refresh_questions()

    def _on_drop_question(self) -> None:
        it = self.q_list.currentItem()
        if it is None:
            return
        self.db.delete_question(int(it.data(Qt.UserRole)))
        self.refresh_questions()
        self.q_out.clear()

    def _build_map_tab(self) -> QWidget:
        w, v = _page()
        top = QHBoxLayout()
        self.btn_map = QPushButton("生成漏洞地图")
        self.btn_map.setIcon(icons.get("map", theme.ON_ACCENT, 16))
        self.btn_map.setObjectName("primary")
        self.btn_calib = QPushButton("用反馈校准权重")
        self.btn_rekp = QPushButton("重建知识点")
        self.btn_rekp.setToolTip("不重听音频，只按现存转写重算知识点；"
                                 "术语抽取规则升级后用这个，而不用重新分析一遍")
        top.addWidget(self.btn_map)
        top.addWidget(self.btn_calib)
        top.addWidget(self.btn_rekp)
        top.addStretch(1)
        v.addLayout(top)

        # 知识点掌握度：回看点什么"熄不熄灭"看这张表，不是看地图文字
        kh = QHBoxLayout()
        self.lbl_kp = QLabel("知识点掌握度（点一行看它出现在哪些回看点）")
        self.lbl_kp.setObjectName("hint")
        kh.addWidget(self.lbl_kp, 1)
        self.btn_kp_ok = QPushButton("我会了（记为通过）")
        self.btn_kp_ok.setToolTip("等于一次自测答对；门控是一把硬尺子，到 0.9 才算补上")
        self.btn_kp_no = QPushButton("还不会")
        self.btn_kp_no.setToolTip("记一次答错：近因加权会明显拉低掌握度")
        kh.addWidget(self.btn_kp_ok)
        kh.addWidget(self.btn_kp_no)
        v.addLayout(kh)
        table_card, tc = _card(pad=theme.GAP_S, spacing=0)
        self.kp_table = QTableWidget(0, 6)
        self.kp_table.setHorizontalHeaderLabels(
            ["知识点", "掌握度", "状态", "记忆保持", "回看点数", "证据次数"])
        self.kp_table.setFrameShape(QFrame.NoFrame)
        self.kp_table.setShowGrid(False)
        hdr = self.kp_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for c in range(1, 6):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.kp_table.verticalHeader().setVisible(False)
        self.kp_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.kp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.kp_table.setMinimumHeight(150)
        self.kp_table.itemSelectionChanged.connect(self._on_kp_selected)
        tc.addWidget(self.kp_table)
        v.addWidget(table_card)
        self._kp_card = table_card

        self.map_view = QTextBrowser()
        self.map_view.setPlaceholderText(
            "点上面任一知识点，这里列出它出现在哪些回看点、依据是什么。\n"
            "点「生成漏洞地图」出整节课的报告：回看点清单、热区分布、跨节关联、复习顺序。")
        self.map_view.setMinimumHeight(180)
        # 详情装进卡片并加一行标题：底部那块区域很高，什么都不写就像"页面塌了一块"，
        # 加个"详情 / 报告"的头，空着也像在等你看，而不像没做完。
        detail_card, dcv = _card(pad=theme.GAP_S, spacing=theme.GAP_XS)
        dh = QHBoxLayout()
        dh.setSpacing(theme.GAP_S)
        dh.addWidget(_section("详情 / 报告"))
        dh.addStretch(1)
        self.lbl_map_hint = QLabel("选中知识点看回看点依据；点「生成漏洞地图」看整节报告")
        self.lbl_map_hint.setObjectName("muted")
        dh.addWidget(self.lbl_map_hint)
        dcv.addLayout(dh)
        dcv.addWidget(self.map_view, 1)
        v.addWidget(detail_card, 1)
        self.btn_map.clicked.connect(self._on_map)
        self.btn_calib.clicked.connect(self._on_calibrate)
        self.btn_rekp.clicked.connect(self._on_rebuild_knowledge)
        self.btn_kp_ok.clicked.connect(lambda: self._on_kp_mark(True))
        self.btn_kp_no.clicked.connect(lambda: self._on_kp_mark(False))
        return w

    # -- 知识点掌握度 ---------------------------------------------------
    def _fit_kp_table(self) -> None:
        rows = max(1, self.kp_table.rowCount())
        row_h = self.kp_table.verticalHeader().defaultSectionSize() or 26
        want = self.kp_table.horizontalHeader().height() + row_h * rows + 10
        # 只设 max 的话布局按 sizeHint 给高度，表格永远只露出五六行；
        # 所以上下都钉住，超出上限就让表格自己滚。
        h = max(200, min(want, 460))
        self.kp_table.setMinimumHeight(h)
        self.kp_table.setMaximumHeight(h)
        if getattr(self, "_kp_card", None) is not None:
            self._kp_card.setMaximumHeight(h + 2 * theme.GAP_S)

    def refresh_kp_table(self, keep_kp_id: int | None = None) -> None:
        """重画掌握度表。keep_kp_id：重画后把选中行落回这个知识点上。

        为什么要保留选中：点「我会了」之后整表会重画，选中行会掉到第一行，
        右边详情就显示成别的知识点了——看着像"点了没生效"。
        """
        cid = self._current_course_id()
        rows = self.engine.kp_overview(cid) if cid else []
        self.kp_table.setRowCount(0)
        for r in rows:
            i = self.kp_table.rowCount()
            self.kp_table.insertRow(i)
            name = QTableWidgetItem(r["name"])
            name.setData(Qt.UserRole, r["id"])
            self.kp_table.setItem(i, 0, name)
            pct = int(round(r["mastery"] * 100))
            m = QTableWidgetItem(f"{pct}%")
            color = {"mastered": f"{theme.OK}", "learning": f"{theme.WARN}",
                     "weak": f"{theme.BAD}", "unknown": f"{theme.TEXT_3}"}.get(r["status"], f"{theme.TEXT_3}")
            m.setForeground(QBrush(QColor(color)))
            self.kp_table.setItem(i, 1, m)
            self.kp_table.setItem(i, 2, QTableWidgetItem(STATUS_CN[r["status"]]))
            # 记忆保持：掌握度是"现在会不会"，这一列是"现在还记得多少"
            ret = r.get("retention")
            if ret is None:
                cell = QTableWidgetItem("—")
                cell.setForeground(QBrush(QColor(f"{theme.TEXT_3}")))
            else:
                cell = QTableWidgetItem(f"{int(round(ret * 100))}%")
                if r.get("overdue"):
                    cell.setText(f"{int(round(ret * 100))}% 该复习")
                    cell.setForeground(QBrush(QColor(f"{theme.WARN}")))
                    cell.setToolTip(f"稳定度 {r.get('stability_days', 0)} 天，"
                                    f"已过 {r.get('elapsed_h', 0)} 小时")
                else:
                    cell.setForeground(QBrush(QColor(f"{theme.TEXT_3}")))
                    cell.setToolTip(f"稳定度 {r.get('stability_days', 0)} 天，"
                                    f"还有约 {r.get('due_in_h', 0)} 小时到期")
            self.kp_table.setItem(i, 3, cell)
            self.kp_table.setItem(i, 4, QTableWidgetItem(str(r["knots"])))
            self.kp_table.setItem(i, 5, QTableWidgetItem(str(r["attempts"])))
        n_done = sum(1 for r in rows if r["status"] == "mastered")
        n_due = sum(1 for r in rows if r.get("overdue"))
        tail = f"，{n_due} 个该复习" if n_due else ""
        n_ret = 0
        try:
            n_ret = self.db.count_retired_points(cid) if cid else 0
        except Exception:                        # noqa: BLE001
            n_ret = 0
        # 退役的要写出来：不然用户会问"我明明见过『必要条件』，怎么没了"
        ret_txt = f"，{n_ret} 个已退役（旧规则抽的，没有回看点可重讲）" if n_ret else ""
        self.lbl_kp.setText(
            f"知识点掌握度：{len(rows)} 个，已补上 {n_done} 个{tail}{ret_txt}"
            f"（门槛 {int(round(DEFAULT_GATE * 100))}%，点一行看它出现在哪些回看点）")
        self._fit_kp_table()
        # 重画后把选中行落回原来那个知识点，并让右侧详情跟着刷新
        if keep_kp_id is not None:
            for i in range(self.kp_table.rowCount()):
                it = self.kp_table.item(i, 0)
                if it is not None and it.data(Qt.UserRole) == int(keep_kp_id):
                    self.kp_table.selectRow(i)
                    break
            self._on_kp_selected()

    def _on_kp_selected(self) -> None:
        """点知识点 → 看它出现在哪些回看点上（这是"漏洞"和"知识点"之间的那条边）。"""
        row = self.kp_table.currentRow()
        if row < 0:
            return
        item = self.kp_table.item(row, 0)
        kp_id = item.data(Qt.UserRole) if item else None
        if kp_id is None:
            return
        kps = self.db.get_kp(int(kp_id))
        knots = self.db.kp_knots(int(kp_id))
        lines = [f"<h3 style='color:{theme.TEXT}'>{kps['name'] if kps else ''}</h3>"]
        if kps is not None:
            from ..cognitive import mastery as kpm
            st = kpm.status_of(float(kps["mastery"] or 0), int(kps["attempts"] or 0),
                               str(kps["kp_type"] or "concept"))
            lines.append(
                f"<p>掌握度 <b>{int(float(kps['mastery'] or 0) * 100)}%</b>"
                f"　状态 <b>{STATUS_CN[st]}</b>"
                f"　证据 {int(kps['attempts'] or 0)} 次"
                f"　门槛 {int(float(kps['gate'] or DEFAULT_GATE) * 100)}%</p>")
        if not knots:
            lines.append(f"<p style='color:{theme.TEXT_3}'>还没有回看点挂在这个知识点上。</p>")
        else:
            lines.append("<ul>")
            for k in knots:
                solved = self.engine.knot_solved(int(k["id"]))
                mark = "✅ 已补上" if solved else "⚠️ 未解决"
                # 标出节次：演示数据里同一讲可能有两份，不标会像重复行
                keys = k.keys() if hasattr(k, "keys") else []
                idx = k["session_idx"] if "session_idx" in keys else None
                where = f"第 {int(idx)} 节　" if idx else ""
                lines.append(
                    f"<li>{where}{_hms(float(k['t_start']))}　分数 "
                    f"{float(k['score'] or 0):.2f}　{mark}"
                    f"　<span style='color:{theme.TEXT_3}'>{(k['evidence'] or '')[:60]}</span></li>")
            lines.append("</ul>")
        self.map_view.setHtml("".join(lines))

    def _on_kp_mark(self, ok: bool) -> None:
        row = self.kp_table.currentRow()
        if row < 0:
            self.status.showMessage("先在表里点一个知识点", 4000)
            return
        kp_id = self.kp_table.item(row, 0).data(Qt.UserRole)
        if kp_id is None:
            return
        res = self.engine.record_kp_attempt(int(kp_id), 1.0 if ok else 0.0,
                                           source="confirm_useful" if ok else "self_test")
        # 传 kp_id 进去：重画后选中行还停在这个知识点上，详情才不会是旧的
        self.refresh_kp_table(keep_kp_id=int(kp_id))
        self._load_session_view()
        self.status.showMessage(
            f"{res['name']}：掌握度 {int(res['mastery'] * 100)}%"
            f"（{STATUS_CN[res['status']]}）", 6000)

    # ------------------------------------------------------------------
    # 问模型（对话 + 知识库）
    # ------------------------------------------------------------------
    def _build_chat_tab(self) -> QWidget:
        # 这一页是左右两栏，所以不用 _page() 的竖向版心，单独建（但边距/间距同一套数值）
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(theme.GAP_L, theme.GAP_M, theme.GAP_L, theme.GAP_L)
        lay.setSpacing(theme.GAP_M)

        # 左：知识库。整块装进一张卡片，列表自己不画框（一张卡只该有一个边）
        left_card, lv = _card(pad=theme.GAP_M, spacing=theme.GAP_S, shadow=True)
        lv.addWidget(_section("知识库"))
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_XS)
        self.btn_kb_import = QPushButton("导入…")
        self.btn_kb_import.clicked.connect(self._on_kb_import)
        self.btn_kb_del = QPushButton("删除")
        self.btn_kb_del.setIcon(icons.get("close"))
        self.btn_kb_del.clicked.connect(self._on_kb_delete)
        self.btn_kb_rebuild = QPushButton("重建索引")
        self.btn_kb_rebuild.setToolTip("按当前切块参数重切所有资料；"
                                       "读不回原文件的那份会明确报出来，不会静默跳过")
        self.btn_kb_rebuild.clicked.connect(self._on_kb_rebuild)
        for b in (self.btn_kb_import, self.btn_kb_del, self.btn_kb_rebuild):
            row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)
        self.lbl_kb_hint = QLabel("还没有导入资料。支持 txt/md/srt/vtt/docx/pptx/pdf；"
                                  "导入后引用会带页码或时间码，能回到原文。")
        self.lbl_kb_hint.setWordWrap(True)
        self.lbl_kb_hint.setObjectName("muted")
        lv.addWidget(self.lbl_kb_hint)
        self.kb_list = QListWidget()
        self.kb_list.setObjectName("plain")
        self.kb_list.setFrameShape(QFrame.NoFrame)
        self.kb_list.setMinimumHeight(88)
        self.kb_list.setMaximumHeight(170)
        self.kb_list.setToolTip("双击看它被切成了几段；'全局'的文档对本门课所有节可见")
        lv.addWidget(self.kb_list)

        line = QFrame()
        line.setObjectName("divider")
        line.setFixedHeight(1)
        lv.addWidget(line)

        lv.addWidget(_section("检索测试"))
        self.ed_kb_query = QLineEdit()
        self.ed_kb_query.setPlaceholderText("输入术语，看命中哪些片段")
        self.ed_kb_query.returnPressed.connect(self._on_kb_search)
        lv.addWidget(self.ed_kb_query)
        self.kb_hits = QListWidget()
        self.kb_hits.setObjectName("plain")
        self.kb_hits.setFrameShape(QFrame.NoFrame)
        self.kb_hits.setMinimumHeight(80)
        self.kb_hits.setMaximumHeight(150)
        self.kb_hits.setVisible(False)       # 没检索过就不占位置
        lv.addWidget(self.kb_hits)
        lv.addStretch(1)
        left_card.setFixedWidth(340)
        # 顶对齐 + 不拉伸：知识库就那么点内容，卡片拉满整列会变成一张大白纸
        lay.addWidget(left_card, 0, Qt.AlignTop)

        # 右：对话
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setHtml(_chat_html([], None))
        rv.addWidget(self.chat_view, 1)
        self.ed_ask = QPlainTextEdit()
        self.ed_ask.setPlaceholderText("问点什么…（Enter 发送，Shift+Enter 换行）")
        self.ed_ask.setMaximumHeight(76)
        self.ed_ask.installEventFilter(self)
        rv.addWidget(self.ed_ask)
        row2 = QHBoxLayout()
        self.btn_ask = QPushButton("发送")
        self.btn_ask.setObjectName("primary")
        self.btn_ask.setIcon(icons.get("send", theme.ON_ACCENT, 16))
        self.btn_ask.clicked.connect(self._on_ask)
        self.btn_clear_chat = QPushButton("清空对话")
        self.btn_clear_chat.setIcon(icons.get("close"))
        self.btn_clear_chat.clicked.connect(self._on_clear_chat)
        self.chk_use_kb = QCheckBox("用知识库")
        self.chk_use_kb.setChecked(True)
        self.chk_use_ctx = QCheckBox("带上本节的回看点")
        self.chk_use_ctx.setChecked(True)
        row2.addWidget(self.btn_ask)
        row2.addWidget(self.btn_clear_chat)
        row2.addSpacing(10)
        row2.addWidget(self.chk_use_kb)
        row2.addWidget(self.chk_use_ctx)
        self.btn_chat_win = QPushButton("弹成小窗")
        self.btn_chat_win.setIcon(icons.get("chat"))
        self.btn_chat_win.setToolTip("开一个置顶小窗，边看网课边问；\n"
                                     "可以拖图片/PDF 进去，也可以截图后 Ctrl+V 粘贴")
        self.btn_chat_win.clicked.connect(self._open_chat_window)
        row2.addWidget(self.btn_chat_win)
        row2.addStretch(1)
        self.lbl_chat_state = QLabel("")
        self.lbl_chat_state.setObjectName("hint")
        row2.addWidget(self.lbl_chat_state)
        rv.addLayout(row2)
        lay.addWidget(right, 1)
        return w

    def _refresh_kb(self) -> None:
        course_id = self._current_course_id()
        self.kb_list.clear()
        try:
            docs = self.engine.kb_docs(course_id or None)
        except Exception:  # noqa: BLE001
            docs = []
        for d in docs:
            scope = "本课" if d["course_id"] else "全局"
            it = QListWidgetItem(
                f"{d['title']}  ·  {scope}  ·  {d['chars']} 字 / {d['chunks']} 段")
            it.setData(Qt.UserRole, d["id"])
            it.setToolTip(f"{d['note']}\n{d['path']}")
            self.kb_list.addItem(it)
        if docs:
            total = sum(int(d["chunks"] or 0) for d in docs)
            msg = f"共 {len(docs)} 个资料、{total} 段可检索；提问时会先查这里再回答。"
            try:
                idx = self.engine.kb_index_status()
                if idx["stale"]:
                    msg += ("　⚠️ 索引与当前切块参数不一致（签名 "
                            f"{idx['stored'][:6]}→{idx['sig'][:6]}），点「重建索引」")
                    self.btn_kb_rebuild.setStyleSheet(f"color:{theme.WARN};")
                else:
                    self.btn_kb_rebuild.setStyleSheet("")
            except Exception:  # noqa: BLE001
                pass
            self.lbl_kb_hint.setText(msg)
        else:
            self.lbl_kb_hint.setText("未导入资料。支持 txt/md/srt/vtt/docx/pptx/pdf；"
                                     "导入后引用会带页码或时间码，能回到原文。")
        # 空的列表还占着 170px 高，会在这张卡里留一块"没内容的洞"：
        # 没资料就把它藏起来，让说明文字和下面的检索框连在一起。
        self.kb_list.setVisible(bool(docs))

    def _current_course_id(self) -> str:
        """当前课程：优先选中会话，其次最近一节，最后最近一门课。

        没有回退的话，软件刚打开（还没点任何会话）时知识点表会是空的，
        看起来像"数据没了"——实际上只是没选课。
        """
        sid = self.current_session
        if not sid and hasattr(self, "session_list"):
            # QListWidget 没有 currentData()（那是 QComboBox 的），要从 currentItem 取
            item = self.session_list.currentItem()
            sid = item.data(Qt.UserRole) if item is not None else None
        if sid:
            row = self.db.get_session(str(sid))
            if row is not None:
                return str(row["course_id"])
        for r in self.db.list_sessions() or []:
            return str(r["course_id"])
        courses = self.db.list_courses() or []
        return str(courses[0]["id"]) if courses else ""

    # -- 从 B 站导入 / 看这一节 -------------------------------------------
    def _on_bili_import(self) -> None:
        from .bili_dialog import BiliImportDialog

        dlg = BiliImportDialog(self)
        if dlg.exec() != QDialog.Accepted or not dlg.chosen:
            return
        res = self.engine.import_series(dlg.course_title, dlg.chosen, source="bili")
        self._refresh_sessions()
        msg = f"已导入 {res['created']} 节到「{dlg.course_title}」"
        if res["skipped"]:
            msg += f"（{res['skipped']} 节重复，跳过）"
        msg += "；点一节右键「看这一节」就能开录"
        self.status.showMessage(msg, 12000)

    def _open_session_video(self, sid: str, start_s: float | None = None) -> None:
        """在浏览器打开这一节（可以带秒数），并顺手把自动录打开。

        为什么不自己内嵌播放：视频在 B 站，版权在 B 站，播放器也在 B 站。
        我们能做的是"准时开始听"，而不是再造一个播放器。
        """
        row = self.db.get_session(sid) if sid else None
        url = str(row["video_url"] or "") if row else ""
        if not url:
            self.status.showMessage("这一节没有原始链接（手动录的节没有）", 8000)
            return
        if start_s is not None and start_s > 0:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}t={int(start_s)}"
        if not bili.open_in_browser(url):
            self.status.showMessage("打不开浏览器，链接已复制到剪贴板", 9000)
            QApplication.clipboard().setText(url)
            return
        if self.chk_autoweb.isChecked() and not self.engine.recording:
            self._start_web_watcher()
        self.status.showMessage("已在浏览器打开；开始播放就会自动录这一节", 10000)

    def _on_kb_import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入知识库资料（可多选）", "",
            "资料 (*.txt *.md *.markdown *.srt *.vtt *.docx *.pptx *.pdf);;所有文件 (*.*)")
        if not paths:
            return
        course_id = self._current_course_id()

        def job():
            out = []
            for pth in paths:
                try:
                    out.append(self.engine.kb_import(pth, course_id=course_id))
                except Exception as e:  # noqa: BLE001
                    out.append({"ok": False, "title": Path(pth).name,
                                "reason": str(e)})
            return out

        def done(results):
            ok = [r for r in results if r.get("ok")]
            bad = [r for r in results if not r.get("ok")]
            self._refresh_kb()
            msg = f"导入 {len(ok)} 个资料，共 {sum(r['chunks'] for r in ok)} 段" if ok else "没有导入成功"
            if bad:
                msg += "；失败：" + "；".join(f"{r['title']}（{r['reason']}）" for r in bad)
            self.lbl_kb_hint.setText(msg)
            self.status.showMessage(msg, 9000)

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: QMessageBox.critical(self, "导入失败", e))
        self._spawn(w)

    def _on_kb_delete(self) -> None:
        it = self.kb_list.currentItem()
        if it is None:
            return
        doc_id = it.data(Qt.UserRole)
        if QMessageBox.question(self, "删除资料", f"删除「{it.text()}」？") != QMessageBox.Yes:
            return
        self.db.delete_kb_doc(int(doc_id))
        self._refresh_kb()
        self.status.showMessage("已删除", 4000)

    def _on_kb_rebuild(self) -> None:
        self.status.showMessage("正在按当前参数重切所有资料…")
        w = Worker(self.engine.rebuild_kb_index)
        w.done.connect(self._kb_rebuild_done)
        w.failed.connect(lambda e: self.status.showMessage(f"重建失败：{e}", 12000))
        self._spawn(w)

    def _kb_rebuild_done(self, res: dict) -> None:
        self._refresh_kb()
        msg = f"重建完成：{res.get('rebuilt', 0)} 份资料、{res.get('chunks', 0)} 段可检索"
        if res.get("missing"):
            msg += f"；{len(res['missing'])} 份原文件已找不到（{('、'.join(res['missing'][:3]))}）"
        if res.get("failed"):
            msg += f"；{len(res['failed'])} 份切不出内容"
        self.status.showMessage(msg, 15000)

    def _on_kb_search(self) -> None:
        q = self.ed_kb_query.text().strip()
        self.kb_hits.clear()
        if not q:
            self.kb_hits.setVisible(False)
            return
        self.kb_hits.setVisible(True)
        hits = self.engine.kb_search(q, course_id=self._current_course_id(), k=6)
        if not hits:
            self.kb_hits.addItem("没有命中（知识库为空，或这个词在资料里没出现）")
            return
        for h in hits:
            it = QListWidgetItem(f"{h['score']:.2f}  《{h['doc']}》第 {h['ord']} 段 | "
                                 f"{h['snippet'][:70]}")
            it.setToolTip(h["text"][:900])
            self.kb_hits.addItem(it)

    def _append_chat(self, role: str, text: str, sources: list | None = None) -> None:
        self._chat_msgs.append({"role": role, "content": text,
                                "sources": sources or []})
        self.chat_view.setHtml(_chat_html(self._chat_msgs, self.settings.llm.model))
        self.chat_view.moveCursor(QTextCursor.End)

    def _on_clear_chat(self) -> None:
        self._chat_msgs = []
        if getattr(self, "_chat_win", None) is not None:
            self._chat_win.msgs = self._chat_msgs      # 小窗跟着换到新的列表对象
        if self.current_session:
            self.engine.clear_chat(self.current_session)
        self.chat_view.setHtml(_chat_html([], None))

    # -- 边看边问的小窗 ---------------------------------------------------
    def _open_chat_window(self) -> None:
        """开/前置「问模型」小窗。

        小窗和这一页共用同一个 msgs 列表与 session_id：窗口里问过的，回来还能看到；
        关掉窗口时把这一页重画一遍，避免"窗里说过的话主页看不到"。
        """
        win = getattr(self, "_chat_win", None)
        if win is not None and win.isVisible():
            win.raise_()
            win.activateWindow()
            return
        from .chat_window import ChatWindow

        self._chat_win = ChatWindow(
            self.engine, self.settings, self._chat_msgs,
            session_id=self.current_session or "",
            on_close=self._on_chat_window_closed, parent=self)
        self._chat_win.show()
        self._chat_win.raise_()
        self._chat_win.activateWindow()

    def _on_chat_window_closed(self) -> None:
        # 窗口里可能清空/新增过消息：回到主页时重新画一遍
        self.chat_view.setHtml(_chat_html(self._chat_msgs, self.settings.llm.model))
        try:
            self._refresh_kb()
        except Exception:                             # noqa: BLE001
            pass

    def _on_ask(self) -> None:
        q = self.ed_ask.toPlainText().strip()
        if not q:
            return
        self.ed_ask.clear()
        self._append_chat("user", q)
        sid = self.current_session or ""
        self.btn_ask.setEnabled(False)
        self.lbl_chat_state.setText("思考中…")
        # 控件只能在主线程读，先取值再进后台线程（跨线程碰 Qt 控件会炸）
        use_kb = self.chk_use_kb.isChecked()
        use_ctx = self.chk_use_ctx.isChecked()

        def job():
            return self.engine.ask(q, session_id=sid,
                                   use_kb=use_kb, use_context=use_ctx)

        def done(res):
            self.btn_ask.setEnabled(True)
            if res.get("error"):
                self.lbl_chat_state.setText("失败")
                self._append_chat("assistant",
                                  f"（调用失败：{res['error']}）\n\n"
                                  f"检查设置页的 base_url / api_key / 模型名。")
                return
            self.lbl_chat_state.setText("")
            self._append_chat("assistant", res["answer"], res.get("sources"))

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: (self.btn_ask.setEnabled(True),
                                    self.lbl_chat_state.setText("失败"),
                                    QMessageBox.critical(self, "对话失败", e)))
        self._spawn(w)

    # ------------------------------------------------------------------
    # 自动抓屏（算法决定时机）
    # ------------------------------------------------------------------
    def _start_autocap(self) -> None:
        from dsh_app.autocap import AutoCapConfig

        c = getattr(self.settings, "autocap", None) or AutoCapConfig()
        self._autocap_cfg = c
        self._autocap_prev: bytes | None = None
        self._autocap_last_t = 0.0
        self._autocap_log: list[float] = []
        self._autocap_stats = {"taken": 0, "skip": 0}
        self._autocap_busy = False
        if not hasattr(self, "_autocap_timer"):
            self._autocap_timer = QTimer(self)
            self._autocap_timer.timeout.connect(self._autocap_tick)
        if c.enabled:
            self._autocap_timer.start(int(max(2.0, c.interval_s) * 1000))
            if hasattr(self, "lbl_autocap"):
                self.lbl_autocap.setText(
                    f"已启用：录制中每 {c.interval_s:.0f} 秒检查一次，"
                    f"画面变化 ≥{c.change_threshold:.0%} 且间隔 ≥{c.min_gap_s:.0f} 秒才抓")
        else:
            self._autocap_timer.stop()
            if hasattr(self, "lbl_autocap"):
                self.lbl_autocap.setText("未启用")

    def _autocap_tick(self) -> None:
        from dsh_app import autocap as ac

        cfg = getattr(self, "_autocap_cfg", None)
        if cfg is None or not cfg.enabled or self._autocap_busy:
            return
        if not (self.engine.recording and self.engine.recorder):
            return
        try:
            from dsh_app.screen import grab_primary_screen
            from PySide6.QtGui import QImage

            png = grab_primary_screen()
            img = QImage.fromData(png).scaled(64, 36, Qt.IgnoreAspectRatio,
                                              Qt.SmoothTransformation)
            img = img.convertToFormat(QImage.Format_Grayscale8)
            cur = bytes(img.constBits())
        except Exception:  # noqa: BLE001
            return
        now = time.time()
        ratio = ac.change_ratio(self._autocap_prev or b"", cur)
        since = now - self._autocap_last_t if self._autocap_last_t else 1e9
        hour_ago = now - 3600
        self._autocap_log = [t for t in self._autocap_log if t >= hour_ago]
        d = ac.decide(ratio, since, len(self._autocap_log), cfg)
        if not d.take:
            self._autocap_stats["skip"] += 1
            self.lbl_autocap.setText(
                f"自动抓屏：已抓 {self._autocap_stats['taken']} 帧，"
                f"跳过 {self._autocap_stats['skip']} 次（{d.reason}）")
            return
        self._autocap_prev = cur
        self._autocap_last_t = now
        self._autocap_log.append(now)
        self._autocap_busy = True
        t_note = float(self.engine.recorder.duration)
        sid = self.current_session

        def job(frames=(png,)):
            from dsh_app.screen import build_extractor

            ext = build_extractor(self.settings)
            ok, why = ext.available()
            if not ok:
                return {"ok": False, "reason": why}
            res = ext.extract(frames[0])
            if res.error:
                return {"ok": False, "reason": res.error}
            if not (res.text or "").strip():
                return {"ok": False, "reason": "画面里没有课程文字"}
            info = self.engine.capture_screen_note(sid, text=res.text,
                                                   kind=res.kind, t=t_note,
                                                   source="auto")
            return {"ok": True, "chars": info.get("chars", 0), "t": t_note}

        def done(r):
            self._autocap_busy = False
            if r.get("ok"):
                self._autocap_stats["taken"] += 1
                self.lbl_autocap.setText(
                    f"自动抓屏：已抓 {self._autocap_stats['taken']} 帧"
                    f"（最近 {_mmss(r['t'])}，{r['chars']} 字）")
                self.status.showMessage(
                    f"自动抓屏：{_mmss(r['t'])} 的屏幕内容已入库（{r['chars']} 字）", 5000)
            else:
                self.lbl_autocap.setText(
                    f"自动抓屏：跳过（{r.get('reason', '')}）；已抓 "
                    f"{self._autocap_stats['taken']} 帧")

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: setattr(self, "_autocap_busy", False))
        self._spawn(w)

    def _build_settings_tab(self) -> QWidget:
        w, v = _page()

        # 录音设备（系统回环）
        g0 = QGroupBox("录音设备（系统回环，采的是系统输出，不碰浏览器）")
        h0 = QHBoxLayout(g0)
        self.cmb_capture = QComboBox()
        self.cmb_capture.setMinimumWidth(460)
        self._fill_capture_devices()
        h0.addWidget(QLabel("设备"))
        h0.addWidget(self.cmb_capture, 1)
        self.btn_capture_probe = QPushButton("试读 1 秒")
        self.btn_capture_probe.clicked.connect(self._on_probe_capture)
        h0.addWidget(self.btn_capture_probe)
        v.addWidget(g0)
        self.lbl_capture_hint = QLabel(
            "提示：蓝牙耳机在没有声音播放时不会出数据（试读会失败），开始播放课程后即可正常录音。")
        self.lbl_capture_hint.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px; padding:2px 4px;")
        v.addWidget(self.lbl_capture_hint)

        # 自动抓屏（算法决定时机，不用快捷键）
        # 链接导入（B站字幕需要登录凭据）
        g_src = QGroupBox("链接导入（B站字幕需登录凭据，只存本机配置）")
        h_src = QHBoxLayout(g_src)
        self.ed_sessdata = QLineEdit(getattr(getattr(self.settings, "source", None),
                                             "bili_sessdata", ""))
        self.ed_sessdata.setEchoMode(QLineEdit.Password)
        self.ed_sessdata.setPlaceholderText("可选：浏览器登录 B站后 Cookie 里的 SESSDATA 值")
        self.ed_sessdata.setMinimumWidth(420)
        h_src.addWidget(QLabel("SESSDATA"))
        h_src.addWidget(self.ed_sessdata, 1)
        v.addWidget(g_src)

        g_ac = QGroupBox("屏幕内容（自动）")
        hac = QHBoxLayout(g_ac)
        self.chk_autocap = QCheckBox("录制中自动抓取屏幕并识别")
        self.chk_autocap.setChecked(bool(getattr(getattr(self.settings, "autocap", None),
                                               "enabled", False)))
        self.chk_autocap.setToolTip(
            "由算法决定时机：画面变化 ≥12% 且距上次 ≥30 秒才抓，每小时最多 40 帧；\n"
            "画面里没有课程文字的结果会被丢弃。")
        self.lbl_autocap = QLabel("未启用")
        self.lbl_autocap.setObjectName("hint")
        hac.addWidget(self.chk_autocap)
        hac.addSpacing(12)
        hac.addWidget(self.lbl_autocap, 1)
        v.addWidget(g_ac)

        # 行为检测（不看屏幕）
        g_bx = QGroupBox("听课行为（不看屏幕，只听声音）")
        hbx = QHBoxLayout(g_bx)
        self.chk_pause = QCheckBox("检测回退 / 暂停（判断你是不是卡住了）")
        self.chk_pause.setChecked(bool(getattr(self.settings.capture, "detect_pauses", True)))
        self.chk_pause.setToolTip(
            "回退：同一段声音又出现一次（指纹余弦 ≥0.985，连续 2 秒）；\n"
            "暂停：播放器停住时环回设备仍在出数据但全是 0（数字静音 ≥3 秒）。\n"
            "两项都不需要录屏、不需要浏览器插件。")
        self.sp_pause = QDoubleSpinBox()
        self.sp_pause.setRange(1.0, 30.0)
        self.sp_pause.setSingleStep(0.5)
        self.sp_pause.setDecimals(1)
        self.sp_pause.setSuffix(" 秒")
        self.sp_pause.setValue(float(getattr(self.settings.capture, "pause_min_s", 3.0)))
        self.sp_pause.setToolTip("静音超过这个长度才算暂停（短于此当换气忽略）")
        hbx.addWidget(self.chk_pause)
        hbx.addSpacing(10)
        hbx.addWidget(QLabel("最短静音"))
        hbx.addWidget(self.sp_pause)
        hbx.addStretch(1)
        v.addWidget(g_bx)

        # 生成层
        g1 = QGroupBox("生成层（重讲与诊断）—— 本地/云端一键切换，协议完全兼容")
        f1 = QFormLayout(g1)
        f1.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        f1.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        f1.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        f1.setHorizontalSpacing(14)
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItem("云端 API", "cloud")
        self.cmb_backend.addItem("本地模型", "local")
        self.cmb_backend.addItem("关闭生成", "off")
        f1.addRow("后端", self.cmb_backend)
        # 服务商：选它就等于填好地址 + 给出候选模型，用户只需要贴 key
        self.cmb_preset = QComboBox()
        for p in CLOUD_PRESETS:
            self.cmb_preset.addItem(p["label"], p["key"])
        f1.addRow("服务商", self.cmb_preset)
        self.ed_base = QLineEdit(self.settings.llm.base_url)
        self.ed_base.setPlaceholderText("选服务商后自动填好；自定义时才需要改")
        f1.addRow("接入地址", self.ed_base)
        # key：右边给一个"去哪拿"的入口，省得用户去搜
        self.ed_key = QLineEdit(self.settings.llm.api_key)
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_key.setPlaceholderText("粘贴 API Key（只存本机）")
        self.btn_key_site = QPushButton("去哪拿 Key？")
        self.btn_key_site.setFixedWidth(120)
        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self.ed_key, 1)
        key_row.addWidget(self.btn_key_site)
        f1.addRow("API Key", self._wrap(key_row))
        # 模型：下拉（列表由「刷新列表」从端点拉真实可用项），也可手填
        self.cmb_llm_model = QComboBox()
        self.cmb_llm_model.setEditable(True)
        self.cmb_llm_model.setMinimumWidth(320)
        self.btn_models = QPushButton("刷新列表")
        self.btn_models.setFixedWidth(96)
        m_row = QHBoxLayout()
        m_row.setContentsMargins(0, 0, 0, 0)
        m_row.addWidget(self.cmb_llm_model, 1)
        m_row.addWidget(self.btn_models)
        f1.addRow("模型", self._wrap(m_row))
        # 视觉：默认跟主模型一致，需要单独指定时再勾掉
        self.cmb_vision = QComboBox()
        self.cmb_vision.setEditable(True)
        self.cmb_vision.setMinimumWidth(320)
        self.chk_vision_same = QCheckBox("屏幕识别也用这个模型")
        v_row = QHBoxLayout()
        v_row.setContentsMargins(0, 0, 0, 0)
        v_row.addWidget(self.cmb_vision, 1)
        v_row.addWidget(self.chk_vision_same)
        f1.addRow("视觉模型", self._wrap(v_row))
        self.lbl_preset_hint = QLabel("")
        self.lbl_preset_hint.setWordWrap(True)
        self.lbl_llm_status = QLabel("")
        self.lbl_llm_status.setWordWrap(True)
        self.lbl_llm_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f1.addRow("", self.lbl_llm_status)
        self.cmb_asr = QComboBox()
        for label, key in (("自动选择", "auto"), ("OpenAI 兼容转写服务", "cloud"),
                           ("本地 faster-whisper", "local_whisper"),
                           ("同名侧车字幕", "sidecar"), ("离线演示（非真实转写）", "offline")):
            self.cmb_asr.addItem(label, key)
        f1.addRow("转写后端", self.cmb_asr)
        self.ed_asr_model = QLineEdit(self.settings.asr.realtime_model)
        f1.addRow("转写模型", self.ed_asr_model)
        row = QHBoxLayout()
        self.btn_save_cfg = QPushButton("保存设置")
        self.btn_probe = QPushButton("探测能力")
        self.chk_demo_asr = QCheckBox("允许演示转写（合成文本，仅用于看流程）")
        self.chk_demo_asr.setChecked(bool(getattr(self.settings.asr, "allow_demo", False)))
        f1.addRow("", self.chk_demo_asr)
        self.chk_preload = QCheckBox("启动时预热模型（课上不冷启动）")
        self.chk_preload.setChecked(self.settings.llm.preload_on_start)
        row.addWidget(self.btn_save_cfg)
        row.addWidget(self.btn_probe)
        row.addWidget(self.chk_preload)
        row.addStretch(1)
        f1.addRow("", self._wrap(row))
        v.addWidget(g1)

        # 本地转写（Whisper）：依赖 / 模型 / 实测，三件事分开做
        gw = QGroupBox("本地转写（Whisper）：离线转写，免费不限量")
        vw = QVBoxLayout(gw)
        hw = QHBoxLayout()
        self.cmb_whisper = QComboBox()
        for it in WHISPER_SIZES:
            self.cmb_whisper.addItem(it["label"], it["key"])
        self.cmb_whisper.setMinimumWidth(300)
        # 默认选中配置里那档（或本机已装的最好那档），而不是列表第一个 tiny
        want = str(getattr(getattr(self.settings, "asr", None), "local_size", "") or "")
        idx = self.cmb_whisper.findData(want) if want else -1
        if idx < 0:
            try:
                installed = [m["key"] for m in whisper_status()["models"] if m["installed"]]
            except Exception:  # noqa: BLE001
                installed = []
            if installed:
                idx = self.cmb_whisper.findData(installed[-1])
        if idx >= 0:
            self.cmb_whisper.setCurrentIndex(idx)
        self.btn_w_refresh = QPushButton("检测状态")
        self.btn_w_deps = QPushButton("安装依赖")
        self.btn_w_dl = QPushButton("下载模型")
        self.btn_w_test = QPushButton("测试转写")
        for b in (self.cmb_whisper, self.btn_w_refresh, self.btn_w_deps,
                  self.btn_w_dl, self.btn_w_test):
            hw.addWidget(b)
        hw.addStretch(1)
        vw.addLayout(hw)
        self.lbl_whisper = QLabel("点「检测状态」看本机情况")
        self.lbl_whisper.setWordWrap(True)
        self.lbl_whisper.setMinimumHeight(84)      # 状态是多行，别被压成一行
        self.lbl_whisper.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vw.addWidget(self.lbl_whisper)
        v.addWidget(gw)

        # 回看点识别灵敏度：阈值就是"什么算卡住"，不能写死在代码里
        g_sens = QGroupBox("回看点识别（改完对下一节分析生效；已分析过的要看新阈值请重新分析）")
        hs = QHBoxLayout(g_sens)
        self.cmb_sens = QComboBox()
        for label, gm, pct in (("保守：只标特别明显的", 0.55, 75.0),
                               ("标准（推荐）", 0.45, 70.0),
                               ("积极：宁可多标，不要漏掉", 0.35, 65.0)):
            self.cmb_sens.addItem(label, (gm, pct))
        self.cmb_sens.setMinimumWidth(210)
        hs.addWidget(QLabel("灵敏度"))
        hs.addWidget(self.cmb_sens)
        hs.addSpacing(theme.GAP_M)
        hs.addWidget(QLabel("判定分"))
        self.sp_grade = QDoubleSpinBox()
        self.sp_grade.setRange(0.20, 0.90)
        self.sp_grade.setSingleStep(0.05)
        self.sp_grade.setDecimals(2)
        self.sp_grade.setToolTip("融合分高于它才算回看点。调低=标得更多")
        hs.addWidget(self.sp_grade)
        hs.addWidget(QLabel("象限分位"))
        self.sp_pct = QDoubleSpinBox()
        self.sp_pct.setRange(40.0, 95.0)
        self.sp_pct.setSingleStep(5.0)
        self.sp_pct.setDecimals(0)
        self.sp_pct.setToolTip("负荷高/低按本节内分位数切，不是固定 0.5")
        hs.addWidget(self.sp_pct)
        hs.addStretch(1)
        self.btn_reanalyze_now = QPushButton("重新分析本节")
        self.btn_reanalyze_now.setIcon(icons.get("refresh"))
        hs.addWidget(self.btn_reanalyze_now)
        v.addWidget(g_sens)
        self._load_sensitivity()
        self.cmb_sens.currentIndexChanged.connect(self._on_sens_preset)
        self.btn_reanalyze_now.clicked.connect(self._on_reanalyze_current)

        # 本地部署
        g2 = QGroupBox("本地模型：检测 + 一键部署（方案 4.4 清单；说话人分离已按裁决移除）")
        v2 = QVBoxLayout(g2)
        hb = QHBoxLayout()
        self.btn_detect = QPushButton("检测本机模型与算力")
        self.cmb_model = QComboBox()
        self.cmb_model.setMinimumWidth(320)
        self.btn_gen_script = QPushButton("生成部署脚本")
        self.btn_launch = QPushButton("执行部署并启动")
        for b in (self.btn_detect, self.cmb_model, self.btn_gen_script, self.btn_launch):
            hb.addWidget(b)
        hb.addStretch(1)
        v2.addLayout(hb)
        self.deploy_view = QTextBrowser()
        # 设置页现在整体可滚动，所以这块不必再"撑满剩余高度"：
        # 一撑就把下面的存储/数据挤出屏幕，用户反而看不到新加的东西。
        self.deploy_view.setMinimumHeight(180)
        v2.addWidget(self.deploy_view)
        v.addWidget(g2)

        # 存储：音频是本软件最占空间的东西，必须能看见、能收、能清
        g_st = QGroupBox("存储（系统回环录音很占地方：一节 90 分钟的 WAV 近 0.9 GB）")
        vst = QVBoxLayout(g_st)
        r1 = QHBoxLayout()
        self.lbl_storage = QLabel("—")
        self.lbl_storage.setObjectName("muted")
        self.lbl_storage.setWordWrap(True)
        r1.addWidget(self.lbl_storage, 1)
        self.btn_audio_dir = QPushButton("打开音频目录")
        self.btn_audio_dir.setIcon(icons.get("search"))
        r1.addWidget(self.btn_audio_dir)
        vst.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("保留天数"))
        self.sp_keep_days = QDoubleSpinBox()
        self.sp_keep_days.setRange(0, 365)
        self.sp_keep_days.setDecimals(0)
        self.sp_keep_days.setSuffix(" 天")
        self.sp_keep_days.setToolTip("0 = 永不自动清理。音频删了课还在，只是不能再重新转写这一节")
        r2.addWidget(self.sp_keep_days)
        self.chk_compress = QCheckBox("录完压成 FLAC（16k 单声道，约省 9 倍空间）")
        self.chk_compress.setToolTip("转写只用 16k 单声道；压缩后照样能重新分析、重新转写")
        r2.addWidget(self.chk_compress)
        self.chk_keep_raw = QCheckBox("保留原始 WAV")
        self.chk_keep_raw.setToolTip("勾上就不压缩。占空间，但导出给别的工具最省事")
        r2.addWidget(self.chk_keep_raw)
        r2.addStretch(1)
        self.btn_sweep = QPushButton("立即清理过期音频")
        self.btn_sweep.setIcon(icons.get("close"))
        r2.addWidget(self.btn_sweep)
        vst.addLayout(r2)
        v.addWidget(g_st)
        self.btn_audio_dir.clicked.connect(
            lambda: self._open_dir(AUDIO_DIR))
        self.btn_sweep.clicked.connect(self._on_sweep_audio)

        # 数据：几十小时的回看点、掌握度、复习排程都在一个 sqlite 文件里
        g_bk = QGroupBox("数据（学习记录只有一个库文件，别只信硬盘）")
        hbk = QHBoxLayout(g_bk)
        self.lbl_backup = QLabel("—")
        self.lbl_backup.setObjectName("muted")
        self.lbl_backup.setWordWrap(True)
        hbk.addWidget(self.lbl_backup, 1)
        self.btn_backup = QPushButton("立即备份")
        self.btn_backup.setIcon(icons.get("check"))
        self.btn_restore = QPushButton("从备份恢复…")
        self.btn_restore.setIcon(icons.get("refresh"))
        self.btn_backup_dir = QPushButton("打开备份目录")
        self.btn_backup_dir.setIcon(icons.get("search"))
        for b in (self.btn_backup, self.btn_restore, self.btn_backup_dir):
            hbk.addWidget(b)
        v.addWidget(g_bk)
        self.btn_backup.clicked.connect(self._on_backup_now)
        self.btn_restore.clicked.connect(self._on_restore_backup)
        self.btn_backup_dir.clicked.connect(lambda: self._open_dir(maintenance.backup_dir()))

        # 自动抓取的"体检"：丢掉过什么、要不要调阈值——让机器自己攒证据
        g_dg = QGroupBox("自动抓取体检（它会自己攒样本，不用你记）")
        vdg = QVBoxLayout(g_dg)
        self.lbl_discard = QLabel("—")
        self.lbl_discard.setObjectName("muted")
        self.lbl_discard.setWordWrap(True)
        vdg.addWidget(self.lbl_discard)
        rdg = QHBoxLayout()
        self.btn_discard_apply = QPushButton("按记录调阈值")
        self.btn_discard_apply.setIcon(icons.get("check"))
        self.btn_discard_apply.setToolTip("把「最短保留时长」调到最近被丢掉的那些抓取之上，"
                                          "少录几次提示音")
        self.btn_discard_clear = QPushButton("清空记录")
        self.btn_discard_clear.setIcon(icons.get("close"))
        rdg.addWidget(self.btn_discard_apply)
        rdg.addWidget(self.btn_discard_clear)
        rdg.addStretch(1)
        vdg.addLayout(rdg)
        v.addWidget(g_dg)
        self.btn_discard_apply.clicked.connect(self._on_apply_discard_hint)
        self.btn_discard_clear.clicked.connect(self._on_clear_discards)

        # 知识库的"另一半"：不只有你上传的讲义，还有每节看过的课
        g_kb = QGroupBox("知识库：看过的课自动归档")
        hkb = QHBoxLayout(g_kb)
        self.lbl_kb_auto = QLabel("—")
        self.lbl_kb_auto.setObjectName("muted")
        self.lbl_kb_auto.setWordWrap(True)
        hkb.addWidget(self.lbl_kb_auto, 1)
        self.btn_kb_archive_all = QPushButton("把看过的课全部归档")
        self.btn_kb_archive_all.setIcon(icons.get("upload"))
        self.btn_kb_archive_all.setToolTip("每节转写完会自动归档一次；这个按钮补历史")
        hkb.addWidget(self.btn_kb_archive_all)
        v.addWidget(g_kb)
        self.btn_kb_archive_all.clicked.connect(self._on_archive_all_kb)

        self._refresh_storage_info()
        self._refresh_discard_info()
        self._refresh_kb_auto_info()

        self.btn_w_refresh.clicked.connect(self._refresh_whisper)
        self.btn_w_deps.clicked.connect(self._on_install_whisper_deps)
        self.btn_w_dl.clicked.connect(self._on_download_whisper)
        self.btn_w_test.clicked.connect(self._on_test_whisper)
        self.btn_save_cfg.clicked.connect(self._on_save_settings)
        self.btn_probe.clicked.connect(self._on_probe)
        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_gen_script.clicked.connect(self._on_gen_script)
        self.btn_launch.clicked.connect(self._on_launch)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset)
        self.btn_models.clicked.connect(self._on_fetch_models)
        self.btn_key_site.clicked.connect(self._on_open_key_site)
        self.chk_vision_same.toggled.connect(self._on_vision_same)
        self.ed_base.editingFinished.connect(self._on_base_edited)
        self._load_settings_to_widgets()
        QTimer.singleShot(1200, self._autofetch_models)
        return w

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _build_shortcuts(self) -> None:
        # 用户裁决：不要快捷键。所有动作都在界面上，抓屏由算法自己决定时机。
        pass

    # ------------------------------------------------------------------
    # 设置读写
    # ------------------------------------------------------------------
    def _load_settings_to_widgets(self) -> None:
        s = self.settings
        idx = self.cmb_backend.findData(s.llm.backend)
        self.cmb_backend.setCurrentIndex(max(0, idx))
        # 地址被手改过（例如换成公司网关 / MaaS 专用域名）就别再冒充某个服务商，
        # 否则界面显示"阿里云百炼"、实际连的是别的地址，用户会被误导。
        p = provider(s.llm.preset)
        key = p["key"]
        want_base = (p.get("base_url") or "").strip()
        got_base = (s.llm.base_url or "").strip()
        if got_base and want_base and got_base != want_base:
            key = "custom"
            p = provider("custom")
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.setCurrentIndex(max(0, self.cmb_preset.findData(key)))
        self.cmb_preset.blockSignals(False)
        self.ed_base.setText(s.llm.base_url)
        self.ed_key.setText(s.llm.api_key)
        # 下拉先放预设候选，保证"没联网也有东西可选"；真列表由刷新填
        self._fill_models(p.get("models") or [], s.llm.model)
        vision = getattr(s.llm, "vision_model", "") or ""
        same = (not vision) or (vision == s.llm.model)
        self.chk_vision_same.setChecked(same)
        self._fill_vision(p.get("vision_models") or [], vision or s.llm.model)
        self.cmb_vision.setEnabled(not same)
        self._show_preset_hint(p)
        idx = self.cmb_asr.findData(s.asr.backend)
        self.cmb_asr.setCurrentIndex(max(0, idx))
        self.ed_asr_model.setText(s.asr.realtime_model)

    def _fill_models(self, candidates: list[str], current: str) -> None:
        """填模型下拉：候选在前，当前值一定保留（哪怕它不在候选里）。"""
        self.cmb_llm_model.blockSignals(True)
        keep = self.cmb_llm_model.currentText().strip()
        self.cmb_llm_model.clear()
        seen = []
        for m in list(candidates) + ([current] if current else []):
            if m and m not in seen:
                seen.append(m)
        for m in seen:
            self.cmb_llm_model.addItem(m, m)
        want = current or keep
        if want:
            i = self.cmb_llm_model.findText(want)
            if i >= 0:
                self.cmb_llm_model.setCurrentIndex(i)
            else:
                self.cmb_llm_model.setEditText(want)
        self.cmb_llm_model.blockSignals(False)

    def _fill_vision(self, candidates: list[str], current: str) -> None:
        self.cmb_vision.blockSignals(True)
        self.cmb_vision.clear()
        seen = []
        for m in list(candidates) + ([current] if current else []):
            if m and m not in seen:
                seen.append(m)
        for m in seen:
            self.cmb_vision.addItem(m, m)
        if current:
            i = self.cmb_vision.findText(current)
            if i >= 0:
                self.cmb_vision.setCurrentIndex(i)
            else:
                self.cmb_vision.setEditText(current)
        self.cmb_vision.blockSignals(False)

    def _show_preset_hint(self, p: dict) -> None:
        bits = []
        if p.get("hint"):
            bits.append(p["hint"])
        if p.get("needs_key") is False:
            bits.append("这类端点不需要 API Key。")
        self.lbl_preset_hint.setText("　".join(bits))
        self.lbl_preset_hint.setStyleSheet(f"color:{theme.TEXT_3}; font-size:12px;")
        self.btn_key_site.setVisible(bool(p.get("key_url")))
        self.btn_key_site.setEnabled(bool(p.get("key_url")))
        self.btn_key_site.setToolTip(p.get("key_url") or "")
        # 本地端点不需要 key，输入框就别装作必填
        self.ed_key.setEnabled(p.get("needs_key") is not False)


    def _collect_settings(self) -> None:
        try:
            self.settings.capture.device_name = self.cmb_capture.currentData() or ""
        except Exception:
            pass
        try:
            self.settings.asr.allow_demo = bool(self.chk_demo_asr.isChecked())
        except Exception:
            pass
        s = self.settings
        s.llm.backend = self.cmb_backend.currentData()
        s.llm.preset = self.cmb_preset.currentData()
        s.llm.base_url = self.ed_base.text().strip()
        s.llm.api_key = self.ed_key.text().strip()
        s.llm.model = self.cmb_llm_model.currentText().strip()
        # 勾了"同一个模型"就同步过去，避免两处不一致
        if self.chk_vision_same.isChecked():
            s.llm.vision_model = s.llm.model
        else:
            s.llm.vision_model = self.cmb_vision.currentText().strip()
        if hasattr(self, "ed_sessdata") and getattr(s, "source", None) is not None:
            s.source.bili_sessdata = self.ed_sessdata.text().strip()
        if hasattr(self, "chk_autocap") and getattr(s, "autocap", None) is not None:
            s.autocap.enabled = self.chk_autocap.isChecked()
            s.capture.detect_pauses = self.chk_pause.isChecked()
            s.capture.pause_min_s = float(self.sp_pause.value())
        s.asr.backend = self.cmb_asr.currentData()
        s.asr.realtime_model = self.ed_asr_model.text().strip() or s.asr.realtime_model
        s.llm.preload_on_start = self.chk_preload.isChecked()
        # 存储策略（录完压不压、留多久）和回看点灵敏度：都是用户点出来就要生效的
        if hasattr(self, "sp_keep_days"):
            s.capture.keep_audio_days = int(self.sp_keep_days.value())
            s.capture.audio_compress = bool(self.chk_compress.isChecked())
            s.capture.save_raw_audio = bool(self.chk_keep_raw.isChecked())
        if hasattr(self, "sp_grade"):
            s.cognitive.grade_medium = float(self.sp_grade.value())
            s.cognitive.quadrant_percentile = float(self.sp_pct.value())

    def _on_preset(self) -> None:
        """换服务商 = 换地址 + 换候选模型 + 换 key 来源。不动用户已经填的 key。"""
        key = self.cmb_preset.currentData()
        p = provider(key)
        if p.get("base_url"):
            self.ed_base.setText(p["base_url"])
        self._fill_models(p.get("models") or [], p.get("model") or "")
        vision = p.get("vision_model") or p.get("model") or ""
        self._fill_vision(p.get("vision_models") or [], vision)
        same = not p.get("vision_models") and not (p.get("vision_model") or "")
        self.chk_vision_same.setChecked(bool(p.get("vision_model") == p.get("model") or same))
        self.cmb_vision.setEnabled(not self.chk_vision_same.isChecked())
        self._show_preset_hint(p)
        if p.get("asr_model"):
            self.ed_asr_model.setText(p["asr_model"])
        self.lbl_llm_status.setText("")

    def _on_vision_same(self, checked: bool) -> None:
        self.cmb_vision.setEnabled(not checked)

    def _on_base_edited(self) -> None:
        """手改地址后，原来的候选模型就不作数了，提示去刷新。"""
        p = provider(self.cmb_preset.currentData())
        if self.ed_base.text().strip() != (p.get("base_url") or "").strip():
            self.lbl_llm_status.setText("地址改过了，点「刷新列表」取这个端点真正支持的模型。")

    def _on_open_key_site(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        url = provider(self.cmb_preset.currentData()).get("key_url") or ""
        if url:
            QDesktopServices.openUrl(QUrl(url))

    # -- 拉真实模型列表 -------------------------------------------------
    def _on_fetch_models(self) -> None:
        self._collect_settings()
        self.btn_models.setEnabled(False)
        self.lbl_llm_status.setText("正在连端点取模型列表…")
        llm = build_llm(self.settings)
        if not llm.configured:
            self.btn_models.setEnabled(True)
            self.lbl_llm_status.setText("还没填接入地址。")
            return
        w = Worker(llm.probe, False)        # deep=False：只列模型，不动对话
        w.done.connect(self._models_done)
        w.failed.connect(self._models_failed)
        self._spawn(w)

    def _models_failed(self, err: str) -> None:
        self.btn_models.setEnabled(True)
        self.lbl_llm_status.setText(f"取模型列表失败：{err}")
        self.lbl_llm_status.setStyleSheet(f"color:{theme.WARN}; font-size:12px;")

    def _models_done(self, res) -> None:
        self.btn_models.setEnabled(True)
        if not res.models:
            self.lbl_llm_status.setText(
                "端点没有返回模型列表（有些自建网关不支持 /models）。"
                "可以直接在「模型」里手填名字。" + (f" 错误：{res.error}" if res.error else ""))
            self.lbl_llm_status.setStyleSheet(f"color:{theme.WARN}; font-size:12px;")
            return
        p = provider(self.cmb_preset.currentData())
        cur = self.cmb_llm_model.currentText().strip()
        self._fill_models(sorted(res.models), cur)
        vision = vision_of(res.models, p.get("vision_models"))
        self._fill_vision(vision or sorted(res.models), self.cmb_vision.currentText().strip())
        note = f"✓ 已连上，{len(res.models)} 个模型可选"
        if res.latency_ms:
            note += f"（{res.latency_ms:.0f} ms）"
        if cur and cur not in res.models:
            note += f"｜注意：当前「{cur}」不在这个列表里，可能不可用"
        self.lbl_llm_status.setText(note)
        self.lbl_llm_status.setStyleSheet(
            f"color:{theme.OK}; font-size:12px;" if cur in res.models or not cur
            else f"color:{theme.WARN}; font-size:12px;")

    def _autofetch_models(self) -> None:
        """开软件时顺手把模型列表拉一次，用户打开设置页就是可选状态。"""
        try:
            if getattr(self, "_models_fetched", False):
                return
            llm = self.settings.llm
            if llm.backend == "off" or not llm.base_url:
                return
            is_local = "127.0.0.1" in llm.base_url or "localhost" in llm.base_url
            if not (llm.api_key or is_local):
                return          # 没 key 就别去撞 401，省得刷一条红字
            self._models_fetched = True
            self._on_fetch_models()
        except Exception:  # noqa: BLE001
            pass

    def _fill_capture_devices(self) -> None:
        from dsh_app.capture import list_loopback_devices
        cur = self.settings.capture.device_name or ""
        self.cmb_capture.clear()
        try:
            devs = list_loopback_devices()
        except Exception as e:  # noqa: BLE001
            self.cmb_capture.addItem(f"（枚举失败：{e}）", "")
            return
        self.cmb_capture.addItem("默认输出设备的环回（推荐）", "")
        for d in devs:
            mark = "  ◀ 当前默认输出" if d.get("default") else ""
            self.cmb_capture.addItem(f"{d['name']}{mark}", d["name"])
        idx = self.cmb_capture.findData(cur)
        self.cmb_capture.setCurrentIndex(max(0, idx))

    def _on_probe_capture(self) -> None:
        from dsh_app.capture import list_loopback_devices, probe_loopback
        name = self.cmb_capture.currentData() or ""
        try:
            devs = list_loopback_devices()
        except Exception as e:  # noqa: BLE001
            self.lbl_capture_hint.setText(f"枚举设备失败：{e}")
            return
        # 下拉选「默认输出设备的环回」时，必须试读真正的默认环回，
        # 而不是列表里的第一个设备——否则测的和将来录的不是同一个。
        target = None
        if name:
            target = next((d for d in devs if d["name"] == name), None)
        if target is None:
            target = next((d for d in devs if d.get("default")), None)
        if target is None and devs:
            target = devs[0]
        if target is None:
            self.lbl_capture_hint.setText("没有找到可用的环回设备。")
            return
        self.btn_capture_probe.setEnabled(False)
        self.lbl_capture_hint.setText(f"正在试读「{target['name']}」…")
        try:
            r = probe_loopback(int(target["index"]), 1.2)
        except Exception as e:  # noqa: BLE001
            self.lbl_capture_hint.setText(f"试读异常：{e}")
            self.btn_capture_probe.setEnabled(True)
            return
        self.btn_capture_probe.setEnabled(True)
        if r.get("ok"):
            self.lbl_capture_hint.setText(
                f"✓「{target['name'][:34]}」正在出数据"
                f"（{r.get('bytes', 0)} 字节 / {r.get('elapsed', 0)}s），可以录音。")
            self.lbl_capture_hint.setStyleSheet(f"color:{theme.OK}; font-size:12px; padding:2px 4px;")
        else:
            self.lbl_capture_hint.setText(
                f"✗「{target['name'][:34]}」1.2 秒内没有收到音频数据："
                f"如果这是蓝牙耳机，请先在浏览器开始播放课程再试。"
                + (f"（{r['reason']}）" if r.get("reason") else ""))
            self.lbl_capture_hint.setStyleSheet(f"color:{theme.WARN}; font-size:12px; padding:2px 4px;")

    def _on_save_settings(self) -> None:
        self._collect_settings()
        self.settings.save()
        self.engine.llm = build_llm(self.settings)
        self._update_backend_badge()
        self._start_autocap()          # 开关变了就立刻生效，不用重启
        self.status.showMessage("设置已保存", 4000)

    def _update_backend_badge(self) -> None:
        s = self.settings
        if s.llm.backend == "off":
            self.badge.setText("后端：生成已关闭")
        elif "127.0.0.1" in (s.llm.base_url or "") or "localhost" in (s.llm.base_url or ""):
            self.badge.setText(f"后端：本地 {s.llm.model}")
        else:
            key_state = "已配置 key" if s.llm.api_key else "缺 api_key"
            self.badge.setText(f"后端：云端 {s.llm.model}（{key_state}）")

    # ------------------------------------------------------------------
    # 灵敏度 / 存储 / 数据
    # ------------------------------------------------------------------
    def _load_sensitivity(self) -> None:
        cog = self.settings.cognitive
        self.sp_grade.setValue(float(getattr(cog, "grade_medium", 0.45)))
        self.sp_pct.setValue(float(getattr(cog, "quadrant_percentile", 70.0)))
        # 对得上预设就选中它，对不上就留在当前项（手调过的值不该被悄悄改掉）
        for i in range(self.cmb_sens.count()):
            gm, pct = self.cmb_sens.itemData(i)
            if abs(gm - self.sp_grade.value()) < 1e-6 and abs(pct - self.sp_pct.value()) < 1e-6:
                self.cmb_sens.blockSignals(True)
                self.cmb_sens.setCurrentIndex(i)
                self.cmb_sens.blockSignals(False)
                return

    def _on_sens_preset(self) -> None:
        data = self.cmb_sens.currentData()
        if not data:
            return
        gm, pct = data
        self.sp_grade.setValue(float(gm))
        self.sp_pct.setValue(float(pct))

    def _on_reanalyze_current(self) -> None:
        if not self.current_session:
            self.status.showMessage("先选一节再重新分析", 6000)
            return
        self._on_analyze()

    def _refresh_discard_info(self) -> None:
        if not hasattr(self, "lbl_discard"):
            return
        try:
            st = self.db.discard_stats(days=30)
        except Exception:                        # noqa: BLE001
            return
        if not st["n"]:
            self.lbl_discard.setText(
                "最近 30 天没有被丢掉的自动抓取。当前最短保留 "
                f"{float(getattr(self.settings.capture, 'web_min_keep_s', 8.0)):.0f} 秒。")
            return
        cur = float(getattr(self.settings.capture, "web_min_keep_s", 8.0) or 8.0)
        advice = st["max_s"] + 1.0
        tail = (f"建议调到 {advice:.0f} 秒（比最长那次丢掉的再长 1 秒）"
                if advice > cur else "阈值看着够用，先不动")
        self.lbl_discard.setText(
            f"最近 30 天丢掉 {st['n']} 次：中位 {st['median_s']:.1f} 秒、最长 "
            f"{st['max_s']:.1f} 秒；当前阈值 {cur:.0f} 秒。{tail}。")

    def _on_apply_discard_hint(self) -> None:
        st = self.db.discard_stats(days=30)
        if not st["n"]:
            self.status.showMessage("还没有记录，先攒几次再说", 7000)
            return
        val = round(st["max_s"] + 1.0, 1)
        self.settings.capture.web_min_keep_s = val
        try:
            self.settings.save()
        except Exception as e:                   # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self._refresh_discard_info()
        self.status.showMessage(f"最短保留时长已改为 {val:.1f} 秒", 8000)

    def _on_clear_discards(self) -> None:
        self.db.clear_capture_discards()
        self._refresh_discard_info()
        self.status.showMessage("已清空丢弃记录", 6000)

    def _refresh_kb_auto_info(self) -> None:
        if not hasattr(self, "lbl_kb_auto"):
            return
        try:
            seen = len(self.engine.watched_sessions())
            docs = [d for d in self.engine.kb_docs(None) if str(d["path"]).startswith("session:")]
        except Exception:                        # noqa: BLE001
            return
        self.lbl_kb_auto.setText(
            f"已归档 {len(docs)} 节 / 看过 {seen} 节。归档内容带时间码，"
            "问模型时能当依据引出来。")

    def _on_archive_all_kb(self) -> None:
        def job():
            return self.engine.kb_ingest_all()

        def done(r):
            self._refresh_kb_auto_info()
            self._refresh_kb()
            self.status.showMessage(
                f"归档完成：{r['done']} 节成功，{r['failed']} 节跳过", 9000)

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: QMessageBox.warning(self, "归档失败", e))
        self._spawn(w)

    def _refresh_storage_info(self) -> None:
        try:
            u = maintenance.audio_usage()
            last = maintenance.last_backup_time()
            if hasattr(self, "lbl_storage"):
                old = (time.strftime("%Y-%m-%d", time.localtime(u["oldest"]))
                       if u["oldest"] else "—")
                self.lbl_storage.setText(
                    f"音频 {u['files']} 个文件，共 {u['text']}；最老一条 {old}")
            if hasattr(self, "lbl_backup"):
                when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(last))
                        if last else "还没有备份")
                n = len(maintenance.list_backups())
                self.lbl_backup.setText(f"上次备份：{when}（共 {n} 份，自动备份留 7 份）")
            cap_cfg = self.settings.capture
            if hasattr(self, "sp_keep_days"):
                self.sp_keep_days.setValue(float(getattr(cap_cfg, "keep_audio_days", 7)))
            if hasattr(self, "chk_compress"):
                self.chk_compress.setChecked(bool(getattr(cap_cfg, "audio_compress", True)))
            if hasattr(self, "chk_keep_raw"):
                self.chk_keep_raw.setChecked(bool(getattr(cap_cfg, "save_raw_audio", True)))
        except Exception:                  # noqa: BLE001
            pass

    def _open_dir(self, path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def _on_sweep_audio(self) -> None:
        try:
            r = maintenance.sweep_audio(self.db, self.settings)
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "清理失败", str(e))
            return
        if r.get("skipped"):
            msg = "保留天数是 0（永不清理），没有删任何文件"
        else:
            msg = f"删掉 {r['deleted']} 个音频文件，释放 {r['text']}（课和回看点都还在）"
        self.status.showMessage(msg, 9000)
        self._refresh_storage_info()

    def _on_backup_now(self) -> None:
        try:
            dest = maintenance.backup_db(self.db, tag="manual")
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "备份失败", str(e))
            return
        self.status.showMessage(f"已备份：{dest.name}", 9000)
        self._refresh_storage_info()

    def _on_restore_backup(self) -> None:
        items = maintenance.list_backups()
        if not items:
            QMessageBox.information(self, "没有备份", "先点一次「立即备份」，备份目录就有了。")
            return
        # 恢复会覆盖当前库：分析线程还在写的时候换文件，等于把这次分析结果扔掉
        if self._workers or self._live_count:
            QMessageBox.warning(self, "先等一下",
                                "有分析或录制正在进行。等它跑完再恢复，"
                                "否则这次的结果会被覆盖掉。")
            return
        labels = [f"{it['when']}　{it['name']}　{it['text']}" for it in items]
        label, ok = QInputDialog.getItem(self, "从备份恢复", "选一份备份：", labels, 0, False)
        if not ok:
            return
        target = items[labels.index(label)]["path"]
        if QMessageBox.question(
                self, "确认恢复",
                f"用「{target.name}」覆盖当前数据库？\n"
                "当前库会先自动备份一份（pre-restore），手滑也找得回来。") != QMessageBox.Yes:
            return
        try:
            r = maintenance.restore_backup(self.db, target)
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "恢复失败", str(e))
            return
        # 库换了，界面上所有东西都得重新读一遍
        self.current_session = ""
        self.current_knots = []
        self._refresh_sessions()
        self._refresh_storage_info()
        self.status.showMessage(
            f"已恢复 {r['restored_from']}（{r['sessions']} 节）；旧库留在 {r['safety']}", 12000)

    def _maybe_daily_backup(self) -> None:
        """每天第一次打开自动备份一次。状态记在 ui_state.json，不额外开一张表。

        双保险：除了看状态文件里的日期，还看备份目录里最新那份的时间。
        状态文件是和问模型小窗共用的，任何一边整体重写都会把对方的键带走；
        只看日期的话，键丢了就会变成"每开一次备份一次"。
        """
        try:
            state = self._load_ui_state()
            today = time.strftime("%Y-%m-%d")
            if state.get("backup_last") == today:
                return
            if time.time() - maintenance.last_backup_time() < 20 * 3600:
                return                      # 20 小时内备份过，够了
            maintenance.backup_db(self.db, tag="auto")
            state["backup_last"] = today
            self._save_ui_state(state)
            self._refresh_storage_info()
        except Exception:                  # noqa: BLE001
            pass

    def _startup_maintenance(self) -> None:
        """启动后顺手做两件事：清过期音频、报告磁盘家底。都不打断用户。"""
        try:
            r = maintenance.sweep_audio(self.db, self.settings)
            if r.get("deleted"):
                self.status.showMessage(
                    f"已按保留策略清理 {r['deleted']} 个过期音频，释放 {r['text']}", 8000)
            self._refresh_storage_info()
        except Exception:                  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 回看：只给时间点，不假装能回放
    # ------------------------------------------------------------------
    def _on_view_changed(self, a: float, b: float) -> None:
        """视野变了就更新右上角那枚小标签，让人知道"现在看的是哪一段"。"""
        if not hasattr(self, "lbl_zoom_state"):
            return
        total = max(1e-6, float(self.timeline.duration or 0.0))
        if b - a >= total - 1.0:
            self.lbl_zoom_state.setText("全貌")
        else:
            mins = (b - a) / 60.0
            span = f"{mins:.0f} 分钟" if mins >= 1 else f"{b - a:.0f} 秒"
            self.lbl_zoom_state.setText(f"{_mmss(a)}–{_mmss(b)}（{span}）")

    def _current_knot_row(self) -> Any | None:
        row = self.knot_table.currentRow()
        if row < 0:
            return None
        item = self.knot_table.item(row, 0)
        kid = item.data(Qt.UserRole) if item is not None else None
        for k in self.current_knots:
            if int(getattr(k, "id", -1)) == int(kid or -1):
                return k
        return None

    def _selected_knot_any(self) -> Any | None:
        """回看点表选中的行；没选就退回时间轴上选中的那个。"""
        k = self._current_knot_row()
        if k is not None:
            return k
        kid = getattr(self.timeline, "selected_id", -1)
        for k2 in self.current_knots:
            if int(getattr(k2, "id", -1)) == int(kid or -1):
                return k2
        return None

    def _on_copy_timestamp(self, with_evidence: bool = False) -> None:
        """把这一刻抄到剪贴板：视频还在原播放器里，用户自己拖过去看。

        with_evidence=True 时连判定依据一起抄——贴到笔记里就自带"为什么是这里"。
        """
        k = self._selected_knot_any()
        if k is None:
            self.status.showMessage("先选一个回看点（表里一行，或时间轴上的缎带）", 8000)
            return
        ts = _mmss(k.t_start)
        if with_evidence:
            grade = "重点回看" if k.grade == "strong" else "一般回看"
            body = f"{ts}　{grade}（{k.score:.2f}）｜{k.evidence or '（没有依据文本）'}"
            if k.excerpt:
                body += "\n原话：" + str(k.excerpt)
            QApplication.clipboard().setText(body)
            self.lbl_ts.setText(f"已复制 {ts} 和判定依据")
            self.lbl_ts.setStyleSheet(f"color:{theme.ACCENT_PRESS}; font-size:12px;")
            self.status.showMessage(f"已复制 {ts} 的时间点和判定依据（可直接贴进笔记）", 9000)
            return
        row = self.db.get_session(self.current_session) if self.current_session else None
        vurl = str(row["video_url"] or "") if row else ""
        if vurl:
            # 有原始链接就复制**带时间点的深链**：点开直接落在那一秒，不用自己拖
            sep = "&" if "?" in vurl else "?"
            link = f"{vurl}{sep}t={int(float(k.t_start))}"
            QApplication.clipboard().setText(link)
            self.lbl_ts.setText(f"已复制 {ts} 的链接（点开直接到这一刻）")
            self.lbl_ts.setStyleSheet(f"color:{theme.ACCENT_PRESS}; font-size:12px;")
            self.status.showMessage(f"已复制带时间点的链接：{link}", 12000)
            return
        QApplication.clipboard().setText(ts)
        self.lbl_ts.setText(f"已复制 {ts}　←　回视频里拖到这一刻")
        self.lbl_ts.setStyleSheet(f"color:{theme.ACCENT_PRESS}; font-size:12px;")
        self.status.showMessage(
            f"已复制 {ts}：切到视频那个窗口，把进度条拖到这一刻（回看点前后各听半分钟）", 9000)

    def _where_to_watch(self) -> str:
        """这节该回哪儿看：自动抓的网页视频知道站点，手动录的就只能说"视频"。"""
        sid = self.current_session
        if not sid:
            return "视频"
        try:
            row = self.db.get_session(sid)
        except Exception:                        # noqa: BLE001
            return "视频"
        if not row:
            return "视频"
        if str(row["source"] or "") != "webvideo":
            return "视频"
        # 自动抓取的课是按站点建的（"哔哩哔哩·网课"），拿它当"去哪儿看"最直观
        try:
            crows = self.db.query("SELECT title FROM courses WHERE id=?", (row["course_id"],))
            course = str(crows[0]["title"]).strip() if crows else ""
        except Exception:                        # noqa: BLE001
            course = ""
        return f"{course} 的播放器" if course else "视频"

    def _refresh_ts_hint(self) -> None:
        """选中行变了就更新提示：这一行的时刻是多少，以及"去哪儿看"。"""
        if not hasattr(self, "lbl_ts"):
            return
        k = self._selected_knot_any()
        if k is None:
            self.lbl_ts.setText("选中一行，把时间点抄下来回视频里看")
            self.lbl_ts.setStyleSheet("")
            return
        grade = "重点回看" if k.grade == "strong" else "一般回看"
        where = self._where_to_watch()
        self.lbl_ts.setText(f"{_mmss(k.t_start)}　{grade}　—　回{where}拖到这一刻看那段板书")
        self.lbl_ts.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px;")

    def _on_dismiss_knot(self) -> None:
        k = self._selected_knot_any()
        if k is None:
            self.status.showMessage("先选一个回看点", 6000)
            return
        try:
            self.engine.dismiss_knot(int(k.id))
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "标记失败", str(e))
            return
        self._reload_knots_after_feedback(int(k.id))

    def _on_confirm_knot(self) -> None:
        k = self._selected_knot_any()
        if k is None:
            self.status.showMessage("先选一个回看点", 6000)
            return
        try:
            self.engine.confirm_knot(int(k.id))
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "记录失败", str(e))
            return
        self.status.showMessage("已记录：确实卡住（这条反馈会进校准样本）", 8000)

    def _on_restore_knot(self) -> None:
        k = self._selected_knot_any()
        if k is None:
            self.status.showMessage("先选一个回看点", 6000)
            return
        try:
            self.engine.restore_knot(int(k.id))
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "撤销失败", str(e))
            return
        self._reload_knots_after_feedback(int(k.id))

    def _reload_knots_after_feedback(self, keep_id: int | None = None) -> None:
        """反馈之后重画表格，并把刚才那一行重新选中。

        不重选的话，"标错了 → 点撤销"这个最自然的动作会静默失效：撤销按钮
        拿不到当前行，只在状态栏说一句"先选一个回看点"，人以为按钮坏了。
        """
        self.current_knots = self.db.get_knots(self.current_session) if self.current_session else []
        self._fill_knot_table(self.current_knots)
        self._fill_reword_knots()
        if keep_id is not None:
            self._select_knot_row(int(keep_id))
        self._safe_refresh_kp()

    def _select_knot_row(self, knot_id: int) -> None:
        """按回看点 id 选中表格里那一行（找不到就什么都不做）。"""
        kid = int(knot_id)
        for r in range(self.knot_table.rowCount()):
            item = self.knot_table.item(r, 0)
            if item is not None and int(item.data(Qt.UserRole) or -1) == kid:
                self.knot_table.selectRow(r)
                self.timeline.set_selected(kid)
                return

    def _on_session_menu(self, pos) -> None:
        """左栏会话右键：重命名 / 移到别的课 / 删除。

        整理这件事必须能就地做。以前只能靠"重录一遍"，课程列表越用越乱。
        """
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        if not sid:
            return
        row = self.db.get_session(sid)
        if row is None:
            return
        menu = QMenu(self)
        a_ren = menu.addAction("重命名本讲…")
        sub = menu.addMenu("移到课程")
        courses = self.db.list_courses()
        acts_course = {}
        for c in courses:
            act = sub.addAction(c["title"])
            act.setCheckable(True)
            act.setChecked(c["id"] == row["course_id"])
            acts_course[act] = c["id"]
        if not courses:
            sub.setEnabled(False)
        # 转写是唯一会"失败"的环节（模型没下、字幕缺、音频太短），所以这里给一条
        # 重试入口：失败的那一节不必删掉重录，音频还在就再跑一次。
        a_watch = menu.addAction("看这一节（浏览器打开并开录）")
        a_watch.setEnabled(bool(str(row["video_url"] or "")))
        a_retry = menu.addAction("重新转写这一节")
        a_retry.setEnabled(bool(str(row["audio_path"] or "")))
        menu.addSeparator()
        a_del = menu.addAction("删除本讲…")
        chosen = menu.exec(self.session_list.viewport().mapToGlobal(pos))
        if chosen is a_watch:
            self.current_session = sid
            self._load_session_view()
            self._open_session_video(sid)
        elif chosen is a_retry:
            self.current_session = sid
            self.status.showMessage("重新转写中…", 8000)
            self._run_pipeline(sid, auto=False)
        elif chosen is a_ren:
            self._rename_session(sid, str(row["title"] or ""))
        elif chosen is a_del:
            self._delete_session(sid, str(row["title"] or ""))
        elif chosen in acts_course:
            cid = acts_course[chosen]
            if cid != row["course_id"]:
                self.db.update_session(sid, course_id=cid)
                self._refresh_sessions()
                self.status.showMessage("已移到别的课程", 6000)

    def _rename_session(self, sid: str, old: str) -> None:
        name, ok = QInputDialog.getText(self, "重命名本讲", "名字：", text=old)
        if not ok:
            return
        name = (name or "").strip()
        if not name or name == old:
            return
        self.db.update_session(sid, title=name)
        self._refresh_sessions()
        self.status.showMessage(f"已改名为「{name}」", 6000)

    def _delete_session(self, sid: str, title: str) -> None:
        row = self.db.get_session(sid)
        n_knots = len(self.db.get_knots(sid))
        if QMessageBox.question(
                self, "删除本讲",
                f"删掉「{title or sid}」？\n\n"
                f"会一起删掉这节的转写、{n_knots} 个回看点、题目和音频记录。\n"
                "课的知识点掌握度会保留（别的节可能也讲过）。\n"
                "删之前建议先点一次「立即备份」。") != QMessageBox.Yes:
            return
        if self._live_count and self.current_session == sid:
            QMessageBox.warning(self, "先停下", "这一节正在录制/分析，先停再删。")
            return
        try:
            self.db.delete_session(sid)
        except Exception as e:             # noqa: BLE001
            QMessageBox.warning(self, "删除失败", str(e))
            return
        if self.current_session == sid:
            self.current_session = ""
            self.current_knots = []
            self._clear_session_views()
        self._refresh_sessions()
        self._refresh_storage_info()
        self.status.showMessage(f"已删除「{title}」", 7000)

    def _clear_session_views(self) -> None:
        """当前会话被删掉之后，把依赖它的视图清空，别留着上一节的数据。"""
        try:
            self.timeline.set_data([], [], 0.0)
            self._fill_knot_table([])
            self._fill_reword_knots()
            self._clear_quiz_area()
            self._ensure_quiz_placeholder()
            self.map_view.setHtml("")
            self.lbl_coverage.setText("")
        except Exception:                  # noqa: BLE001
            pass

    def _on_knot_menu(self, pos) -> None:
        """表格右键：和下面那排按钮同一套动作，手在表上时不用来回移。"""
        row = self.knot_table.rowAt(pos.y())
        if row >= 0:
            self.knot_table.selectRow(row)
        menu = QMenu(self)
        a1 = menu.addAction(icons.get("link"), "复制时间点")
        a1b = menu.addAction(icons.get("export"), "复制时间点 + 判定依据")
        a_open = menu.addAction(icons.get("play"), "在浏览器打开这一秒")
        sess_row = self.db.get_session(self.current_session) if self.current_session else None
        a_open.setEnabled(bool(str((sess_row["video_url"] if sess_row else "") or "")))
        menu.addSeparator()
        a2 = menu.addAction(icons.get("check"), "确实卡住")
        a3 = menu.addAction(icons.get("close"), "不算回看点")
        a4 = menu.addAction(icons.get("refresh"), "撤销忽略")
        chosen = menu.exec(self.knot_table.viewport().mapToGlobal(pos))
        if chosen is a1:
            self._on_copy_timestamp()
        elif chosen is a1b:
            self._on_copy_timestamp(with_evidence=True)
        elif chosen is a_open:
            k2 = self._selected_knot_any()
            if k2 is not None:
                self._open_session_video(self.current_session, float(k2.t_start))
        elif chosen is a2:
            self._on_confirm_knot()
        elif chosen is a3:
            self._on_dismiss_knot()
        elif chosen is a4:
            self._on_restore_knot()

    def _active_knots(self) -> list:
        """参与统计与建议的回看点：用户判过"不算"的不再出现（表里仍然能看到）。"""
        bad = self.db.dismissed_knot_ids(self.current_session) if self.current_session else set()
        return [k for k in self.current_knots if int(getattr(k, "id", -1)) not in bad]

    # ------------------------------------------------------------------
    # 托盘与复习提醒
    # ------------------------------------------------------------------
    def _build_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return                          # 离屏/无桌面环境：整块跳过
        try:
            tray = QSystemTrayIcon(icons.app_icon(64), self)
            menu = QMenu(self)
            act_show = menu.addAction("显示主窗口")
            act_show.triggered.connect(self._show_from_tray)
            self.act_due = menu.addAction("今天该复习 0 个")
            self.act_due.triggered.connect(self._goto_review)
            menu.addSeparator()
            act_quit = menu.addAction("退出")
            act_quit.triggered.connect(self.close)
            tray.setContextMenu(menu)
            tray.setToolTip(APP_NAME)
            tray.activated.connect(lambda why: self._show_from_tray()
                                   if why == QSystemTrayIcon.DoubleClick else None)
            tray.show()
            self._tray = tray
        except Exception:                  # noqa: BLE001
            self._tray = None

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _goto_review(self) -> None:
        self._show_from_tray()
        self.nav.setCurrentRow(3)           # 漏洞地图：复习顺序就在这页

    def _check_review_notice(self) -> None:
        try:
            due = self.db.due_points()
        except Exception:                  # noqa: BLE001
            return
        if self._tray is not None:
            self._tray.setToolTip(f"{APP_NAME}　今天该复习 {len(due)} 个知识点")
            if hasattr(self, "act_due"):
                self.act_due.setText(f"今天该复习 {len(due)} 个")
        if not due:
            return
        today = time.strftime("%Y-%m-%d")
        state = self._load_ui_state()
        if state.get("notice_last") == today:
            return                          # 一天只提醒一次，别变成骚扰
        names = "、".join(str(r["name"]) for r in due[:4])
        more = f" 等 {len(due)} 个" if len(due) > 4 else ""
        msg = f"{names}{more}"
        if self._tray is not None:
            self._tray.showMessage("今天有知识点该复习了", msg,
                                   QSystemTrayIcon.Information, 8000)
        self.status.showMessage(f"该复习了：{msg}（漏洞地图里看顺序）", 12000)
        state["notice_last"] = today
        self._save_ui_state(state)

    # ------------------------------------------------------------------
    # 启动探测
    # ------------------------------------------------------------------
    def _startup_probe(self) -> None:
        llm = self.settings.llm
        if llm.backend == "off":
            self.status.showMessage("生成层已关闭：回看点识别照常，重讲不可用")
            return
        # 云端没有 key 就不要去撞端点：否则每次启动都刷一条 401，既难看又浪费请求
        is_local = "127.0.0.1" in (llm.base_url or "") or "localhost" in (llm.base_url or "")
        if llm.backend == "cloud" and not llm.api_key:
            self.status.showMessage(
                "生成层待配置：设置页填入 api_key 后即可用云端重讲（回看点识别不受影响）")
            return
        if not (llm.base_url and (llm.api_key or is_local)):
            self.status.showMessage("尚未配置模型端点：回看点识别可离线运行，重讲需要配置后可用")
            return
        if llm.preload_on_start:
            self.status.showMessage("正在预热模型…")
            w = Worker(self.engine.preload_llm)
            w.done.connect(lambda r: self.status.showMessage(f"{r[1]}", 8000))
            w.failed.connect(lambda e: self.status.showMessage(f"预热失败：{e[:120]}", 8000))
            self._spawn(w)

    def _safe_refresh_kp(self) -> None:
        try:
            self.refresh_kp_table()
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self, w: Worker) -> None:
        self._workers.append(w)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        w.start()

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def eventFilter(self, obj, ev):  # noqa: N802
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        if obj is getattr(self, "ed_ask", None) and ev.type() == QEvent.KeyPress:
            ke = ev
            if isinstance(ke, QKeyEvent) and ke.key() in (Qt.Key_Return, Qt.Key_Enter):
                if ke.modifiers() & Qt.ShiftModifier:
                    return False          # Shift+Enter 正常换行
                self._on_ask()
                return True
        return super().eventFilter(obj, ev)

    def _refresh_sessions(self) -> None:
        self.session_list.clear()
        for r in self.db.list_sessions():
            course = self.db.query("SELECT title FROM courses WHERE id=?", (r["course_id"],))
            ct = course[0]["title"] if course else "未命名课程"
            title = (r["title"] or "未命名").strip()
            head = title if title.startswith("第") else f"第 {r['idx']} 节　{title}"
            item = QListWidgetItem(f"{head}\n{ct}　{_mmss(r['duration'] or 0)}")
            item.setData(Qt.UserRole, r["id"])
            self.session_list.addItem(item)
        # 左栏底部一行统计：一节课的时候左栏很空，这里给个"家底"
        try:
            cid = self._current_course_id()
            n_knot = self.db.count_knots(cid) if cid else 0
            n_kp = len(self.engine.kp_overview(cid)) if cid else 0
            self.lbl_side_stats.setText(
                f"共 {self.session_list.count()} 节 · 回看点 {n_knot} 个 · 知识点 {n_kp} 个")
        except Exception:                          # noqa: BLE001
            self.lbl_side_stats.setText("")
        if self.session_list.count() and not self.current_session:
            self.session_list.setCurrentRow(0)
            self._on_session_activated(self.session_list.item(0))

    def _on_session_activated(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.UserRole)
        if not sid or sid == self.current_session:
            return
        self.current_session = sid
        # 小窗开着的时候切课：不跟着换会拿上一节的回看点做上下文，答得莫名其妙
        if getattr(self, "_chat_win", None) is not None:
            self._chat_win.session_id = sid
        self._load_session_view()
        self._fill_reword_knots()
        try:
            self.refresh_kp_table()
        except Exception:  # noqa: BLE001
            pass

    def _load_session_view(self) -> None:
        sid = self.current_session
        if not sid:
            return
        if hasattr(self, "lbl_ex"):
            self.refresh_exercise_label()
        wins = [dict(r) for r in self.db.get_windows(sid)]
        knots = self.db.get_knots(sid)
        self.current_knots = knots
        row = self.db.get_session(sid)
        dur = float(row["duration"] or 0) if row else 0.0
        bands = [
            KnotBand(id=k.id, t_start=k.t_start, t_end=k.t_end, score=k.score,
                     grade=k.grade, level=k.level, evidence=k.evidence)
            for k in knots
        ]
        self.timeline.set_data(wins, bands, dur)
        self._fill_knot_table(knots)
        self._fill_reword_knots()
        # 换了节就把题目面板清掉：留着上一节的题会让人以为这一节测过了
        if self._quiz_session and self._quiz_session != sid:
            self._clear_quiz_area()
            self._quiz_session = ""
        try:
            self.refresh_quiz_knots()
            self.refresh_questions()
        except Exception:  # noqa: BLE001
            pass
        self._refresh_coverage_hint()
        vurl = str(row["video_url"] or "") if row else ""
        if vurl and not wins and not knots:
            self.status.showMessage(
                "这一节还没录：右键「看这一节」在浏览器打开，播放就会自动开始录", 12000)
            return
        self.status.showMessage(f"{len(knots)} 个回看点，{len(wins)} 个分析窗口")

    def _refresh_coverage_hint(self) -> None:
        """覆盖度诚实提示：产出的画面 ≠ 覆盖了所有回看点（借 visual_coverage 的原则）。"""
        sid = self.current_session
        if not sid or not hasattr(self, "lbl_coverage"):
            return
        try:
            cov = self.engine.capture_coverage(sid)
        except Exception:  # noqa: BLE001
            return
        if not cov["knots"]:
            self.lbl_coverage.setText("")
            return
        if cov["covered"] == cov["knots"]:
            self.lbl_coverage.setText(
                f"画面覆盖：{cov['covered']}/{cov['knots']} 个回看点有画面证据"
                f"（共 {cov['frames']} 帧）")
            self.lbl_coverage.setStyleSheet(f"color:{theme.OK}; font-size:12px;")
        else:
            self.lbl_coverage.setText(
                f"画面覆盖：{cov['covered']}/{cov['knots']} 个回看点有画面证据"
                f"（共 {cov['frames']} 帧）——{cov['missing']} 个回看点只有声音，没有画面")
            self.lbl_coverage.setStyleSheet(f"color:{theme.WARN}; font-size:12px;")

    def _fill_knot_table(self, knots) -> None:
        self.knot_table.setRowCount(0)
        # 按时间排，和左侧时间轴从左到右一致；严重程度看颜色和分数列就够
        for k in sorted(knots, key=lambda x: x.t_start):
            r = self.knot_table.rowCount()
            self.knot_table.insertRow(r)
            vals = [
                _mmss(k.t_start),
                "重点" if k.grade == "strong" else "一般",
                f"{k.score:.2f}",
                QUADRANT_CN.get(k.quadrant, k.quadrant),
                k.evidence or "",
                {"pending": "待解决", "explained": "已重讲",
                 "mastered": "已掌握", "dismissed": "已忽略"}.get(k.status, k.status),
            ]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                if c == 1 and k.status != "dismissed":
                    it.setForeground(QColor(f"{theme.BAD}" if k.grade == "strong" else f"{theme.WARN}"))
                # 已忽略的行整行变灰：还看得见（能撤销），但不抢注意力
                if k.status == "dismissed":
                    it.setForeground(QColor(theme.TEXT_3))
                    it.setToolTip("你标记过「不算回看点」：不参与重讲选项、统计和复习建议")
                self.knot_table.setItem(r, c, it)
            self.knot_table.item(r, 0).setData(Qt.UserRole, k.id)
        self.knot_table.resizeColumnsToContents()
        hdr = self.knot_table.horizontalHeader()
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        # 注意：循环变量不能叫 w —— 外层的 w 是正在建的 QWidget，覆盖掉会被回收
        for col, px in ((0, 78), (1, 68), (2, 62), (3, 96), (5, 78)):
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.knot_table.setColumnWidth(col, px)
        # 按行数定高：卡片贴合内容，多余的空间还给时间轴（不然下面一大片白）
        row_h = self.knot_table.verticalHeader().defaultSectionSize() or 30
        want = self.knot_table.horizontalHeader().height() + row_h * max(1, len(knots)) + 8
        h = max(104, min(want, 340))
        self.knot_table.setMinimumHeight(h)
        self.knot_table.setMaximumHeight(h)

    def _fill_reword_knots(self) -> None:
        self.reword_knot.clear()
        # 只列"还算数"的回看点：用户判过误判的不该继续出现在待重讲选项里
        for k in self._active_knots():
            # 选项只留"定位信息"，原文摘录放 tooltip：整段塞进下拉会把后面的控件挤走
            self.reword_knot.addItem(
                f"[{_mmss(k.t_start)}] {('强' if k.grade == 'strong' else '中')}回看点 "
                f"{k.score:.2f}", k.id)
            self.reword_knot.setItemData(self.reword_knot.count() - 1,
                                         (k.excerpt or k.evidence or "")[:400],
                                         Qt.ToolTipRole)
        if self.reword_knot.count() == 0 and self.current_session:
            self.reword_knot.addItem("（本节没有可用回看点：都被你标成误判了？右键可撤销）", None)

    def _on_knot_selected(self, knot_id: int) -> None:
        for r in range(self.knot_table.rowCount()):
            it = self.knot_table.item(r, 0)
            if it and it.data(Qt.UserRole) == knot_id:
                self.knot_table.selectRow(r)
                for i in range(self.reword_knot.count()):
                    if self.reword_knot.itemData(i) == knot_id:
                        self.reword_knot.setCurrentIndex(i)
                        break
                break

    def _open_reword_for_row(self, row: int) -> None:
        it = self.knot_table.item(row, 0)
        if not it:
            return
        kid = it.data(Qt.UserRole)
        for i in range(self.reword_knot.count()):
            if self.reword_knot.itemData(i) == kid:
                self.reword_knot.setCurrentIndex(i)
                break
        self.tabs.setCurrentIndex(1)
        self._on_reword()

    # ------------------------------------------------------------------
    # 录制
    # ------------------------------------------------------------------
    def _on_autoweb_toggled(self, on: bool) -> None:
        """勾选即开始盯浏览器音频会话；取消即停止（正在录的话按手动停收尾）。"""
        try:
            self.settings.capture.auto_web_capture = bool(on)
            self.settings.save()
        except Exception:                          # noqa: BLE001
            pass
        if on:
            self._start_web_watcher()
        else:
            self._stop_web_watcher("关闭自动录")

    def _maybe_autostart_web(self) -> None:
        if self.chk_autoweb.isChecked() and not self.engine.recording:
            self._start_web_watcher()

    def _start_web_watcher(self) -> None:
        from .. import webvideo as wv

        if getattr(self, "_web_watch", None) and self._web_watch.running:
            return
        cap_set = self.settings.capture
        procs = wv.BROWSER_PROCS
        if (cap_set.web_browsers or "").strip():
            procs = frozenset(wv._norm_proc(x) for x in
                              cap_set.web_browsers.replace("，", ",").split(",") if x.strip())
        cfg = wv.WatcherConfig(start_after_s=float(cap_set.web_start_after_s),
                               stop_after_s=float(cap_set.web_stop_after_s),
                               procs=procs)
        self._web_watch = wv.WebVideoWatcher(
            cfg,
            on_start=lambda meta: self._web_sig.start.emit(meta.as_dict()),
            on_stop=lambda meta, why, dur: self._web_sig.stop.emit(
                {"why": why, "duration": dur, **meta.as_dict()}),
            on_state=lambda snap: self._web_sig.state.emit(snap),
            # 会话可能赖着 Active（浏览器把音频流开着但没声音），
            # 所以停录还要求"回环里真的安静了"。判据必须是**这段录音自己的电平**：
            # 判不了东西（没在录 / 电平是陈的）时 is_capture_quiet 返回 False，
            # 也就是继续录——网课放到一半被切成十几个 20 秒碎片是更糟的错。
            is_quiet=self.engine.is_capture_quiet,
        )
        self._web_watch.start()
        if not self.engine.recording:
            self.lbl_rec.setText("待命（看网页视频自动录）")
            self.lbl_rec.setStyleSheet(f"color:{theme.TEXT_3};")
        self.status.showMessage("已开启网页视频自动录：去浏览器播放那节课就行（不区分站点）", 9000)

    def _web_capture_active(self) -> bool:
        """当前这次录制是不是自动起的（决定状态标签怎么说）。"""
        sid = getattr(self.engine, "_record_session_id", "") or ""
        if not sid:
            return False
        try:
            row = self.db.get_session(sid)
        except Exception:                        # noqa: BLE001
            return False
        return bool(row and str(row["source"]) == "webvideo")

    def _stop_web_watcher(self, why: str = "") -> None:
        w = getattr(self, "_web_watch", None)
        if w is None:
            return
        try:
            w.stop(reason=why or "manual")
        except Exception:                          # noqa: BLE001
            pass
        self._web_watch = None
        self.status.showMessage("已关闭网页视频自动录", 5000)

    # 下面的三个槽都在主线程里跑（线程里只 emit，不碰界面）
    def _web_on_start(self, meta: dict) -> None:
        if self.engine.recording:
            return
        self._web_last_proc = str(meta.get("proc") or "")
        try:
            sid = self.engine.start_web_capture(meta)
        except Exception as e:                     # noqa: BLE001
            self.status.showMessage(f"自动录失败：{str(e)[:120]}", 12000)
            return
        self.current_session = sid
        self._live_count = 0
        self._record_started_at = time.time()
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_rec.setText("● 自动录制中")
        self.lbl_rec.setStyleSheet(f"color:{theme.BAD};")
        title = meta.get("page_title") or meta.get("site") or "网页视频"
        self.status.showMessage(f"检测到 {meta.get('site', '网页')} 在放视频，已自动开始录：{title}",
                                12000)

    def _web_on_stop(self, info: dict) -> None:
        if not self.engine.recording:
            return
        res = self.engine.stop_web_capture(reason=str(info.get("why") or "silence"))
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_rec.setText("未录制")
        self.lbl_rec.setStyleSheet(f"color:{theme.TEXT_3};")
        if not res.get("ok"):
            self._refresh_sessions()
            return
        dur = float(res.get("duration") or 0)

        # 太短的抓取直接丢掉：浏览器弹个提示音、广告前奏、视频预览都会触发录制，
        # 录到 6 秒空白再送去转写，只会换来一句"转写失败"的报错框——那是噪音，
        # 不是用户想看的课。宁可什么都不留，也不要在列表里堆垃圾节。
        if dur < float(getattr(self.settings.capture, "web_min_keep_s",
                               AUTO_CAPTURE_MIN_KEEP_S) or AUTO_CAPTURE_MIN_KEEP_S):
            self._discard_short_capture(res.get("session_id", ""), dur)
            return

        self._refresh_sessions()
        self.status.showMessage(f"自动录制结束：{_mmss(dur)}，开始转写与分析…", 12000)
        self._run_pipeline(res["session_id"], auto=True)

    def _discard_short_capture(self, sid: str, dur: float) -> None:
        """把"只响了几秒"的自动抓取整个撤掉：会话、事件、音频一起清，**但记一笔**。"""
        if not sid:
            return
        row = self.db.get_session(sid)
        audio = str(row["audio_path"] or "") if row else ""
        try:
            course = self.db.query("SELECT title FROM courses WHERE id=?",
                                   (row["course_id"],)) if row else []
            self.db.add_capture_discard(
                proc=str(self._web_last_proc or ""),
                site=str(course[0]["title"] if course else ""),
                title=str(row["title"] or "") if row else "",
                duration=float(dur), reason="too_short",
                threshold=float(getattr(self.settings.capture, "web_min_keep_s", 8.0) or 8.0))
            self._refresh_discard_info()
        except Exception:                        # noqa: BLE001
            pass
        try:
            self.db.delete_session(sid)
        except Exception:                       # noqa: BLE001
            pass
        if audio:
            try:
                Path(audio).unlink(missing_ok=True)
            except Exception:                   # noqa: BLE001
                pass
        if self.current_session == sid:
            self.current_session = ""
        self._refresh_sessions()
        self.status.showMessage(
            f"只响了 {dur:.1f} 秒，当提示音忽略了（不是网课，没留记录）", 10000)

    def _web_on_state(self, snap: dict) -> None:
        """状态提示：只在 arming（等着看是不是真在看）时提示，别打扰。"""
        if snap.get("state") == "arming" and not self.engine.recording:
            self.lbl_rec.setText("… 检测到声音")
            self.lbl_rec.setStyleSheet(f"color:{theme.WARN};")
        elif snap.get("state") == "idle" and not self.engine.recording:
            if self.chk_autoweb.isChecked():
                self.lbl_rec.setText("待命（看网页视频自动录）")
                self.lbl_rec.setStyleSheet(f"color:{theme.TEXT_3};")

    def _on_record(self) -> None:
        if self.engine.recording:
            return
        from PySide6.QtWidgets import QInputDialog
        course = self._ask_course_name("开始录制 · 选课程", self._current_course_title())
        if not course:
            return
        # 节号按**这门课**已有的节数算，不是全部会话数
        cid = self.db.find_course(course)
        have = self.db.count_sessions(cid) if cid else 0
        title, ok = QInputDialog.getText(
            self, "开始录制",
            f"这一节的标题（「{course}」已有 {have} 节）：",
            text=f"第 {have + 1} 节")
        if not ok:
            return
        try:
            sid = self.engine.start_recording(course.strip(), (title or "未命名").strip())
        except Exception as e:
            QMessageBox.critical(self, "无法开始录制",
                                 f"{e}\n\n提示：确认系统有正在播放的输出设备；"
                                 f"蓝牙耳机需要先连上再启动。")
            return
        self.current_session = sid
        self._live_count = 0
        self._record_started_at = time.time()
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_rec.setText("● 录制中")
        self.lbl_rec.setStyleSheet(f"color:{theme.BAD};")
        self.status.showMessage("录制已开始：现在去浏览器播放那节课。Ctrl+M 可手动标记回看点。")

    def _on_stop(self) -> None:
        if not self.engine.recording:
            return
        sid, path, dur = self.engine.stop_recording()
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_rec.setText("未录制")
        self.lbl_rec.setStyleSheet(f"color:{theme.TEXT_3};")
        self._refresh_sessions()
        self.status.showMessage(f"录制结束：{_mmss(dur)}，开始转写与分析…")
        self._run_pipeline(sid)

    def _on_mark(self) -> None:
        if self.engine.recording:
            self.engine.mark_manual()
            self._live_count += 1
            self.status.showMessage(f"已标记（本次录制第 {self._live_count} 个手动标记）", 3000)
            return
        # 没有在录制：允许事后给当前会话补一个"我没听懂"的时间点。
        # 这是零录制用户唯一的行为信号来源，所以不能只是忽略。
        if not self.current_session:
            self.status.showMessage("先选一个会话，或开始录制后再标记", 3000)
            return
        from PySide6.QtWidgets import QInputDialog
        row = self.db.get_session(self.current_session)
        total = float(row["duration"] or 0)
        txt, ok = QInputDialog.getText(
            self, "手动标记回看点",
            f"这一节总长 {_mmss(total)}。填你没听懂的时间点（mm:ss 或秒数），"
            f"可加说明，例如「12:30 梯度为零那里」：",
            text="")
        if not ok or not txt.strip():
            return
        import re
        m = re.match(r"^\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{2})\s*(.*)$", txt.strip())
        t = note = None
        if m:
            h = int(m.group(1) or 0)
            t = h * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            note = (m.group(4) or "").strip()
        else:
            m2 = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(.*)$", txt.strip())
            if m2:
                t = float(m2.group(1))
                note = (m2.group(2) or "").strip()
        if t is None:
            QMessageBox.warning(self, "格式不对", "请填类似 12:30 或 750 的时间。")
            return
        if total and t > total + 1:
            QMessageBox.warning(self, "超出时长", f"这个时间点超过了本节时长 {_mmss(total)}。")
            return
        self.engine.add_manual_mark(self.current_session, t, note)
        self._run_analyze(self.current_session, prefix=f"已在 {_mmss(t)} 补了一个标记")

    def _run_analyze(self, sid: str, prefix: str = "") -> None:
        def job():
            ana = self.engine.analyze_session(sid)
            return {"knots": len(ana.knots), "windows": len(ana.windows)}

        self.btn_analysis.setEnabled(False)
        w = Worker(job)
        w.done.connect(lambda r: self._analyze_done(sid, r, prefix))
        w.failed.connect(lambda e: self._pipeline_failed(e))
        self._spawn(w)

    def _analyze_done(self, sid: str, r: dict, prefix: str) -> None:
        self.btn_analysis.setEnabled(True)
        self.current_session = sid
        self._refresh_sessions()
        self._load_session_view()
        self._fill_reword_knots()
        self.tabs.setCurrentIndex(0)
        msg = f"{prefix}｜{r['windows']} 个分析窗口，{r['knots']} 个回看点"
        self.status.showMessage(msg, 8000)

    def _run_pipeline(self, sid: str, auto: bool = False) -> None:
        """跑转写 + 分析。auto=True 表示这是自动抓取起头的，失败不许弹框。

        为什么自动的要静音失败：用户此刻正看着视频，屏幕上突然盖一个"分析失败"
        的模态框，等于把自动化的代价还给他。自动抓取的产物是"顺手多出来的一节"，
        没成就算了——状态栏说一句、会话行标个错，人想看再点一下就行。
        """
        def job():
            res, note = self.engine.transcribe(sid)
            ana = self.engine.analyze_session(sid)
            return {"asr": res.backend, "real": res.is_real, "note": note,
                    "knots": len(ana.knots), "windows": len(ana.windows)}

        self.btn_analysis.setEnabled(False)
        w = Worker(job)
        w.done.connect(lambda r: self._pipeline_done(sid, r))
        w.failed.connect(lambda e: self._pipeline_failed(e, sid=sid, auto=auto))
        self._spawn(w)

    def _pipeline_done(self, sid: str, r: dict) -> None:
        self.btn_analysis.setEnabled(True)
        # 归档进知识库：同一门课里，第 5 讲的问题能检索到第 2 讲的原话（带时间码）
        try:
            if self.engine.kb_ingest_session(sid).get("ok"):
                self._refresh_kb()
                self._refresh_kb_auto_info()
        except Exception:                        # noqa: BLE001
            pass
        self.current_session = sid
        self._refresh_sessions()
        self._load_session_view()
        self._fill_reword_knots()
        flag = "" if r["real"] else "（注意：演示后端，文本为合成内容）"
        self.status.showMessage(
            f"分析完成：{r['knots']} 个回看点 / {r['windows']} 个窗口｜转写 {r['asr']}{flag}", 12000)
        if not r["real"]:
            QMessageBox.information(
                self, "当前是离线演示转写",
                "没有可用的真实转写后端，本次使用了**演示后端**：文本是合成的，"
                "用于验证链路与界面，不代表你刚才那节课的内容。\n\n"
                "要接真实转写：设置页填一个 OpenAI 兼容的转写端点（云端 paraformer/whisper，"
                "或本地 FunASR/whisper 服务），或者把 .srt 字幕放在音频同目录。")

    def _pipeline_failed(self, err: str, sid: str = "", auto: bool = False) -> None:
        self.btn_analysis.setEnabled(True)
        if auto:
            self._refresh_sessions()
            self.status.showMessage(
                f"自动抓取的这一节没转写出来（{err[:60]}）—行还留着，想看就手动点分析", 15000)
            return
        QMessageBox.critical(self, "分析失败", err)

    def _on_analyze(self) -> None:
        if not self.current_session:
            QMessageBox.information(self, "提示", "先选一个会话")
            return
        sid = self.current_session

        def job():
            ana = self.engine.analyze_session(sid)
            return {"knots": len(ana.knots), "windows": len(ana.windows)}

        self.btn_analysis.setEnabled(False)
        w = Worker(job)
        w.done.connect(lambda r: (self.btn_analysis.setEnabled(True),
                                  self._load_session_view(), self._fill_reword_knots(),
                                  self.status.showMessage(
                                      f"重新分析完成：{r['knots']} 个回看点", 8000)))
        w.failed.connect(self._pipeline_failed)
        self._spawn(w)

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择课程音视频 / 字幕 / 文本", "",
            "全部支持的格式 (*.mp4 *.mkv *.flv *.webm *.mov *.avi *.mp3 *.m4a *.wav *.flac *.aac *.srt *.vtt *.txt *.md);;"
            "媒体文件 (*.mp4 *.mkv *.flv *.webm *.mov *.avi *.mp3 *.m4a *.wav *.flac *.aac);;"
            "字幕/文本 (*.srt *.vtt *.txt *.md)")
        if not path:
            return
        src = Path(path)
        if src.suffix.lower() in (".srt", ".vtt", ".txt", ".md"):
            self._import_transcript(src)
            return
        course = self._ask_course_name(f"导入 {src.name} · 选课程",
                                       self._current_course_title())
        if not course:
            return
        self.status.showMessage("正在提取音轨…")
        sid = None
        try:
            ensure_dirs()
            if src.suffix.lower() in (".srt", ".vtt"):
                wav = None
                cid = self.db.ensure_course(course)
                sid = self.db.create_session(cid, src.stem, source="file")
            else:
                wav = AUDIO_DIR / f"{int(time.time())}_{src.stem[:20]}.wav"
                cap.import_media(src, wav)
                cid = self.db.ensure_course(course)
                sid = self.db.create_session(cid, src.stem, str(wav), source="file")
                self.db.update_session(sid, duration=cap.audio_duration(wav))
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"{e}")
            return
        self.current_session = sid
        self._run_pipeline(sid)

    def _on_screen_capture(self) -> None:
        if not self.current_session:
            QMessageBox.information(self, "提示", "先开始录制或选一个会话，再抓屏幕。")
            return
        from PySide6.QtWidgets import QInputDialog
        from dsh_app import screen as scr

        t_now = None
        if self.engine.recording and self.engine.recorder:
            t_now = float(self.engine.recorder.duration)
        if t_now is None:
            row = self.db.get_session(self.current_session)
            total = float(row["duration"] or 0)
            txt, ok = QInputDialog.getText(
                self, "这一帧对应的时间点",
                f"当前没有在录制。请填这一屏内容对应的时间点（mm:ss），"
                f"本节总长 {_mmss(total)}：", text="")
            if not ok or not txt.strip():
                return
            import re
            m = re.match(r"^\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{2})\s*$", txt.strip())
            if m:
                t_now = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            else:
                try:
                    t_now = float(txt.strip())
                except ValueError:
                    QMessageBox.warning(self, "格式不对", "请填 12:30 这样的时间。")
                    return

        try:
            frame = scr.grab_primary_screen()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "抓屏失败", f"{e}")
            return
        ext = scr.build_extractor(self.settings)
        okv, why = ext.available()
        if not okv:
            QMessageBox.warning(
                self, "视觉模型不可用",
                f"{why}\n\n请到「设置与模型部署」填 api_key 和视觉模型名"
                f"（例如 qwen-vl-max），保存后再试。")
            return
        self.status.showMessage(f"正在用 {ext.model} 识别屏幕…")

        def job():
            return ext.extract(frame)

        w = Worker(job)
        w.done.connect(lambda res: self._screen_done(res, t_now))
        w.failed.connect(lambda e: QMessageBox.critical(self, "识别失败", f"{e}"))
        self._spawn(w)

    def _screen_done(self, res, t_now: float) -> None:
        from PySide6.QtWidgets import QInputDialog
        if getattr(res, "error", ""):
            QMessageBox.critical(self, "识别失败", f"{res.error}")
            return
        text = (res.text or "").strip()
        if not text:
            self.status.showMessage(
                "这一屏没有识别到课程文字（可能是纯视频画面）；需要的话可手动补一条", 8000)
            return
        # 模型会认错字，给用户改的机会再入库
        edited, ok = QInputDialog.getMultiLineText(
            self, f"识别结果（{res.model}，{res.latency_s:.1f}s）",
            f"时间点 {_mmss(t_now)}｜{res.kind}｜可直接修改后确定：", text)
        if not ok:
            return
        try:
            info = self.engine.capture_screen_note(
                self.current_session, text=edited.strip(), kind=res.kind,
                t=t_now, source="vision")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "入库失败", f"{e}")
            return
        if not info.get("ok"):
            self.status.showMessage(info.get("reason", "没入库"), 6000)
            return
        self._run_analyze(self.current_session,
                          prefix=f"已把 {_mmss(t_now)} 的屏幕内容加入分析（{info['chars']} 字）")

    def _on_visual_evidence(self) -> None:
        sid = self.current_session
        if not sid:
            QMessageBox.information(self, "提示", "先选一个会话。")
            return
        if not self.engine.session_url_ref(sid):
            QMessageBox.information(
                self, "取不到画面",
                "只有「从链接导入」的 B站会话能取画面（用故事板，不下载视频）。"
                "录制或本地文件导入的会话没有画面来源。")
            return
        if not self.current_knots:
            QMessageBox.information(self, "提示", "这一节还没有回看点，先做一次分析。")
            return
        sess = getattr(getattr(self.settings, "source", None), "bili_sessdata", "")
        self.btn_vision.setEnabled(False)
        self.status.showMessage(f"正在取 {min(len(self.current_knots), 6)} 个回看点的画面…")

        def job():
            return self.engine.attach_visual_evidence(sid, sess, max_knots=6)

        def done(r):
            self.btn_vision.setEnabled(True)
            if not r.get("ok"):
                self.status.showMessage(r.get("reason", "没取到画面"), 8000)
                return
            lines = [f"共试 {r['tried']} 个回看点，{r['saved']} 个识别到文字：", ""]
            for x in r["results"]:
                if x.get("ok"):
                    lines.append(f"{_mmss(x['t'])}（{x['kind']}，{x['chars']} 字）："
                                 f"{x['text'][:70]}")
                else:
                    lines.append(f"{_mmss(x['t'])}：失败 —— {x.get('reason', '')}")
            QMessageBox.information(self, "画面取证结果",
                                    "\n".join(lines))
            self._run_analyze(sid, prefix=f"已补 {r['saved']} 处画面证据")

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: (self.btn_vision.setEnabled(True),
                                    QMessageBox.critical(self, "取画面失败", e)))
        self._spawn(w)

    def _on_import_url(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self, "从链接导入",
            "粘贴视频链接（B站 / YouTube 等），也可以只贴 BV 号：\n"
            "有字幕就只取字幕，不下载视频画面。", text="")
        url = (text or "").strip()
        if not ok or not url:
            return
        sess = getattr(getattr(self.settings, "source", None), "bili_sessdata", "")
        self.status.showMessage("正在探测链接…")

        def job():
            return self.engine.probe_url(url, sess)

        def done(p):
            if not p.get("ok"):
                QMessageBox.warning(self, "探测失败", p.get("error") or "未知原因")
                self.status.clearMessage()
                return
            self._url_probe_done(p, url, sess)

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: QMessageBox.critical(self, "探测失败", e))
        self._spawn(w)

    def _url_probe_done(self, p: dict, url: str, sess: str) -> None:
        mins = float(p.get("duration") or 0) / 60
        lines = [f"标题：{p.get('title') or '（未知）'}",
                 f"时长：{mins:.0f} 分钟" + (f"｜分 {p['n_pages']} P" if p.get("n_pages", 1) > 1 else "")]
        if p.get("has_subtitle"):
            langs = "、".join(p.get("langs") or []) or "未知语言"
            lines.append(f"字幕：有（{langs}）→ 只取字幕，不下载视频、不需要转写模型")
        else:
            lines.append("字幕：没有")
            if p.get("note"):
                lines.append(f"说明：{p['note']}")
            mb = p.get("audio_mb_estimate") or 0
            lines.append(f"替代方案：只下载音轨（约 {mb:.1f} MB）后转写，需要可用的转写后端。")
        lines.append("")
        lines.append("现在导入吗？")
        if QMessageBox.question(self, "确认导入", "\n".join(lines)) != QMessageBox.Yes:
            self.status.clearMessage()
            return
        self.status.showMessage("正在导入…")
        title = (p.get("title") or "").strip()[:60]
        # 归到哪门课要问一次：以前没选会话时静默落到「链接课程」，
        # 同一门课换个入口进来就变成另一门了
        course = self._ask_course_name(
            "从链接导入 · 选课程",
            self._current_course_title() or self._recent_course_title())
        if not course:
            self.status.clearMessage()
            return

        def job():
            return self.engine.import_url(url, course, title, sessdata=sess,
                                          progress=lambda m: None)

        def done(res):
            self._refresh_sessions()
            self.current_session = res["session_id"]
            self.knot_table.setRowCount(0)
            self._run_analyze(res["session_id"],
                              prefix=f"已从链接导入 {res['segments']} 段"
                                     f"（{res['source']}，{_mmss(res['duration'])}）")

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: QMessageBox.critical(self, "导入失败", e))
        self._spawn(w)

    def _current_course_title(self) -> str:
        if not self.current_session:
            return ""
        row = self.db.get_session(self.current_session)
        if not row:
            return ""
        c = self.db.get_course(str(row["course_id"]))
        return str(c["title"]) if c else ""

    # -- 课程名：可编辑下拉，默认带最近用过的那门 ------------------------
    def _course_rows(self) -> list:
        try:
            return list(self.db.list_courses())
        except Exception:                          # noqa: BLE001
            return []

    def _recent_course_title(self) -> str:
        rows = self._course_rows()
        return str(rows[0]["title"]) if rows else ""

    def _ask_course_name(self, caption: str, default: str = "") -> str | None:
        """问课程名：已有课程在下拉里，也能直接敲新的。

        返回去掉首尾空白的课名；取消或留空返回 None，调用方直接放弃本次动作。
        名字的"是不是同一门课"由 DB.title_key 判（空格/全半角括号/【】前缀会折叠）。
        """
        rows = self._course_rows()
        dlg = QDialog(self)
        dlg.setWindowTitle(caption)
        dlg.setMinimumWidth(480)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("课程名（选一个已有的，或直接敲新的）：", dlg))
        cb = QComboBox(dlg)
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.NoInsert)
        cb.setMinimumWidth(420)
        mapping: dict[str, str] = {}
        for r in rows:
            name = str(r["title"])
            n = int(r["n_sessions"] or 0)
            disp = f"{name}　—　{n} 节" if n else f"{name}　—　还没有节"
            cb.addItem(disp, name)
            mapping[disp] = name
        pick = (default or self._recent_course_title()).strip()
        if pick:
            cb.setCurrentText(pick)
        lay.addWidget(cb)
        hint = QLabel("不用每次打全名：多打空格、全半角括号、「【完整版】」这类前缀"
                      "都会自动归到同一门课。", dlg)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_3};")
        lay.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        # Qt 默认按钮文案是英文，界面其余部分全是中文，这里显式改掉
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.button(QDialogButtonBox.Ok).setDefault(True)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        cb.lineEdit().selectAll()
        if dlg.exec() != QDialog.Accepted:
            return None
        text = cb.currentText().strip()
        name = (mapping.get(text) or text).strip()
        return name or None

    def _import_transcript(self, src: Path) -> None:
        """零录制：只用字幕/文本建立会话，不需要录任何东西。"""
        from PySide6.QtWidgets import QInputDialog
        course = self._ask_course_name(f"导入 {src.name} · 选课程",
                                       self._current_course_title())
        if not course:
            return
        title, ok = QInputDialog.getText(self, "导入字幕/文本", "这一节的标题：",
                                         text=src.stem)
        if not ok:
            return
        hint = None
        if src.suffix.lower() in (".txt", ".md"):
            mins, ok = QInputDialog.getDouble(
                self, "需要课程时长",
                "纯文本没有时间戳，请填这一节的时长（分钟）：\n"
                "时间轴会按字数比例估算，回看点时间只能作参考。",
                45.0, 1.0, 600.0, 1)
            if not ok:
                return
            hint = mins * 60.0
        try:
            info = self.engine.import_transcript(src, course.strip(),
                                                 (title or src.stem).strip(), hint)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", f"{e}")
            return
        self.current_session = info["session_id"]
        self._refresh_sessions()
        self._run_analyze(info["session_id"],
                          prefix=f"已导入 {info['segments']} 段（{_mmss(info['duration'])}）"
                                 + ("，时间轴为估算值" if info["estimated_timeline"] else ""))

    def _on_demo(self) -> None:
        sid = self.engine.seed_demo()
        self.current_session = sid
        self._refresh_sessions()
        self._load_session_view()
        self._fill_reword_knots()
        self.tabs.setCurrentIndex(0)
        self.status.showMessage("已生成演示会话（转写文本为合成内容）", 8000)

    # ------------------------------------------------------------------
    # 重讲 / 反馈
    # ------------------------------------------------------------------
    def _on_export_notes(self) -> None:
        sid = self.current_session
        if not sid:
            QMessageBox.information(self, "提示", "先选一节课")
            return
        try:
            res = self.engine.export_notes(sid)
        except Exception as e:  # noqa: BLE001
            self.status.showMessage(f"导出失败：{e}", 10000)
            return
        if not res.get("ok"):
            self.status.showMessage(res.get("error") or "导出失败", 10000)
            return
        self.status.showMessage(
            f"已导出 {res['chars']} 字到 {res['path']}（{res['knots']} 个回看点）", 15000)
        QMessageBox.information(
            self, "笔记已导出",
            f"已写入：\n{res['path']}\n\n共 {res['chars']} 字，{res['knots']} 个回看点。\n"
            "移动或改名这一份不会影响软件里的数据。")
    def _on_ask_current_knot(self) -> None:
        """把当前选中的回看点整理成一句问题，切到对话页。"""
        if not self.current_knots:
            QMessageBox.information(self, "提示", "还没有回看点，先做一次分析。")
            return
        idx = self.reword_knot.currentIndex()
        if idx < 0 or idx >= len(self.current_knots):
            return
        k = self.current_knots[idx]
        t = float(k.t_start)
        nxt = float(k.t_end)
        body = ""
        try:
            segs = self.db.get_segments(self.current_session)
            body = " ".join(str(r["text"]) for r in segs
                            if float(r["t_end"]) > t and float(r["t_start"]) < nxt)[:600]
        except Exception:  # noqa: BLE001
            body = ""
        q = (f"第 {_mmss(t)}–{_mmss(nxt)} 这一段我没跟上。\n"
             f"判定依据：{k.evidence or '（无）'}\n")
        if body.strip():
            q += f"这一段的内容是：「{body.strip()}」\n"
        q += "请用我容易理解的方式讲清楚，并指出我可能需要补的前置知识。"
        self.tabs.setCurrentIndex(3)
        self.ed_ask.setPlainText(q)
        self.ed_ask.setFocus()

    def _on_reword(self) -> None:
        kid = self.reword_knot.currentData()
        if kid is None:
            QMessageBox.information(self, "提示", "这个会话还没有回看点")
            return
        key = self.reword_strategy.currentData()
        self.reword_out.setPlainText("正在生成…（本地模型首次调用可能需要几秒）")
        w = Worker(self.engine.reword, kid, key)
        w.done.connect(self._reword_done)
        w.failed.connect(lambda e: self.reword_out.setPlainText(f"生成失败：{e}"))
        self._spawn(w)

    def _reword_done(self, r) -> None:
        head = f"【{r.strategy_name}】"
        if r.degraded:
            head += "（未接入生成模型，以下为结构化提纲）"
        self.reword_out.setPlainText(f"{head}\n\n{r.text}")
        if r.error:
            self.status.showMessage(f"降级原因：{r.error[:160]}", 12000)
        self._load_session_view()

    def _feedback(self, action: str) -> None:
        kid = self.reword_knot.currentData()
        if kid is None:
            return
        res = self.engine.record_feedback(int(kid), action)
        self._load_session_view()
        try:
            self.refresh_kp_table()
        except Exception:  # noqa: BLE001
            pass
        extra = ""
        if res.get("kp"):
            names = "、".join(f"{k['name']} {int(k['mastery'] * 100)}%"
                             for k in res["kp"][:3])
            extra = f"；知识点已更新：{names}"
        self.status.showMessage(f"已记录：{action}{extra}", 6000)

    # ------------------------------------------------------------------
    # 漏洞地图
    # ------------------------------------------------------------------
    def _on_map(self) -> None:
        if not self.current_session:
            return
        self.refresh_kp_table()
        d = self.engine.diagnosis(self.current_session)
        self.map_view.setHtml(_diagnosis_html(d))

    def _on_rebuild_knowledge(self) -> None:
        """重建知识点：术语抽取规则变了，不该要求用户重听一遍音频。"""
        cid = self._current_course_id()
        if not cid:
            self.status.showMessage("先选一门课", 6000)
            return
        self.status.showMessage("正在按现存转写重算知识点…")
        w = Worker(lambda: self.engine.rebuild_knowledge(cid))
        w.done.connect(self._rebuild_knowledge_done)
        w.failed.connect(lambda e: self.status.showMessage(f"重建失败：{e}", 12000))
        self._spawn(w)

    def _rebuild_knowledge_done(self, res: dict) -> None:
        rows = (res.get("courses") or [{}])[0]
        self.refresh_kp_table()
        if self.current_session:
            self._load_session_view()
        msg = (f"知识点已重算：{len(rows.get('remaining') or [])} 个"
               f"；清掉没验证也没挂回看点的旧知识点 {len(rows.get('pruned') or [])} 个")
        self.status.showMessage(msg, 15000)

    def _on_calibrate(self) -> None:
        r = self.engine.recalibrate()
        self.status.showMessage(
            f"校准完成：{r['samples']} 条样本，权重 "
            + ", ".join(f"{k}={v:.3f}" for k, v in r["weights"].items()), 12000)
        QMessageBox.information(
            self, "权重校准（方案 5.2.6）",
            f"参与训练的样本：{r['samples']} 条\n"
            f"当前权重：\n" + "\n".join(f"  {k} = {v:.4f}" for k, v in r["weights"].items())
            + f"\n  bias = {r['bias']:.4f}\n\n"
            "样本 < 20 时学习率减半，> 100 后按规模衰减；没有任何反馈时不会改动权重。")

    # ------------------------------------------------------------------
    # 探测 / 部署
    # ------------------------------------------------------------------
    def _on_probe(self) -> None:
        self._collect_settings()
        self.settings.save()
        self.engine.llm = build_llm(self.settings)
        self.deploy_view.setPlainText("正在探测…")
        w = Worker(self.engine.llm.probe, True)
        w.done.connect(self._probe_done)
        w.failed.connect(lambda e: self.deploy_view.setPlainText(f"探测失败：{e}"))
        self._spawn(w)

    def _probe_done(self, res) -> None:
        lines = [
            f"端点：{res.base_url}",
            f"结论：{res.summary()}",
            "",
        ]
        if res.models:
            lines.append("可用模型（前 30 个）：")
            lines += [f"  - {m}" for m in res.models[:30]]
        if res.error:
            lines += ["", f"错误：{res.error}"]
        lines += ["", "说明：能力探测只验证「可达 / 有模型 / 能对话」，不验证质量。"
                      "不支持的能力会在界面上标注为「当前后端不支持」，而不是静默失败。"]
        self.deploy_view.setPlainText("\n".join(lines))
        # 探测顺手拿到真模型列表，就直接填进下拉，省得用户再点一次「刷新列表」
        if res.models:
            p = provider(self.cmb_preset.currentData())
            cur = self.cmb_llm_model.currentText().strip()
            self._fill_models(sorted(res.models), cur)
            vision = vision_of(res.models, p.get("vision_models"))
            self._fill_vision(vision or sorted(res.models),
                              self.cmb_vision.currentText().strip())
            self.lbl_llm_status.setText(
                f"✓ {len(res.models)} 个模型可选"
                + (f"（{res.latency_ms:.0f} ms）" if res.latency_ms else "")
                + ("｜对话正常" if res.chat_ok else "｜对话未验证"))
            self.lbl_llm_status.setStyleSheet(f"color:{theme.OK}; font-size:12px;")
        self._update_backend_badge()

    # -- 本地转写（Whisper） --------------------------------------------
    def _whisper_size(self) -> str:
        return self.cmb_whisper.currentData() or DEFAULT_WHISPER_SIZE

    def _refresh_whisper(self) -> None:
        st = whisper_status()
        lines = [f"依赖：{'可用' if st['deps_ok'] else '缺失'} —— {st['deps']}"]
        have = [m for m in st["models"] if m["installed"]]
        if have:
            lines.append("已装模型：" + "、".join(f"{m['key']}（{m['mb']:.0f} MB）"
                                                for m in have))
        else:
            lines.append("已装模型：无（挑一个尺寸点「下载模型」）")
        lines.append(f"模型目录：{st['models_dir']}")
        lines.append("结论：" + ("可以离线转写" if st["ready"] else "还不能转写"))
        self.lbl_whisper.setText("\n".join(lines))

    def _on_install_whisper_deps(self) -> None:
        self.btn_w_deps.setEnabled(False)
        self.lbl_whisper.setText("正在安装 faster-whisper…（走阿里云镜像）")

        def job():
            return install_whisper_deps()

        def done(r):
            ok, log = r
            self.btn_w_deps.setEnabled(True)
            self.lbl_whisper.setText(("安装完成" if ok else "安装失败")
                                     + "：\n" + (log or "")[-600:])
            self._refresh_whisper()

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: (self.btn_w_deps.setEnabled(True),
                                    self.lbl_whisper.setText(f"安装失败：{e}")))
        self._spawn(w)

    def _on_download_whisper(self) -> None:
        size = self._whisper_size()
        self.btn_w_dl.setEnabled(False)

        def job():
            msgs = []
            ok, msg = download_whisper(size, progress=lambda m: msgs.append(m))
            return ok, msg, msgs[-1] if msgs else ""

        def done(r):
            ok, msg, last = r
            self.btn_w_dl.setEnabled(True)
            self.lbl_whisper.setText(("完成" if ok else "失败") + f"：{msg}")
            self._refresh_whisper()

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: (self.btn_w_dl.setEnabled(True),
                                    self.lbl_whisper.setText(f"下载失败：{e}")))
        self._spawn(w)

    def _on_test_whisper(self) -> None:
        audio = self._find_test_audio()
        if not audio:
            QMessageBox.information(
                self, "没有可测试的音频",
                "先用「导入视频/音频…」或「从链接导入…」拿到一段音频，再来测试。\n"
                "（转写只吃音频文件，不需要视频画面）")
            return
        size = self._whisper_size()
        self.btn_w_test.setEnabled(False)
        self.lbl_whisper.setText(f"正在用 {size} 转写：{Path(audio).name}…")

        def job():
            return transcribe_audio(audio, size)

        def done(r):
            self.btn_w_test.setEnabled(True)
            if not r.get("ok"):
                self.lbl_whisper.setText(f"转写失败：{r.get('error')}")
                return
            rows = r["segments"][:5]
            lines = [f"成功：{r['n']} 段｜{r['media_seconds']:.0f}s 音频用了 "
                     f"{r['seconds']:.1f}s（{r['speed']:.1f}× 实时）｜{r['device']}",
                     ""]
            lines += [f"{_mmss(x['t_start'])} {x['text'][:60]}" for x in rows]
            self.lbl_whisper.setText("\n".join(lines))

        w = Worker(job)
        w.done.connect(done)
        w.failed.connect(lambda e: (self.btn_w_test.setEnabled(True),
                                    self.lbl_whisper.setText(f"转写失败：{e}")))
        self._spawn(w)

    def _find_test_audio(self) -> str:
        """找一个能测的音频：当前会话的音频 → data/media 里最新的一个。"""
        sid = self.current_session
        if sid:
            row = self.db.get_session(sid)
            if row and row["audio_path"] and Path(str(row["audio_path"])).exists():
                suf = Path(str(row["audio_path"])).suffix.lower()
                if suf not in (".srt", ".vtt", ".txt", ".md"):
                    return str(row["audio_path"])
        media = Path("data") / "media"
        if media.exists():
            files = [p for p in media.iterdir()
                     if p.is_file() and p.suffix.lower() in
                     (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".mp4")]
            if files:
                return str(max(files, key=lambda p: p.stat().st_mtime))
        return ""

    def _on_detect(self) -> None:
        self.deploy_view.setPlainText("正在检测本机算力与模型…")
        w = Worker(detect_all)
        w.done.connect(self._detect_done)
        w.failed.connect(lambda e: self.deploy_view.setPlainText(f"检测失败：{e}"))
        self._spawn(w)

    def _detect_done(self, d: dict) -> None:
        self.cmb_model.clear()
        for m in d["models"]:
            self.cmb_model.addItem(f"{m['name']}（{m['state']}｜:{m['port']}）", m["key"])
        gpu = d["gpu"]
        gtxt = ("未检测到 NVIDIA GPU" if not gpu["available"] else
                "、".join(f"{g['name']} {g['total_mb'] / 1024:.0f}GB" for g in gpu["gpus"]))
        lines = [
            "== 算力 ==",
            f"GPU：{gtxt}",
            f"内存：{d['mem']['total_gb']} GB（可用 {d['mem']['available_gb']} GB）",
            f"可用容量估计：{d['capacity_gb']} GB",
            "",
            "== 建议 ==",
            f"后端：{d['recommendation']['backend']}｜{d['recommendation']['model']}",
            d["recommendation"]["reason"],
            "",
            "== 方案 4.4 的模型清单 ==",
        ]
        for m in d["models"]:
            head = f"[{m['state']}] {m['name']}　{m['plan_ref']}　约 {m['size_gb']} GB"
            lines.append(head)
            if m["serving"]:
                lines.append(f"    端点：{m['endpoint']}（模型 {', '.join(m['models'][:3])}）")
            if m["downloaded"]:
                lines.append(f"    权重：{m['download_path']}")
            if m["note"]:
                lines.append(f"    备注：{m['note']}")
        lines += ["", "提示：点「生成部署脚本」先看脚本内容，确认后再「执行部署并启动」。"
                      "下载走 ModelScope，脚本会把权重落到 data/models/<模型>。"]
        self.deploy_view.setPlainText("\n".join(lines))

    def _on_gen_script(self) -> None:
        key = self.cmb_model.currentData()
        if not key:
            QMessageBox.information(self, "提示", "先点「检测本机模型与算力」")
            return
        r = one_click_deploy(key, dry_run=True)
        if not r.get("ok"):
            self.deploy_view.setPlainText(str(r.get("error")))
            return
        self.deploy_view.setPlainText(f"脚本已生成：{r['script']}\n\n{r['text']}")
        self.status.showMessage(f"已写出 {r['script']}", 6000)

    def _on_launch(self) -> None:
        key = self.cmb_model.currentData()
        if not key:
            QMessageBox.information(self, "提示", "先点「检测本机模型与算力」")
            return
        if QMessageBox.question(
            self, "确认部署",
            f"将执行部署脚本（下载权重并启动 OpenAI 兼容服务）。\n"
            f"权重可能达到数十 GB，确认继续？",
        ) != QMessageBox.Yes:
            return
        r = one_click_deploy(key, dry_run=False)
        if r.get("ok"):
            self.deploy_view.setPlainText(
                f"已在后台启动：{r['script']}\n日志：{r['log']}\n\n"
                f"服务起来后，把设置页的 base_url 改成 http://127.0.0.1:<端口>/v1，"
                f"模型名填 {key}，再点「探测能力」。")
        else:
            self.deploy_view.setPlainText(str(r.get("error")))

    # ------------------------------------------------------------------
    # 定时刷新
    # ------------------------------------------------------------------
    def _on_capture_level(self, level: float) -> None:
        """录音线程回调：只存数，不碰控件（跨线程改 Widget 会崩）。

        真正的重画在 `_on_tick` 里做。这个数也正是"网页视频该不该停录"的
        判据来源（`Engine.is_capture_quiet`），所以它必须有值。
        """
        self._level = float(level)

    def _on_tick(self) -> None:
        if self.engine.recording and self.engine.recorder:
            self.level.setValue(int(min(1.0, self._level) * 1000))
            rec = self.engine.recorder
            dur = rec.duration
            if not rec.receiving and rec.idle_seconds > 5.0:
                # 蓝牙输出在没播放时不出数据：这时候必须明说，否则用户以为在录
                self.lbl_rec.setText(
                    f"● 录制中 0:00　收不到音频（{rec.idle_seconds:.0f}s）—— 请开始播放")
                self.lbl_rec.setStyleSheet(f"color:{theme.WARN};")
                self.status.showMessage(
                    f"当前设备「{rec.device_label}」没有音频流：请在浏览器里开始播放课程；"
                    f"如果一直收不到，去设置页换一个回环设备并点「试读」。", 6000)
            else:
                auto = "自动" if self._web_capture_active() else ""
                self.lbl_rec.setText(
                    f"● {auto}录制中 {_mmss(dur)}　手动标记 {self._live_count}")
                self.lbl_rec.setStyleSheet(f"color:{theme.BAD};")
            self.timeline.set_playhead(dur)
            try:
                live = self.engine.live_tick()
                if live:
                    self.status.showMessage(
                        f"实时分析：最近 {self.settings.cognitive.window_s:.0f}s 内有 "
                        f"{len(live)} 处疑似回看点（课后统一交付）", 4000)
            except Exception:
                pass

    def closeEvent(self, ev) -> None:
        try:
            if self.engine.recording:
                self.engine.stop_recording()
        except Exception:
            pass
        # 后台线程（预热/探测/分析）必须先收敛，否则 Qt 会在析构时报
        # "QThread: Destroyed while thread is still running" 并可能崩在退出路径上
        self.ui_timer.stop()
        # 监听线程必须先停：它会在后台起停录制，退出时不能让它继续跑
        try:
            self._stop_web_watcher("退出程序")
        except Exception:
            pass
        for w in list(self._workers):
            try:
                w.wait(5000)
            except Exception:
                pass
        self._workers.clear()
        super().closeEvent(ev)


# --------------------------------------------------------------------------
def _diagnosis_html(d: dict) -> str:
    if d.get("error"):
        return f"<p>{d['error']}</p>"
    s = d["session"]
    out = [f"<h2 style='color:{theme.ACCENT_PRESS}'>漏洞地图 · {s['title']}</h2>",
           f"<p style='color:{theme.TEXT_3}'>时长 {s['duration_text']}　回看点 {d['n_knots']} 个"
           f"（重点回看 {d['n_strong']} 个）</p>"]

    out.append(f"<h3 style='color:{theme.TEXT}'>一、本节回看点清单</h3>")
    if not d["knots"]:
        out.append(f"<p style='color:{theme.TEXT_3}'>本节没有产出回看点。</p>")
    else:
        out.append("<table cellspacing='0' cellpadding='6' width='100%' "
                   f"style='border:1px solid {theme.BORDER}'>")
        out.append(f"<tr style='background:{theme.SURFACE_ALT};color:{theme.TEXT_2}'>"
                   "<th align='left'>时间</th><th align='left'>等级</th><th align='left'>分数</th>"
                   "<th align='left'>知识点</th><th align='left'>成因 / 证据</th></tr>")
        for k in d["knots"]:
            color = f"{theme.BAD}" if k["grade"] == "strong" else f"{theme.WARN}"
            out.append(
                f"<tr><td>{k['t']}</td>"
                f"<td style='color:{color}'>{'重点回看' if k['grade'] == 'strong' else '中等'}</td>"
                f"<td>{k['score']:.2f}</td>"
                f"<td style='color:{theme.TEXT_2}'>{'、'.join(k['terms']) or '—'}</td>"
                f"<td>{k['evidence']}</td></tr>")
        out.append("</table>")

    out.append(f"<h3 style='color:{theme.TEXT}'>二、回看点热度在时间轴上的分布</h3>")
    if d["distribution"]:
        mx = max(x["count"] for x in d["distribution"])
        for b in d["distribution"]:
            n = int(28 * b["count"] / mx) if mx else 0
            out.append(f"<div style='color:{theme.TEXT_3}'>{b['label']}　"
                       f"<span style='color:{theme.ACCENT}'>{'■' * max(1, n)}</span> {b['count']}</div>")
    else:
        out.append(f"<p style='color:{theme.TEXT_3}'>无分布数据</p>")

    out.append(f"<h3 style='color:{theme.TEXT}'>三、与历史节次的关联</h3>")
    if d["related"]:
        out.append("<ul>")
        for r in d["related"][:15]:
            out.append(f"<li><b>{r['name']}</b> —— 第 {r['session_idx']} 节"
                       f"《{r['session']}》{r['t']} 也卡过（分数 {r['score']}）</li>")
        out.append("</ul>")
    else:
        out.append(f"<p style='color:{theme.TEXT_3}'>本课程其他节次中还没有同类回看点。</p>")

    out.append(f"<h3 style='color:{theme.TEXT}'>四、建议的复习顺序（按依赖，不按时间）</h3>")
    if d["plan"]:
        out.append("<ol>")
        for p in d["plan"][:20]:
            st = {"pending": "待解决", "explained": "已重讲",
                  "mastered": "已掌握"}.get(p["status"], p["status"])
            out.append(f"<li>{p['t']}　<b>{'、'.join(p['terms']) or '未命名知识点'}</b>"
                       f"　<span style='color:{theme.TEXT_3}'>{st}｜{p['evidence']}</span></li>")
        out.append("</ol>")
    else:
        out.append(f"<p style='color:{theme.TEXT_3}'>无</p>")

    if d["root_causes"]:
        out.append(f"<h3 style='color:{theme.TEXT}'>五、结构性根因（跨节反复出现）</h3><ul>")
        for r in d["root_causes"]:
            out.append(f"<li><b>{r['name']}</b>　在 {r['sessions']} 节中出现，"
                       f"共 {r['knots']} 次，平均分数 {r['avg_score']}</li>")
        out.append("</ul>")
    return "".join(out)


def launch_app() -> int:
    ensure_dirs()
    settings = Settings.load()
    db = DB()
    app = QApplication.instance() or QApplication([])
    theme.apply_font(app)
    app.setStyleSheet(STYLE)
    win = MainWindow(settings, db)
    win.show()
    return app.exec()