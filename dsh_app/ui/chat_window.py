"""问模型小窗：一个能一直浮在网课页面旁边的小对话窗。

为什么要独立窗口（而不是只在页签里问）：
听课的时候是「一边放视频一边问」，主窗口一最大化就把视频盖住了。小窗可以
置顶、可以缩到只够看两轮对话，问完关掉不打扰。它和主窗口共用一个消息列表，
所以窗口里问的、页签里也看得到（关掉窗口后主窗口会重画一遍）。

三个入口，都为"边看边问"服务：
1. 拖文件/截图进来 → 直接当附件（图片走视觉模型，文档抽成文本当资料）；
2. 复制截图后 Ctrl+V → 粘成附件（在看网课截屏问"这块什么意思"最常用）；
3. 只发附件不打字 → 也允许，等于"看看这个"。
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPoint, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSizePolicy, QTextBrowser, QVBoxLayout, QWidget,
)

from .. import attachments as att
from . import theme
from .chat_html import chat_html

UI_STATE = Path("data/ui_state.json")


class _Worker(QThread):
    """一次问答跑在后台：云端模型可能要十几秒，主线程不能等。"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self._fn, self._a, self._kw = fn, a, kw

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._a, **self._kw))
        except Exception as e:                        # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class ChatWindow(QWidget):
    """独立小窗。共用主窗口的 msgs 列表，关掉时回调 on_close 让主窗口刷新。"""

    def __init__(self, engine, settings, msgs: list[dict], session_id: str = "",
                 on_close=None, parent=None):
        super().__init__(parent, Qt.Window)
        self.engine = engine
        self.settings = settings
        self.msgs = msgs
        self.session_id = session_id
        self._on_close = on_close
        self._atts: list[att.Attachment] = []
        self._worker: _Worker | None = None
        self.setWindowTitle("问模型 · 小蟹陪你边看边问")
        from . import icons
        self.setWindowIcon(icons.app_icon())
        self.setAcceptDrops(True)
        self.setMinimumSize(420, 460)
        self._build_ui()
        self._restore_state()

    # ---------------------------------------------------------------- 界面
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(theme.GAP_M, theme.GAP_M, theme.GAP_M, theme.GAP_M)
        root.setSpacing(theme.GAP_S)

        head = QHBoxLayout()
        head.setSpacing(theme.GAP_XS)
        title = QLabel("问模型")
        title.setObjectName("h2")
        head.addWidget(title)
        self.chk_kb = QCheckBox("用知识库")
        self.chk_kb.setChecked(True)
        self.chk_kb.setToolTip("回答优先依据你导入的资料和本节回看点")
        head.addWidget(self.chk_kb)
        self.chk_ctx = QCheckBox("带本节上下文")
        self.chk_ctx.setChecked(True)
        head.addWidget(self.chk_ctx)
        head.addStretch(1)
        self.chk_top = QCheckBox("置顶")
        self.chk_top.setToolTip("一直浮在视频上面，不会被网课页面盖住")
        # 默认置顶：这个小窗就是为了"边放视频边问"存在的，默认被视频盖住就没意义了
        self.chk_top.setChecked(True)
        self.chk_top.toggled.connect(self._on_top)
        head.addWidget(self.chk_top)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.clicked.connect(self._on_clear)
        head.addWidget(self.btn_clear)
        root.addLayout(head)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.setHtml(chat_html(self.msgs, self.settings.llm.model))
        root.addWidget(self.view, 1)

        # 附件条：缩略图 + 名字 + ✕。没有附件时整条隐藏，不占地方。
        self.att_card = QFrame()
        self.att_card.setObjectName("card")
        self.att_row = QHBoxLayout(self.att_card)
        self.att_row.setContentsMargins(theme.GAP_S, theme.GAP_XS, theme.GAP_S, theme.GAP_XS)
        self.att_row.setSpacing(theme.GAP_XS)
        self.att_row.addStretch(1)
        self.att_card.setVisible(False)
        root.addWidget(self.att_card)

        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText(
            "问点什么…（Enter 发送，Shift+Enter 换行）\n"
            "截图直接 Ctrl+V 粘进来，或把图片/PDF 拖到窗口里")
        self.edit.setMaximumHeight(96)
        self.edit.installEventFilter(self)
        root.addWidget(self.edit)

        foot = QHBoxLayout()
        foot.setSpacing(theme.GAP_S)
        self.lbl_state = QLabel("")
        self.lbl_state.setObjectName("muted")
        foot.addWidget(self.lbl_state)
        foot.addStretch(1)
        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("primary")
        self.btn_send.clicked.connect(self._on_send)
        foot.addWidget(self.btn_send)
        root.addLayout(foot)

    # ---------------------------------------------------------------- 置顶/位置
    def _on_top(self, on: bool) -> None:
        flags = self.windowFlags()
        if on:
            self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
        self.show()                       # 改 flags 必须重新 show，否则窗口会消失

    def _state_path(self) -> Path:
        p = Path(UI_STATE)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _restore_state(self) -> None:
        try:
            d = json.loads(self._state_path().read_text(encoding="utf-8"))
            g = d.get("chat_window") or {}
            if g.get("w") and g.get("h"):
                self.resize(int(g["w"]), int(g["h"]))
            else:
                self.resize(520, 640)
            if g.get("x") is not None and g.get("y") is not None:
                self.move(QPoint(int(g["x"]), int(g["y"])))
            if "on_top" in g:
                self.chk_top.setChecked(bool(g["on_top"]))
        except Exception:                             # noqa: BLE001
            self.resize(520, 640)

    def _save_state(self) -> None:
        try:
            p = self._state_path()
            d = {}
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8")) or {}
            d["chat_window"] = {"x": self.x(), "y": self.y(),
                                "w": self.width(), "h": self.height(),
                                "on_top": self.chk_top.isChecked()}
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:                             # noqa: BLE001
            pass

    # ---------------------------------------------------------------- 附件
    def _refresh_atts(self) -> None:
        while self.att_row.count() > 1:                # 留住末尾那个 stretch
            it = self.att_row.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for i, a in enumerate(self._atts):
            self.att_row.insertWidget(self.att_row.count() - 1, self._att_chip(a, i))
        self.att_card.setVisible(bool(self._atts))

    def _att_chip(self, a: att.Attachment, idx: int) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        h = QHBoxLayout(box)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)
        if a.kind == "image" and a.raw:
            img = QImage.fromData(QByteArray(a.raw))
            if not img.isNull():
                pic = QLabel()
                pic.setObjectName("thumb")
                pic.setFixedSize(46, 46)
                pic.setAlignment(Qt.AlignCenter)
                # 截图大多是白底，直接贴在白卡片上看不出边界，所以给缩略图一圈描边
                pic.setPixmap(QPixmap.fromImage(img).scaled(
                    42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                h.addWidget(pic)
        name = a.name if len(a.name) <= 16 else a.name[:13] + "…"
        lab = QLabel(name)
        lab.setObjectName("muted")
        lab.setToolTip(a.path or a.name)
        h.addWidget(lab)
        if a.error:
            lab.setText(f"⚠ {name}")
            lab.setStyleSheet(f"color:{theme.BAD};")
            lab.setToolTip(a.error)
        x = QPushButton("✕")
        x.setObjectName("ghost")
        x.setFixedWidth(24)
        x.clicked.connect(lambda _=False, i=idx: self._remove_att(i))
        h.addWidget(x)
        return box

    def _remove_att(self, idx: int) -> None:
        if 0 <= idx < len(self._atts):
            self._atts.pop(idx)
            self._refresh_atts()

    def add_paths(self, paths: list[str]) -> None:
        got = att.load_paths(paths)
        self._atts.extend(got)
        self._refresh_atts()
        bad = [a for a in got if not a.ok]
        if bad:
            self.lbl_state.setText(f"{len(bad)} 个文件没收下：{bad[0].name}（{bad[0].error}）")

    def add_image(self, img: QImage, name: str = "剪贴板截图.png") -> None:
        if img.isNull():
            return
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        img.save(buf, "PNG")
        self._atts.append(att.image_attachment(bytes(ba), name=name))
        self._refresh_atts()
        self.lbl_state.setText("已粘贴一张截图，直接点发送即可")

    # ---------------------------------------------------------------- 拖放 / 粘贴
    def dragEnterEvent(self, e) -> None:             # noqa: N802
        md = e.mimeData()
        if md.hasUrls() or md.hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:                  # noqa: N802
        md = e.mimeData()
        if md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            if paths:
                self.add_paths(paths)
                e.acceptProposedAction()
                return
        if md.hasImage():
            img = QImage(md.imageData())
            if not img.isNull():
                self.add_image(img, "拖入的截图.png")
                e.acceptProposedAction()

    def eventFilter(self, obj, ev) -> bool:          # noqa: N802
        from PySide6.QtCore import QEvent

        if obj is self.edit and ev.type() == QEvent.KeyPress:
            key = ev.key()
            mods = ev.modifiers()
            # Ctrl+V：剪贴板里是图就收成附件；是纯文本就交回给输入框正常粘贴
            if key == Qt.Key_V and mods & Qt.ControlModifier:
                cb = QGuiApplication.clipboard()
                md = cb.mimeData()
                if md is not None and md.hasImage():
                    img = QImage(md.imageData())
                    if not img.isNull():
                        self.add_image(img)
                        return True
            # Enter 发送，Shift+Enter 换行
            if key in (Qt.Key_Return, Qt.Key_Enter) and not (mods & Qt.ShiftModifier):
                self._on_send()
                return True
        return super().eventFilter(obj, ev)

    # ---------------------------------------------------------------- 发送
    def _on_clear(self) -> None:
        self.msgs.clear()
        self._atts.clear()
        self._refresh_atts()
        self.view.setHtml(chat_html([], self.settings.llm.model))
        self.lbl_state.setText("")

    def _append(self, role: str, text: str, sources: list | None = None) -> None:
        self.msgs.append({"role": role, "content": text, "sources": sources or []})
        self.view.setHtml(chat_html(self.msgs, self.settings.llm.model))
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_send(self) -> None:
        if self._worker is not None:
            return
        q = self.edit.toPlainText().strip()
        images, files, bad = att.split(self._atts)
        if not q and not images and not files:
            self.lbl_state.setText("写点什么，或者粘一张图/拖个文件进来")
            return
        if not getattr(self.settings.llm, "configured", False):
            self.lbl_state.setText("还没配模型：设置页填 base_url 和 api_key")
            return

        label = q or "（只发了附件）"
        self._append("user", label + att.history_label(self._atts))
        self.edit.clear()
        self._atts = []
        self._refresh_atts()
        self.btn_send.setEnabled(False)
        self.lbl_state.setText("思考中…")

        data_urls = [a.data_url for a in images]
        pairs = [(a.name, a.text) for a in files]
        use_kb = self.chk_kb.isChecked()
        use_ctx = self.chk_ctx.isChecked()
        sid = self.session_id

        w = _Worker(self.engine.ask, q, session_id=sid, use_kb=use_kb,
                    use_context=use_ctx, images=data_urls, files=pairs)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker = w
        w.start()

    def _on_done(self, res: dict) -> None:
        self.btn_send.setEnabled(True)
        if res.get("error"):
            self.lbl_state.setText("失败")
            self._append("assistant", f"（调用失败：{res['error']}）")
            return
        self.lbl_state.setText("")
        self._append("assistant", res.get("answer", ""), res.get("sources"))

    def _on_failed(self, err: str) -> None:
        self.btn_send.setEnabled(True)
        self.lbl_state.setText("失败")
        self._append("assistant", f"（调用失败：{err}）")

    # ---------------------------------------------------------------- 关闭
    def closeEvent(self, e) -> None:                 # noqa: N802
        self._save_state()
        if callable(self._on_close):
            self._on_close()
        super().closeEvent(e)