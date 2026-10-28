"""「从 B 站导入课程」对话框：借用登录态，把整门课的分集清单拉进来。

两种拿到登录态的方式，界面上并排放，因为用户手上是什么状态不好猜：

1. **扫码登录**（主力）：在窗口里扫一下，登录态存到 `data/bili_session.json`，
   下次不用再扫。不碰浏览器、不碰密码。
2. **读浏览器登录**：Edge/Chrome 已经登录过 B 站时一键读 cookie。
   *但浏览器开着的时候 cookie 库是独占锁的*（Windows 上连只读都打不开），
   所以这条路只在浏览器关掉时可用——失败时界面会直说，并给「在浏览器登录」的退路。

导入只搬**元信息**：标题、时长、原始链接。视频留在 B 站，我们到点去听。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import bili
from . import theme


class _Job(QThread):
    """一个函数在后台跑，结果回到界面线程（网络请求不能堵 UI）。"""

    ok = Signal(object)
    bad = Signal(str)

    def __init__(self, fn: Callable[[], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:                       # noqa: D401
        try:
            self.ok.emit(self._fn())
        except Exception as e:                   # noqa: BLE001
            self.bad.emit(str(e))


class BiliImportDialog(QDialog):
    """返回值：`course_title` + `chosen`（每项含 title/url/duration/bvid/page）。"""

    def __init__(self, parent=None, transport=None):
        super().__init__(parent)
        # transport 只给自检用：注入假 HTTP 层就能离线把整条流程跑一遍
        self._transport = transport
        self.setWindowTitle("从 B 站导入课程")
        self.resize(820, 640)
        self.course_title = ""
        self.chosen: list[dict] = []
        self._cookies: dict[str, str] = {}
        self._qr_key = ""
        self._poll = QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(self._poll_qr)
        self._jobs: list[_Job] = []
        self._build()
        self._refresh_login()

    # -- 界面 ------------------------------------------------------------
    def _build(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(theme.GAP_L, theme.GAP_M, theme.GAP_L, theme.GAP_M)
        v.setSpacing(theme.GAP_S)

        # 状态和按钮分两行：挤在一行时第三个按钮会被裁掉一半，
        # 用户根本看不见"登录"在哪——入口被裁掉等于没有
        self.lbl_login = QLabel("正在读取登录状态…")
        self.lbl_login.setWordWrap(True)
        v.addWidget(self.lbl_login)
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_S)
        self.btn_qr = QPushButton("扫码登录")
        self.btn_qr.setObjectName("primary")
        self.btn_qr.clicked.connect(self._start_qr)
        self.btn_rebind = QPushButton("读浏览器登录")
        self.btn_rebind.setToolTip("Edge/Chrome 里已经登录过 B 站时，直接借用那份登录态。\n"
                                   "浏览器开着的时候 cookie 库是独占锁，读不到（会提示）")
        self.btn_rebind.clicked.connect(self._read_browser)
        self.btn_browser = QPushButton("在浏览器登录")
        self.btn_browser.setToolTip("跳转到 B 站登录页；登完回来点「读浏览器登录」")
        self.btn_browser.clicked.connect(lambda: bili.open_browser_login())
        for b in (self.btn_qr, self.btn_rebind, self.btn_browser):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        # 兜底：扫码这条路万一被风控/网络卡住，手填一次 SESSDATA 一定能通
        manual = QHBoxLayout()
        manual.setSpacing(theme.GAP_S)
        manual.addWidget(QLabel("或粘 SESSDATA"))
        self.ed_sessdata = QLineEdit()
        self.ed_sessdata.setPlaceholderText(
            "浏览器登录 B 站后 F12 → Application → Cookies → bilibili.com → SESSDATA 的值")
        manual.addWidget(self.ed_sessdata, 1)
        self.btn_manual = QPushButton("用这个登录")
        self.btn_manual.clicked.connect(self._use_manual)
        manual.addWidget(self.btn_manual)
        v.addLayout(manual)

        self.lbl_qr = QLabel()
        self.lbl_qr.setAlignment(Qt.AlignCenter)
        self.lbl_qr.setFixedHeight(232)          # 二维码是正方形：给够高度，别让布局压扁
        self.lbl_qr.setVisible(False)
        v.addWidget(self.lbl_qr)

        src = QHBoxLayout()
        src.setSpacing(theme.GAP_S)
        src.addWidget(QLabel("来源"))
        self.cb_kind = QComboBox()
        self.cb_kind.addItems(["收藏夹", "合集", "观看历史"])
        self.cb_kind.currentIndexChanged.connect(lambda _i: self._load_sources())
        src.addWidget(self.cb_kind)
        self.cb_folder = QComboBox()
        self.cb_folder.setMinimumWidth(260)
        src.addWidget(self.cb_folder, 1)
        self.btn_load = QPushButton("读取清单")
        self.btn_load.clicked.connect(self._load_items)
        src.addWidget(self.btn_load)
        v.addLayout(src)

        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["标题", "时长", "UP 主"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.setColumnWidth(0, 520)
        self.tbl.setColumnWidth(1, 90)
        v.addWidget(self.tbl, 1)

        foot = QHBoxLayout()
        foot.setSpacing(theme.GAP_S)
        self.chk_all = QCheckBox("全选")
        self.chk_all.stateChanged.connect(self._toggle_all)
        foot.addWidget(self.chk_all)
        foot.addWidget(QLabel("导入为课程"))
        self.ed_course = QLineEdit()
        self.ed_course.setMinimumWidth(240)
        foot.addWidget(self.ed_course, 1)
        self.btn_ok = QPushButton("导入选中")
        self.btn_ok.clicked.connect(self._accept_chosen)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        foot.addWidget(self.btn_ok)
        foot.addWidget(self.btn_cancel)
        v.addLayout(foot)

        self.lbl_hint = QLabel("导入只搬标题/时长/链接：视频还在 B 站，点「看这一节」时"
                               "在浏览器打开并自动开始录。")
        self.lbl_hint.setObjectName("muted")
        self.lbl_hint.setWordWrap(True)
        v.addWidget(self.lbl_hint)

    # -- 后台任务小工具 --------------------------------------------------
    def _run(self, fn: Callable[[], object], ok: Callable[[object], None],
             bad: Callable[[str], None] | None = None) -> None:
        job = _Job(fn, self)
        job.ok.connect(ok)
        job.bad.connect(bad or (lambda m: self.lbl_hint.setText(f"出错：{m}")))
        job.finished.connect(lambda: self._jobs.remove(job) if job in self._jobs else None)
        self._jobs.append(job)
        job.start()

    # -- 登录 ------------------------------------------------------------
    def _refresh_login(self) -> None:
        self.lbl_login.setText("正在读取登录状态…")
        self._run(lambda: bili.login_state(self._transport), self._on_login)

    def _on_login(self, st: dict) -> None:
        if st.get("logged_in"):
            who = st.get("uname") or "已登录"
            via = "扫码保存的登录态" if st.get("via") == "saved" else "浏览器登录态"
            self.lbl_login.setText(f"已登录 B 站：{who}（{via}，只读，不写任何东西）")
            self._cookies = st.get("cookies") or {}
            self.btn_qr.setText("换账号")
            self.lbl_qr.setVisible(False)
            self._load_sources()
        else:
            err = str(st.get("error") or "")
            tail = f"（上次校验失败：{err}）" if err else ""
            self.lbl_login.setText(
                "还没登录 B 站。点下面的「扫码登录」用手机扫一下就成；"
                "或者先在浏览器登录 B 站、关掉浏览器后点「读浏览器登录」；"
                f"再不成可以把 SESSDATA 粘到下面那行。{tail}")
            self.btn_qr.setText("扫码登录")

    def _start_qr(self) -> None:
        self._poll.stop()
        self.lbl_qr.setText("正在取二维码…")
        self.lbl_qr.setVisible(True)

        def got(q: dict) -> None:
            self._qr_key = q.get("key", "")
            try:
                png = bili.qr_png(q.get("url", ""), scale=6)
                pm = QPixmap()
                pm.loadFromData(png, "PNG")
                pm = pm.scaled(220, 220, Qt.KeepAspectRatio, Qt.FastTransformation)
                self.lbl_qr.setPixmap(pm)
                self.lbl_qr.setText("")
                self.lbl_hint.setText("用手机 B 站 App 扫这个码；扫完在手机上确认。"
                                      "登录态只存在本机 data/bili_session.json。")
                self._poll.start()
            except Exception as e:               # noqa: BLE001
                self.lbl_qr.setText(f"二维码画不出来：{e}")

        self._run(lambda: bili.qr_generate(self._transport), got,
                  lambda m: self.lbl_qr.setText(f"拿不到二维码：{m}"))

    def _poll_qr(self) -> None:
        if not self._qr_key:
            self._poll.stop()
            return

        def got(r: dict) -> None:
            code = int(r.get("code", -1))
            if r.get("ok"):
                bili.save_login(r["cookies"])
                self._cookies = dict(r["cookies"])
                self.lbl_hint.setText("扫到了，正在校验登录态…")
                self._verify_after_login()
            elif code == 86101:
                self.lbl_hint.setText("还没扫到（86101）。用手机 B 站 App 扫这个码…")
            elif code == 86090:
                self.lbl_hint.setText("扫到了，等你在手机上点「确认登录」（86090）…")
            elif code == 0:
                # 扫码成功但凭据没到手：不能装作在等，更不能悄悄换一张新码
                self._poll.stop()
                self.lbl_hint.setText(
                    "扫码成功了，但 B 站这次没把凭据给出来（细节已记 data/logs/bili.log）。"
                    "再点一次「扫码登录」；或者把浏览器里的 SESSDATA 粘到下面那行。")
            elif code == 86038:
                # 过期不让用户白扫一次：当场换一张新码
                self.lbl_hint.setText("这张码过期了，已经自动换了新的，请扫新码。")
                self._start_qr()
            else:
                self.lbl_hint.setText(f"轮询异常：code={code} {r.get('message') or ''}")

        self._run(lambda: bili.qr_poll(self._qr_key, self._transport), got)

    def _verify_after_login(self) -> None:
        """拿到 cookie 后真的问一次"我是谁"：扫码成功 ≠ 能用。"""
        def job():
            sess = bili.Session(cookies=dict(self._cookies), transport=self._transport)
            me = sess.whoami()
            if not me.get("ok"):
                raise bili.BiliError("B 站说这个登录态无效（可能已被风控或过期）")
            bili.save_login(sess.cookies)          # 补上的指纹一起存
            return {"cookies": sess.cookies, "uname": me.get("uname", "")}

        def ok(r: dict) -> None:
            self._poll.stop()
            self._cookies = r["cookies"]
            self.lbl_qr.setVisible(False)
            self.lbl_hint.setText(f"登录成功：{r['uname'] or '已登录'}。可以选来源读取清单了。")
            self._refresh_login()

        def bad(msg: str) -> None:
            self._poll.stop()
            self.lbl_hint.setText(f"扫码通了但校验失败：{msg}（细节见 data/logs/bili.log）")

        self._run(job, ok, bad)

    def _use_manual(self) -> None:
        val = self.ed_sessdata.text().strip()
        if not val:
            self.lbl_hint.setText("先把 SESSDATA 粘进来。")
            return
        cookies = dict(self._cookies)
        cookies["SESSDATA"] = val
        self._cookies = cookies
        self.lbl_hint.setText("正在校验…")

        def job():
            sess = bili.Session(cookies=dict(cookies), transport=self._transport)
            me = sess.whoami()
            if not me.get("ok"):
                raise bili.BiliError("这个 SESSDATA 用不了（过期或被风控）")
            bili.save_login(sess.cookies)
            return {"cookies": sess.cookies, "uname": me.get("uname", "")}

        self._run(job, lambda r: (setattr(self, "_cookies", r["cookies"]),
                                  self.lbl_login.setText(
                                      f"已登录 B 站：{r['uname'] or '已登录'}（手动 SESSDATA）"),
                                  self._load_sources()),
                  lambda m: self.lbl_hint.setText(m))

    def _read_browser(self) -> None:
        self.lbl_login.setText("正在读浏览器 cookie…")

        def job() -> dict:
            ck = bili.read_cookies()
            if not ck.get("SESSDATA"):
                raise bili.BiliError(
                    "没读到 bilibili 登录态。浏览器开着时 cookie 库是独占锁的："
                    "要么关掉浏览器再试，要么直接用扫码登录。")
            st = bili.Session(cookies=ck).whoami()
            if not st.get("ok"):
                raise bili.BiliError("读到了 cookie 但 B 站说没登录（可能已过期）。")
            return {"cookies": ck, "uname": st.get("uname", "")}

        self._run(job, lambda d: (self._cookies.update(d["cookies"]),
                                  self.lbl_login.setText(
                                      f"已借用浏览器登录态：{d['uname']}"),
                                  self._load_sources()),
                  lambda m: self.lbl_login.setText(m))

    # -- 清单 ------------------------------------------------------------
    def _sess(self) -> bili.Session:
        return bili.Session(cookies=self._cookies, transport=self._transport)

    def _load_sources(self) -> None:
        if not self._cookies.get("SESSDATA"):
            return
        kind = self.cb_kind.currentIndex()
        self.cb_folder.clear()
        if kind == 2:                            # 观看历史不用选文件夹
            self.cb_folder.addItem("最近看过（30 条）", 0)
            return

        def got(rows: list[dict]) -> None:
            for r in rows:
                self.cb_folder.addItem(f"{r['title']}（{r['count']}）",
                                       (r["id"], r.get("mid")))
            if not rows:
                self.cb_folder.addItem("（这里什么都没有）", None)

        if kind == 0:
            self._run(lambda: self._sess().fav_folders(), got)
        else:
            self._run(lambda: self._sess().my_seasons(), got)

    def _load_items(self) -> None:
        if not self._cookies.get("SESSDATA"):
            self.lbl_hint.setText("先登录再读取。")
            return
        kind = self.cb_kind.currentIndex()
        payload = self.cb_folder.currentData()
        self.lbl_hint.setText("正在读取…")

        def job() -> list[dict]:
            s = self._sess()
            if kind == 2:
                items = s.history(pages=1)
            elif kind == 0 and payload:
                items = s.fav_items(int(payload[0]))
            elif payload:
                items = s.season_items(int(payload[0]), int(payload[1] or 1))
            else:
                items = []
            return [i.as_dict() for i in items]

        def got(items: list[dict]) -> None:
            self.tbl.setRowCount(0)
            for it in items:
                r = self.tbl.rowCount()
                self.tbl.insertRow(r)
                cell = QTableWidgetItem(it["title"])
                cell.setFlags(cell.flags() | Qt.ItemIsUserCheckable)
                cell.setCheckState(Qt.Checked)
                cell.setData(Qt.UserRole, it)
                self.tbl.setItem(r, 0, cell)
                self.tbl.setItem(r, 1, QTableWidgetItem(bili.hms(it.get("duration", 0))))
                self.tbl.setItem(r, 2, QTableWidgetItem(it.get("upper", "")))
            if not self.ed_course.text().strip():
                name = self.cb_folder.currentText().split("（")[0]
                self.ed_course.setText(f"B站 · {name}" if name else "B站 · 我的课程")
            self.lbl_hint.setText(f"读到 {len(items)} 条；勾选要导入的，然后「导入选中」。")

        self._run(job, got)

    # -- 结束 ------------------------------------------------------------
    def _toggle_all(self, state: int) -> None:
        on = Qt.Checked if state else Qt.Unchecked
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it is not None:
                it.setCheckState(on)

    def _accept_chosen(self) -> None:
        picked = []
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it is None or it.checkState() != Qt.Checked:
                continue
            data = it.data(Qt.UserRole)
            if data:
                picked.append(data)
        if not picked:
            self.lbl_hint.setText("一条也没勾。")
            return
        self.course_title = self.ed_course.text().strip() or "B站 · 我的课程"
        self.chosen = picked
        self.accept()

    def closeEvent(self, ev) -> None:            # noqa: N802
        self._poll.stop()
        super().closeEvent(ev)