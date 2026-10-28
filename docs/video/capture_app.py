"""实机画面：把真窗口跑起来，按脚本点/拖/打字，逐帧 grab 存 PNG。

为什么不用录屏：PrintWindow/gdigrab 抓不到离屏合成、还会被缩放和撕裂干扰。
`QWidget.grab()` 拿到的是控件自己渲染的那一帧——像素稳定、可重复，和 pdoom 的
"画面是时间的纯函数"是一路的做法。

这一版按 2560×1440 抓（本机就是 2K 屏）：源像素比成片多 1.33 倍，
镜头推到 1.9 倍时还没到"把 1080p 截图放大"的程度，推轨才敢做大。

产物：
    docs/video/work/appseq/%05d.png     2560×1440 逐帧（30 fps）
    docs/video/work/appseq/manifest.json  fps/时长/尺寸 + 每个点击 + 种子信息

用法：
    ./.venv/Scripts/python.exe docs/video/capture_app.py --probe     # 每场抓一帧看看
    ./.venv/Scripts/python.exe docs/video/capture_app.py             # 全片
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # 仓库根：换机器不用改
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "video" / "work" / "appseq"
DB_SRC = ROOT / "data" / "selftest.db"
DB_COPY = ROOT / "docs" / "video" / "work" / "capture.db"

W, H, FPS = 2560, 1440, 30
TOTAL = 54.0                      # 秒（源时间线）
# 探帧时间点：每场演完的那一帧
PROBE_AT = {int(x * FPS) for x in (2.0, 7.0, 12.5, 16.5, 21.0, 25.0, 30.5, 35.0, 38.5,
                                   42.0, 45.0, 48.0, 51.0, 53.5)}

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest       # noqa: E402
from PySide6.QtWidgets import (QApplication, QPushButton,  # noqa: E402
                               QRadioButton)

# ---------------------------------------------------------------- 数据准备

ANSWER = ("一阶近似在「变化足够小」的邻域里成立：把曲线在这一点的切线当成曲线本身用。"
          "代价是误差随步长二阶增长，所以只在小范围内用它。" + chr(10) * 2 +
          "本节 02:41 的原话：「只要偏移足够小，忽略二阶项带来的误差就可以接受」。")

STREAM = ["一阶近似只在很小的邻域里成立",
          "一阶近似只在很小的邻域里成立：把曲线在这一点的切线",
          "一阶近似只在很小的邻域里成立：把曲线在这一点的切线当成曲线本身用。",
          "一阶近似只在很小的邻域里成立：把曲线在这一点的切线当成曲线本身用。代价是误差",
          "一阶近似只在很小的邻域里成立：把曲线在这一点的切线当成曲线本身用。代价是误差随步长二阶增长，",
          ANSWER]

SOURCES = [{"doc": "第 1 讲 讲义.pdf", "source_ref": "第 12 页", "score": "0.82",
            "snippet": "只要偏移足够小，忽略二阶项带来的误差就可以接受"},]

REWORD_TEXT = ("先别管「分布函数」这个词。把它当成一张清单：每个取值范围对应一个概率。\n\n"
               "1）离散的：把每个取值和它的概率列成两行，这就是分布。\n"
               "2）连续的：单点概率是 0，只有区间才有概率，所以看的是面积。\n"
               "3）判断标准：所有概率加起来（或总面积）必须是 1，不然就不是分布。\n\n"
               "回到本节 01:48 那一句：老师说的「分布描述的是取值落在各个区间的概率规律」，"
               "就是把这张清单换成了连续版本。")

QUIZ_SEED = [
    # (prompt, choices, correct_index, evidence, void_reason, kp_name)
    ("按原文，随机变量是什么？",
     ["把样本空间映射到实数轴的一个函数", "全部可能结果的集合",
      "随机变量的加权平均", "偏离均值的平方的期望"], 0,
     "随机变量就是把样本空间映射到实数轴上的一个函数", "", "随机变量"),
    ("原文说的「分布」指的是哪一层东西？",
     ["某一个具体取值出现的概率规律", "样本空间里元素的个数",
      "样本均值的大小", "一次实验的结果"], 0,
     "分布描述的是取值落在各个区间的概率规律", "", "分布"),
    ("这一段里，期望 E(X) 的计算方式是什么？",
     ["每个取值乘它的概率再求和", "取值的中位数", "取值的众数", "最大值减最小值"], 0,
     "", "引文在本节原文里找不到", "期望"),
]


def seed(db_path: Path, sess: str) -> dict:
    """给录像用的演示库塞三道题 + 两条待答 + 一份讲义条目；不动真库。"""
    import sqlite3

    c = sqlite3.connect(db_path)
    kps = {str(r[1]): int(r[0]) for r in c.execute("SELECT id, name FROM knowledge_points")}
    knots = list(c.execute("SELECT id, t_start FROM knots WHERE session_id=? "
                           "ORDER BY t_start LIMIT 3", (sess,)))
    course = c.execute("SELECT course_id FROM sessions WHERE id=?", (sess,)).fetchone()[0]

    c.execute("DELETE FROM quiz_items WHERE session_id=?", (sess,))
    for i, (prompt, choices, ci, ev, void, kp_name) in enumerate(QUIZ_SEED):
        kid = knots[min(i, len(knots) - 1)][0] if knots else None
        c.execute("""INSERT INTO quiz_items
            (session_id, knot_id, kp_id, prompt, choices_json, correct_index,
             evidence, evidence_ok, void_reason, answered_index, answered_at,
             created_at, source)
            VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,?, 'model')""",
                  (sess, kid, kps.get(kp_name), prompt, json.dumps(choices, ensure_ascii=False),
                   ci, ev, 0 if void else 1, void, time.time()))

    c.execute("DELETE FROM kb_chunks WHERE course_id=?", (course,))
    c.execute("DELETE FROM kb_docs WHERE course_id=?", (course,))
    c.execute("""INSERT INTO kb_docs (course_id, title, path, note, chars, n_chunks, created_at)
                 VALUES (?,?,?,?,?,?, datetime('now'))""",
              (course, "第 1 讲 讲义.pdf", "", "随课讲义：随机变量 / 分布 / 期望", 18420, 24))

    c.execute("DELETE FROM pending_questions WHERE session_id=?", (sess,))
    for t, txt, ex in [(118.0, "这里为什么要用一阶近似？", "一阶近似只在很小的邻域里成立"),
                       (286.0, "边界条件和约束条件是一回事吗？", "边界是定义域，约束是附加条件")]:
        c.execute("""INSERT INTO pending_questions
            (session_id, t_sec, text, excerpt, status, answer, created_at, answered_at)
            VALUES (?,?,?,?, 'pending', NULL, ?, NULL)""",
                  (sess, t, txt, ex, time.time()))
    c.commit()
    n = c.execute("SELECT COUNT(*) FROM quiz_items WHERE session_id=?", (sess,)).fetchone()[0]
    c.close()
    return {"session": sess, "quiz_items": n, "knots": len(knots)}


CUR_T = 0.0
CLICKS: list[dict] = []
DETECT: dict = {}
REWORD = None


def note_click(win, widget, local_pt, label=""):
    """记下点击落在窗口坐标的哪里：视频里的指针按这张表走。"""
    p = widget.mapTo(win, local_pt)
    CLICKS.append({"t": round(CUR_T, 3), "x": int(p.x()), "y": int(p.y()), "label": label})


# ---------------------------------------------------------------- 脚本编排


def build_steps(win, app) -> list:
    """返回 [(t_sec, fn)]，帧循环按时间把它们排进去。"""
    ev: list[tuple[float, object]] = []

    def at(t, fn):
        ev.append((float(t), fn))

    # ---- 基础动作
    def nav(row):
        def _f():
            item = win.nav.item(row)
            pos = win.nav.visualItemRect(item).center()
            note_click(win, win.nav.viewport(), pos)
            QTest.mouseClick(win.nav.viewport(), Qt.LeftButton, pos=pos)
        return _f

    def click_session(i):
        def _f():
            item = win.session_list.item(i)
            win.session_list.scrollToItem(item)
            app.processEvents()
            pos = win.session_list.visualItemRect(item).center()
            note_click(win, win.session_list.viewport(), pos)
            QTest.mouseClick(win.session_list.viewport(), Qt.LeftButton, pos=pos)
        return _f

    def scrub(frac):
        def _f():
            w = win.timeline
            r = w.rect()
            pos = QPoint(int(r.width() * frac), int(r.height() * 0.46))
            note_click(win, w, pos)
            QTest.mouseClick(w, Qt.LeftButton, pos=pos)
        return _f

    def click_row(row, col=4):
        def _f():
            t = win.knot_table
            pos = t.visualRect(t.model().index(row, col)).center()
            note_click(win, t.viewport(), pos)
            QTest.mouseClick(t.viewport(), Qt.LeftButton, pos=pos)
        return _f

    def click_kp_row(row):
        def _f():
            t = win.kp_table
            pos = t.visualRect(t.model().index(row, 0)).center()
            note_click(win, t.viewport(), pos)
            QTest.mouseClick(t.viewport(), Qt.LeftButton, pos=pos)
        return _f

    def click_button(btn, label=""):
        def _f():
            if not btn.isVisible():
                return
            pos = btn.rect().center()
            note_click(win, btn, pos, label)
            QTest.mouseClick(btn, Qt.LeftButton, pos=pos)
        return _f

    def click_quiet(btn, label: str = "", after=None):
        """按一下给个涟漪，但挡掉信号：结果由时间线自己安排，避免线程完成时机漂移。"""
        def _f():
            if not btn.isVisible():
                return
            pos = btn.rect().center()
            note_click(win, btn, pos, label)
            btn.blockSignals(True)
            QTest.mouseClick(btn, Qt.LeftButton, pos=pos)
            btn.blockSignals(False)
            if after is not None:
                after()
        return _f

    def answer(q_index: int, choice: int):
        """点第 q_index 道题的某个选项（提交是单独一拍）。"""
        def _f():
            groups: dict[int, list[QRadioButton]] = {}
            for r in win.quiz_holder.findChildren(QRadioButton):
                groups.setdefault(int(r.property("qitem")), []).append(r)
            keys = list(groups)
            if len(keys) <= q_index:
                return
            opts = groups[keys[q_index]]
            if choice < len(opts):
                # 单选按钮被布局拉满整行：点中心会点在空白处，靠左 18px 才是圆点
                pos = QPoint(18, opts[choice].height() // 2)
                note_click(win, opts[choice], pos, "选")
                QTest.mouseClick(opts[choice], Qt.LeftButton, pos=pos)
        return _f

    def submit(q_index: int):
        def _f():
            groups: dict[int, list[QRadioButton]] = {}
            for r in win.quiz_holder.findChildren(QRadioButton):
                groups.setdefault(int(r.property("qitem")), []).append(r)
            keys = list(groups)
            if len(keys) <= q_index:
                return
            card = groups[keys[q_index]][0].parentWidget()
            for b in card.findChildren(QPushButton):
                if b.text() == "提交":
                    pos = b.rect().center()
                    note_click(win, b, pos, "提交")
                    QTest.mouseClick(b, Qt.LeftButton, pos=pos)
                    break
        return _f

    def scroll_to(widget, frac: float):
        def _f():
            bar = widget.verticalScrollBar()
            bar.setValue(int(bar.maximum() * frac))
        return _f

    def type_ask(chunk: str):
        def _f():
            win.ed_ask.insertPlainText(chunk)
            win.ed_ask.setFocus()
        return _f

    def send_ask():
        def _f():
            pos = win.btn_ask.rect().center()
            note_click(win, win.btn_ask, pos, "发送")
            QTest.mouseClick(win.btn_ask, Qt.LeftButton, pos=pos)
        return _f

    def stream(text: str, final: bool = False):
        """模型逐字吐：每次改最后一条助手消息再重画，看起来就是真在生成。"""
        def _f():
            from dsh_app.ui.chat_html import chat_html

            msgs = win._chat_msgs
            if not msgs or msgs[-1]["role"] != "assistant":
                win._append_chat("assistant", text, SOURCES if final else [])
            else:
                msgs[-1]["content"] = text
                if final:
                    msgs[-1]["sources"] = SOURCES
                win.chat_view.setHtml(chat_html(msgs, win.settings.llm.model))
            bar = win.chat_view.verticalScrollBar()
            bar.setValue(bar.maximum())
            if final:
                win.lbl_chat_state.setText("")
                win.btn_ask.setEnabled(True)
        return _f

    def zoom_timeline(factor, anchor):
        def _f():
            win.timeline.zoom_at(factor, anchor)
        return _f

    def quiz_render():
        def _f():
            win._quiz_done({"ok": True, "items": quiz_items(win), "void": []})
        return _f

    # ---- 编舞（秒）
    at(2.9, click_session(3))                 # 点开第 4 行那一节
    at(4.6, scrub(0.16))                      # 拨时间轴
    at(5.9, scrub(0.72))
    at(7.1, scrub(0.45))
    at(8.9, click_row(1))                     # 选回看点
    at(10.4, click_row(2))
    at(11.9, click_button(win.btn_copy_ts, "复制"))   # 复制时间点
    at(13.2, nav(2))                          # 小测与待答
    at(14.2, quiz_render())
    at(15.9, answer(0, 0)); at(17.1, submit(0))       # 第 1 题答对
    at(18.8, lambda: win.quiz_area.verticalScrollBar().setValue(430))
    at(19.9, answer(1, 0)); at(21.0, submit(1))       # 第 2 题
    at(22.4, lambda: win.quiz_area.verticalScrollBar().setValue(
        win.quiz_area.verticalScrollBar().maximum()))  # 第 3 题作废
    at(24.2, nav(3))                          # 漏洞地图
    at(25.0, click_kp_row(0))
    at(26.9, nav(5))                          # 设置与模型部署
    at(27.2, lambda: win._settings_scroll.verticalScrollBar().setValue(
        int(win._settings_scroll.verticalScrollBar().maximum() * 0.55)))
    at(27.9, click_quiet(win.btn_detect, "检测",
                         after=lambda: win.deploy_view.setPlainText("正在检测本机算力与模型…")))
    at(29.0, lambda: win._detect_done(DETECT))         # 本机实测结果，开场时算好
    at(31.4, nav(1))                          # 划段重讲（模型生成）
    at(32.2, click_quiet(win.btn_reword, "生成",
                         after=lambda: win.reword_out.setPlainText("正在生成…（本地模型首次调用可能需要几秒）")))
    at(34.6, lambda: win._reword_done(REWORD))         # 模型成稿
    at(35.0, scroll_to(win.reword_out, 0.0))  # 成稿回到顶部
    at(36.8, nav(4))                          # 问模型
    at(37.6, type_ask("这里的「一阶"))
    at(38.2, type_ask("近似」到底"))
    at(38.8, type_ask("为什么能这么用？"))
    at(39.6, send_ask())
    for k, chunk in enumerate(STREAM):
        at(40.4 + 0.55 * k, stream(chunk, final=(k == len(STREAM) - 1)))
    at(44.4, nav(0))                          # 回到时间轴
    at(45.2, scrub(0.32))
    at(46.2, zoom_timeline(1.9, 150.0))       # 放大看那一段
    at(47.4, zoom_timeline(1.9, 150.0))
    at(48.6, scrub(0.30))
    at(50.2, click_row(1))
    at(51.6, scrub(0.34))
    return ev


def quiz_items(win) -> list[dict]:
    rows = win.db.query("SELECT * FROM quiz_items WHERE session_id=? ORDER BY id",
                        (win.current_session,))
    out = []
    for r in rows:
        out.append({
            "item_id": int(r["id"]),
            "prompt": str(r["prompt"]),
            "choices": json.loads(r["choices_json"] or "[]"),
            "void": not bool(r["evidence_ok"]),
            "void_reason": str(r["void_reason"] or ""),
        })
    return out


# ---------------------------------------------------------------- 主循环


def main() -> int:
    probe = "--probe" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    DB_COPY.parent.mkdir(parents=True, exist_ok=True)
    if DB_COPY.exists():
        DB_COPY.unlink()
    shutil.copy(DB_SRC, DB_COPY)

    from dsh_app.db import DB as _DB
    _db = _DB(DB_COPY)
    rows = _db.list_sessions()
    PICKED_ROW = 3                      # 片子里点开的是列表第 4 行
    sess = str(rows[PICKED_ROW]["id"])
    _db.close()
    info = seed(DB_COPY, sess)
    info["row"] = PICKED_ROW
    print("演示库:", info)

    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine, RewordResult
    from dsh_app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    s = Settings.load()
    db = DB(DB_COPY)
    eng = Engine(db, s)

    def fake_ask(q, session_id="", use_kb=True, use_context=True):
        # 真后端要联网；这里让 Worker 一直"思考"，实际是下面 stream() 一段段写进去
        time.sleep(90)
        return {"answer": ANSWER, "sources": SOURCES}

    def fake_reword(knot_id, strategy_key=None):
        time.sleep(2.4)
        return RewordResult(text=REWORD_TEXT, strategy=str(strategy_key or ""),
                            strategy_name="换一个类比", degraded=False)

    eng.ask = fake_ask
    eng.reword = fake_reword

    global DETECT, REWORD
    from dsh_app.local_deploy import detect_all
    DETECT = detect_all()
    print("本机检测:", DETECT["recommendation"]["backend"], DETECT["recommendation"]["model"])
    REWORD = RewordResult(text=REWORD_TEXT, strategy="analogy",
                          strategy_name="换一个类比", degraded=False)

    win = MainWindow(s, db)
    win.engine.ask = fake_ask
    win.engine.reword = fake_reword
    win.resize(W, H)
    win.show()
    win.raise_()
    win.activateWindow()
    for _ in range(12):
        app.processEvents()
    px = win.grab()
    size = [px.width(), px.height()]
    print("实际抓图尺寸:", size)

    win.ed_base.setText("https://dashscope.aliyuncs.com/compatible-mode/v1")
    win._on_session_activated(win.session_list.item(info["row"]))
    for _ in range(10):
        app.processEvents()

    steps = build_steps(win, app)
    steps.sort(key=lambda kv: kv[0])
    n_frames = int(TOTAL * FPS)

    si = 0
    t_start = time.time()
    for f in range(n_frames):
        t = f / FPS
        global CUR_T
        CUR_T = t
        while si < len(steps) and steps[si][0] <= t:
            fn = steps[si][1]
            try:
                fn()
            except Exception as e:                       # noqa: BLE001
                print(f"  [警告] {t:.1f}s 的动作失败: {e}")
            si += 1
        app.processEvents()

        if probe:
            if f in PROBE_AT:
                q = OUT / f"probe_{f:05d}.png"
                win.grab().save(str(q))
                print("  探帧", f"{t:5.1f}s", q.name)
            continue
        win.grab().save(str(OUT / f"{f:05d}.png"))

    if probe:
        print("probe 模式：只写探帧")
        db.close()
        return 0

    manifest = {"fps": FPS, "size": size, "duration": TOTAL, "frames": n_frames,
                "clicks": CLICKS, "seed": info}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"记录点击 {len(CLICKS)} 次")
    print(f"完成：{n_frames} 帧，用时 {time.time() - t_start:.1f}s → {OUT}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())