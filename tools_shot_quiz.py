"""小测页截图：出题 → 作答 → 掌握度变化 → 待答问题。用假模型，不花钱不打网络。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)      # 必须原生平台：offscreen 没中文字体

from PySide6.QtWidgets import QApplication   # noqa: E402

from dsh_app.config import Settings          # noqa: E402
from dsh_app.db import DB                    # noqa: E402
from dsh_app.llm import JsonCall             # noqa: E402
from dsh_app.pipeline import Engine          # noqa: E402
from dsh_app.ui.main_window import MainWindow  # noqa: E402


class ShootLLM:
    """假模型：按预置脚本出题/答题，界面看到的行为与真模型一致。"""

    def __init__(self, engines: dict):
        self.configured = True
        self.script = engines

    def complete_json(self, system, user, *, expect_key=None, max_tokens=1200) -> JsonCall:
        ev = self.script["evidence"]
        body = {"questions": [{
            "prompt": "关于「必要条件」，下列说法哪一个是正确的？",
            "choices": ["矩阵可逆是方程组有唯一解的必要条件",
                        "矩阵可逆是方程组有唯一解的充分必要条件",
                        "方程组有解与矩阵是否可逆无关",
                        "只要行列式不为零就一定有多个解"],
            "correct_choice_index": 0,
            "evidence": ev,
        }, {
            "prompt": "为什么「边界条件」在求最优解时必须单独检查？",
            "choices": ["因为梯度为零只说明是驻点，不一定取到最优",
                        "因为边界条件可以随便取",
                        "因为边界上的导数一定存在",
                        "因为优化问题都有唯一解"],
            "correct_choice_index": 0,
            "evidence": ev,
        }, {
            "prompt": "这道题用来演示「出处查不到就作废」是怎么处理的。",
            "choices": ["选项甲说法", "选项乙说法", "选项丙说法", "选项丁说法"],
            "correct_choice_index": 1,
            "evidence": "教材第 88 页明确指出这个结论成立",
        }]}
        return JsonCall(payload=body, text=json.dumps(body, ensure_ascii=False),
                        attempts=2 if self.script.get("hardened") else 1,
                        error="", hardened=bool(self.script.get("hardened")))

    def complete_text(self, system, user, max_tokens=1200) -> JsonCall:
        # 答案要跟着问题走，否则截图里会出现"问东答西"的假象
        q = ""
        if "【用户的问题】" in user:
            q = user.split("【用户的问题】")[-1].strip().splitlines()[0][:40]
        if "梯度" in q:
            ans = ("梯度为零只说明这是驻点，不代表最优：还要比较可行域边界的取值，"
                   "并看二阶条件（Hessian）是否正定。")
        elif "边界" in q:
            ans = ("不是一回事。约束条件描述可行域，边界条件是其中的极端情形；"
                   "求最优解时两者都要单独检查。")
        else:
            ans = "因为行秩等于列秩，所以秩同时是行空间与列空间的维数。"
        return JsonCall(text=ans, attempts=1, error="")

    def chat(self, messages, max_tokens=1200, **kw):
        # chat.ask 走的是 chat()，要把真正的 user 消息传下去，答案才跟得上问题
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = m.get("content") or ""
        if isinstance(user, list):
            user = " ".join(str(p.get("text", "")) for p in user if isinstance(p, dict))
        return self.complete_text("", str(user)).text, ""


def pump(app, seconds: float = 0.6) -> None:
    for _ in range(max(1, int(seconds / 0.05))):
        app.processEvents()
        time.sleep(0.05)


def main() -> None:
    db_path = ROOT / "data" / "probe" / "shot_quiz.db"
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    st = Settings.load()
    st.llm.preload_on_start = False
    eng = Engine(db, st)
    sid = eng.seed_demo()
    eng.llm = ShootLLM({"evidence": "占位", "hardened": True})

    knot = sorted(db.get_knots(sid), key=lambda k: k.t_start)[0]
    src = eng.knot_source_text(knot.id)
    ev = (src["text"] or knot.excerpt)[:18]
    eng.llm = ShootLLM({"evidence": ev, "hardened": True})

    # 先攒两条问题，截图里能看到队列
    eng.queue_question(sid, "为什么梯度为零不一定是最优解？", t_sec=118.0)
    eng.queue_question(sid, "边界条件和约束条件是一回事吗？", t_sec=286.0)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(st, db)
    # 关键：窗口自己 new 了一个 Engine，必须把假模型换到**窗口那一个**上，
    # 否则点「出题」打的是真实网络，截图会停在"正在出题…"
    win.engine.llm = eng.llm
    win.show()
    win.current_session = sid
    for _ in range(6):
        app.processEvents()
        time.sleep(0.05)
    win._load_session_view()
    win.tabs.setCurrentIndex(2)
    win.refresh_quiz_knots()
    win.refresh_questions()
    pump(app)

    # 出题（走 Worker 线程，等它落地）
    win._on_make_quiz()
    for _ in range(80):
        app.processEvents()
        time.sleep(0.05)
        if win.quiz_layout.count() > 2:
            break
    pump(app, 0.4)

    # 第一题答对
    cards = [w for w in win.quiz_holder.findChildren(type(win.quiz_holder))]
    from PySide6.QtWidgets import QFrame, QPushButton, QRadioButton
    frames = win.quiz_holder.findChildren(QFrame)
    graded = 0
    for fr in frames:
        radios = fr.findChildren(QRadioButton)
        if not radios:
            continue
        radios[0].setChecked(True)
        for b in fr.findChildren(QPushButton):
            if b.text() == "提交":
                b.click()
                graded += 1
                break
        pump(app, 0.25)
        if graded >= 1:
            break
    # 第二题故意答错（选 B）
    for fr in frames:
        radios = fr.findChildren(QRadioButton)
        if not radios or fr.findChildren(QPushButton)[0].isEnabled() is False:
            continue
        if radios[0].isChecked():
            continue
        radios[1].setChecked(True)
        for b in fr.findChildren(QPushButton):
            if b.text() == "提交":
                b.click()
                break
        pump(app, 0.25)
        break
    pump(app, 0.5)

    shot = ROOT / "docs" / "screenshots" / "20-quiz.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(shot))
    print("已保存:", shot, shot.stat().st_size, "字节")

    # ---- 第二张：待答问题全部回答之后 ----
    win.q_list.setCurrentRow(0)
    pump(app, 0.2)
    win._on_answer_all()
    for _ in range(80):
        app.processEvents()
        time.sleep(0.05)
        if "答了" in win.q_out.toPlainText():
            break
    pump(app, 0.3)
    win.q_list.setCurrentRow(0)
    pump(app, 0.3)
    shot2 = ROOT / "docs" / "screenshots" / "21-queue-answered.png"
    win.grab().save(str(shot2))
    print("已保存:", shot2, shot2.stat().st_size, "字节")
    print("队列回答结果:", win.q_out.toPlainText()[:120].replace(chr(10), " "))

    # ---- 第三张：时间轴页的覆盖度诚实提示（先造一帧屏幕笔记）----
    knot0 = sorted(db.get_knots(sid), key=lambda k: k.t_start)[0]
    db.add_screen_note(sid, float(knot0.t_start) + 3, "板书：必要条件与充分条件的区别", kind="board")
    win.tabs.setCurrentIndex(0)
    win._load_session_view()
    pump(app, 0.5)
    shot3 = ROOT / "docs" / "screenshots" / "22-coverage.png"
    win.grab().save(str(shot3))
    print("已保存:", shot3, shot3.stat().st_size, "字节")
    print("覆盖度提示:", win.lbl_coverage.text())
    win.tabs.setCurrentIndex(2)

    stats = db.quiz_stats(sid)
    print("小测统计:", stats)
    for r in db.query("SELECT prompt, evidence_ok, void_reason, answered_index, correct_index "
                      "FROM quiz_items WHERE session_id=?", (sid,)):
        print("  ", dict(r))
    print("掌握度:", [(r["name"], round(float(r["mastery"] or 0), 3), r["attempts"])
                    for r in db.get_knowledge_points(db.get_session(sid)["course_id"])])
    print("覆盖度:", eng.capture_coverage(sid))

    # 导出笔记截图素材
    exp = eng.export_notes(sid, ROOT / "data" / "probe" / "export_shot.md")
    print("导出:", exp)
    win.close()


if __name__ == "__main__":
    main()