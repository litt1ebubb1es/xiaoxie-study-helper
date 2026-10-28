"""复习排程截图：知识点表带「记忆保持」，到期标黄，下一步给建议。

用假模型，不联网。故意把一个知识点的时间推到 40 天前，让"该复习"能看见。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)

from PySide6.QtWidgets import QApplication   # noqa: E402

from dsh_app.config import Settings          # noqa: E402
from dsh_app.db import DB                    # noqa: E402
from dsh_app.llm import JsonCall             # noqa: E402
from dsh_app.pipeline import Engine          # noqa: E402
from dsh_app.ui.main_window import MainWindow  # noqa: E402


class QuietLLM:
    configured = True

    def complete_json(self, *a, **kw) -> JsonCall:
        return JsonCall(payload={"questions": []}, text='{"questions":[]}',
                        attempts=1, error="")

    def complete_text(self, *a, **kw) -> JsonCall:
        return JsonCall(text="（截图用，不调用真模型）", attempts=1, error="")

    def chat(self, *a, **kw):
        return "（截图用）", ""


def pump(app, seconds: float = 0.5) -> None:
    for _ in range(max(1, int(seconds / 0.05))):
        app.processEvents()
        time.sleep(0.05)


def main() -> None:
    db_path = ROOT / "data" / "probe" / "shot_review.db"
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    st = Settings.load()
    st.llm.preload_on_start = False
    eng = Engine(db, st)
    eng.llm = QuietLLM()
    sid = eng.seed_demo()
    cid = str(db.get_session(sid)["course_id"])

    kps = db.get_knowledge_points(cid)
    # 一个补上了但很久没碰（该复习），一个在补，一个没测过
    k1, k2, k3 = int(kps[0]["id"]), int(kps[1]["id"]), int(kps[2]["id"])
    eng.record_kp_attempt(k1, 1.0, source="quiz")
    eng.record_kp_attempt(k1, 1.0, source="quiz")
    eng.record_kp_attempt(k1, 1.0, source="quiz")
    eng.record_kp_attempt(k2, 1.0, source="quiz")
    old = time.time() - 40 * 86400
    db.execute("UPDATE knowledge_points SET last_evidence_at=?, due_at=? WHERE id=?",
               (old, old, k1))

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(st, db)
    win.engine.llm = eng.llm
    win.show()
    win.current_session = sid
    for _ in range(8):
        app.processEvents()
        time.sleep(0.05)
    win._load_session_view()

    win.tabs.setCurrentIndex(3)          # 漏洞地图：知识点表 + 记忆保持
    win.refresh_kp_table()
    pump(app, 0.6)
    shot = ROOT / "docs" / "screenshots" / "23-review.png"
    win.grab().save(str(shot))
    print("已保存:", shot, shot.stat().st_size, "字节")
    print("表头:", [win.kp_table.horizontalHeaderItem(i).text()
                  for i in range(win.kp_table.columnCount())])
    for r in range(win.kp_table.rowCount()):
        print("  ", [win.kp_table.item(r, c).text() if win.kp_table.item(r, c) else ""
                     for c in range(win.kp_table.columnCount())])
    print("提示:", win.lbl_kp.text())

    # 下一步建议
    win.tabs.setCurrentIndex(2)
    win._on_next_action()
    pump(app, 0.4)
    print("下一步:", win.lbl_next.text())
    shot2 = ROOT / "docs" / "screenshots" / "24-next-action.png"
    win.grab().save(str(shot2))
    print("已保存:", shot2, shot2.stat().st_size, "字节")

    print("复习队列:", [(q["name"], q["retention"], q["overdue"], q["due_in_h"])
                    for q in eng.review_queue(cid)])
    win.close()


if __name__ == "__main__":
    main()