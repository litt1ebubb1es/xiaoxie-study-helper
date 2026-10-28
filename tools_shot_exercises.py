"""截图：小测页 + 导入的练习出题（离线，不调模型）。"""
import os
import sys
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from dsh_app import exercises as ex
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine
from dsh_app.ui.main_window import MainWindow

PROBE = Path("data/probe/shot_exercises.db")
PROBE.parent.mkdir(parents=True, exist_ok=True)
for f in (PROBE, PROBE.with_name(PROBE.name + "-wal"), PROBE.with_name(PROBE.name + "-shm")):
    if f.exists():
        f.unlink()

db = DB(PROBE)
st = Settings.load()
st.llm.preload_on_start = False
st.llm.base_url = ""
st.llm.api_key = ""
eng = Engine(db, st)
sid = eng.seed_demo()
cid = db.get_session(sid)["course_id"]

# 造一份"老师发的练习"：三条按知识点匹配，一条故意对不上术语名
tpl = Path("data/probe/练习.xlsx")
ex.write_template(tpl)
rows = ex.read_table(tpl)
rows += [
    ["方差", "方差的定义是偏离均值的什么？", "平方的期望", "绝对值的期望", "中位数", "众数",
     "A", "方差是 E[(X-EX)^2]。"],
    ["随机变量", "随机变量的本质是什么？", "从样本空间到实数的函数", "一个概率值",
     "一个常数", "一个集合", "A", "随机变量是函数，不是变量。"],
    ["老师没讲过的术语", "这一条对不上任何知识点？", "甲", "乙", "丙", "丁", "A", ""],
]
import csv
p = Path("data/probe/练习-导入.csv")
with p.open("w", encoding="utf-8-sig", newline="") as f:
    csv.writer(f).writerows(rows)
res = eng.import_exercises(p, cid)
print("导入:", res["imported"], "跳过:", res["skipped"], "对上知识点:", res["matched"],
      "没对上:", res["unmatched"])

app = QApplication.instance() or QApplication([])
win = MainWindow(st, db)
win.engine.llm = eng.llm
win.resize(1596, 1000)
win.show()
app.processEvents()

win.current_session = sid
win._load_session_view()
names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
win.tabs.setCurrentIndex(names.index("小测与待答"))
win.refresh_exercise_label()
app.processEvents()

d = Path("docs/screenshots")
d.mkdir(parents=True, exist_ok=True)
out = d / "25-exercise-import.png"
win.grab().save(str(out))
print("练习条:", win.lbl_ex.text())
print("已保存:", out, out.stat().st_size, "字节")

# 再用导入的练习出 3 道题
q = eng.quiz_from_exercises(cid, sid, k=3)
print("出题:", len(q["items"]))
win._quiz_from_ex_done(q)
app.processEvents()
out2 = d / "26-quiz-from-exercises.png"
win.grab().save(str(out2))
print("提示:", win.lbl_quiz.text())
print("已保存:", out2, out2.stat().st_size, "字节")