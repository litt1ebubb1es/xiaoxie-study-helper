"""截图：漏洞地图页的知识点掌握度表（真实库，无模型调用）。"""
import os
import sys
from pathlib import Path

# 必须用原生平台：offscreen 模式下 Qt 找不到字体，中文全变方块，截图没法验版
os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine
from dsh_app.ui.main_window import MainWindow

# 用自己的临时库，不碰用户真实数据（之前的版本直接开 data/app.db，截图会污染它）
PROBE = Path("data/probe/shot_mastery.db")
PROBE.parent.mkdir(parents=True, exist_ok=True)
for f in (PROBE, PROBE.with_name(PROBE.name + "-wal"), PROBE.with_name(PROBE.name + "-shm")):
    if f.exists():
        f.unlink()
db = DB(PROBE)
st = Settings.load()
st.llm.preload_on_start = False
st.llm.base_url = ""      # 截图不联网：走确定性术语抽取
st.llm.api_key = ""
eng = Engine(db, st)
eng.seed_demo()

app = QApplication.instance() or QApplication([])
win = MainWindow(st, db)
win.resize(1596, 1000)
win.show()
app.processEvents()

# 按标题找页签，别写死编号：加过一页以后编号就错位了
names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
win.tabs.setCurrentIndex(names.index("漏洞地图"))
win.refresh_kp_table()
if win.kp_table.rowCount() > 0:
    win.kp_table.selectRow(0)        # 选中第一行 → 右侧显示它出现在哪些回看点
    app.processEvents()

d = Path("docs/screenshots")
d.mkdir(parents=True, exist_ok=True)
out = d / "17-mastery.png"
win.grab().save(str(out))
print("行数:", win.kp_table.rowCount())
for r in range(min(win.kp_table.rowCount(), 6)):
    cells = [win.kp_table.item(r, c).text() for c in range(5)]
    print("  ", cells)
print("提示:", win.lbl_kp.text())
print("已保存:", out, out.stat().st_size, "字节")

# ---------------------------------------------------------------- 18-mastered
# 同一张表，补两条证据之后的样子：模拟"点了两节里都听懂了" → 掌握度/状态/记忆保持都变
row = next((r for r in range(win.kp_table.rowCount())
            if win.kp_table.item(r, 0).text() == "期望"), 0)
win.kp_table.selectRow(row)
for _ in range(2):
    win._on_kp_mark(True)
app.processEvents()
out2 = d / "18-mastered.png"
win.grab().save(str(out2))
print("补证据后:")
for r in range(min(win.kp_table.rowCount(), 6)):
    print("  ", [win.kp_table.item(r, c).text() for c in range(6)])
print("详情行:", getattr(win, "lbl_kp", None) and win.lbl_kp_detail.text()[:60]
      if hasattr(win, "lbl_kp_detail") else "")
print("状态栏:", win.status.currentMessage()[:70])
print("已保存:", out2, out2.stat().st_size, "字节")