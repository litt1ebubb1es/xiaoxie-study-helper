"""截图：网页视频自动录（勾上后的待命状态 + 探测结果）。

真的会起监听线程（它只是看音频会话，不会开始录），所以不能在这台机器上放视频时跑。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from dsh_app import webvideo as wv
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine
from dsh_app.ui.main_window import MainWindow

PROBE = Path("data/probe/shot_webvideo.db")
PROBE.parent.mkdir(parents=True, exist_ok=True)
for f in (PROBE, PROBE.with_name(PROBE.name + "-wal"), PROBE.with_name(PROBE.name + "-shm")):
    if f.exists():
        f.unlink()

db = DB(PROBE)
st = Settings.load()
st.llm.preload_on_start = False
st.llm.base_url = ""
st.llm.api_key = ""
st.capture.auto_web_capture = False
eng = Engine(db, st)
sid = eng.seed_demo()

app = QApplication.instance() or QApplication([])
win = MainWindow(st, db)
win.resize(1596, 1000)
win.show()
app.processEvents()
win.current_session = sid
win._load_session_view()

win.chk_autoweb.setChecked(True)          # 起监听
for _ in range(12):
    app.processEvents()
    time.sleep(0.05)

print("监听着吗:", bool(win._web_watch and win._web_watch.running))
print("状态标签:", win.lbl_rec.text())
print("状态栏:", win.status.currentMessage())
print("会话见到的浏览器:", [s.proc for s in wv.list_sessions()])

d = Path("docs/screenshots")
out = d / "27-webvideo-auto.png"
win.grab().save(str(out))
print("已保存:", out, out.stat().st_size, "字节")

# 模拟"检测到浏览器在放视频"这一瞬间的提示（用假 meta，不动真录音）
win._web_on_state({"state": "arming"})
app.processEvents()
print("arming 提示:", win.lbl_rec.text())

win.chk_autoweb.setChecked(False)
app.processEvents()
win.close()
print("收尾完成")