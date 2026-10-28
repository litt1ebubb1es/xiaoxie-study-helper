"""截图：橙白主题下每个页签的真实渲染（用真平台插件，字体才正常）。

离屏（offscreen）下 Qt 找不到系统字体，中文会变成豆腐块，所以这里不能用 QT_QPA_PLATFORM=offscreen。
"""
import sys
import time
from pathlib import Path

import os
os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine
from dsh_app.ui import theme
from dsh_app.ui.main_window import MainWindow, STYLE, _diagnosis_html

PROBE = Path("data/probe/shot_theme.db")
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
cid = db.get_session(sid)["course_id"]

app = QApplication.instance() or QApplication([])
theme.apply_font(app)
app.setStyleSheet(STYLE)
win = MainWindow(st, db)
win.resize(1560, 980)
win.show()
for _ in range(6):
    app.processEvents()

win.current_session = sid
win._load_session_view()
win.timeline.set_playhead(95)
if win.current_knots:
    win.timeline.set_selected(win.current_knots[0].id)
app.processEvents()

# 造一点"有内容"的状态：一条答对的小测、几条待答问题
kps = db.get_knowledge_points(cid)
if kps:
    e = eng.record_kp_attempt(int(kps[0]["id"]), 1.0, source="quiz")
    print("补一条证据:", e["name"], f"{int(e['mastery']*100)}%")
db.add_question(sid, "为什么分布函数要右连续？", t_sec=88.0)
db.add_question(sid, "泊松分布的参数和期望是同一个数吗？", t_sec=210.0)
win.refresh_kp_table()
win.refresh_quiz_hint()

d = Path("docs/screenshots")
d.mkdir(parents=True, exist_ok=True)
names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
print("页签:", names)

targets = {
    "时间轴与回看点": "28-theme-timeline.png",
    "划段重讲": "29-theme-reword.png",
    "小测与待答": "30-theme-quiz.png",
    "漏洞地图": "31-theme-map.png",
    "问模型": "32-theme-chat.png",
    "设置与模型部署": "33-theme-settings.png",
}
for tab, fn in targets.items():
    if tab not in names:
        print("跳过（没有这个页签）:", tab)
        continue
    win.tabs.setCurrentIndex(names.index(tab))
    for _ in range(6):
        app.processEvents()
    time.sleep(0.15)
    out = d / fn
    win.grab().save(str(out))
    print(f"  {tab} → {fn} {out.stat().st_size} 字节")

# 对话页塞两轮示例，好看清气泡配色
try:
    win.tabs.setCurrentIndex(names.index("问模型"))
    win._chat_msgs.clear()
    win._append_chat("user", "分布函数为什么要右连续？")
    win._append_chat(
        "assistant",
        "因为 F(x)=P(X≤x) 里的「≤」把端点算进去了：\n"
        "从右边逼近 x 时极限等于 F(x)，从左边逼近会漏掉那一点的概率质量。",
        sources=[{"doc": "概率论讲义.pdf", "section": "第2章", "score": "0.82",
                  "snippet": "分布函数右连续，左极限为 P(X<x)。", "ord": 1,
                  "source_ref": "第2章 · 第 14 页"}])
    for _ in range(8):
        app.processEvents()
    out = d / "32-theme-chat.png"
    win.grab().save(str(out))
    print(f"  问模型（有对话）→ {out.name} {out.stat().st_size} 字节")
except Exception as e:                             # noqa: BLE001
    print("对话页截图跳过:", e)

# 设置页拉到存储/数据两块：P0 的"音频占了多少、删了不伤心"就在这里
try:
    win.tabs.setCurrentIndex(names.index("设置与模型部署"))
    for _ in range(6):
        app.processEvents()
    bar = win._settings_scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    for _ in range(8):
        app.processEvents()
    time.sleep(0.15)
    out = d / "36-storage.png"
    win.grab().save(str(out))
    print(f"  设置页（存储/数据）→ {out.name} {out.stat().st_size} 字节")
except Exception as e:                             # noqa: BLE001
    print("存储截图跳过:", e)

# 放大后的时间轴：3 小时的课挤一屏时，回看点只有几像素宽
try:
    win.tabs.setCurrentIndex(names.index("时间轴与回看点"))
    for _ in range(4):
        app.processEvents()
    peak = max((k.t_end for k in win.current_knots), default=0.0)
    win.timeline.zoom_at(3.0, peak / 2 if peak else 60.0)
    win._on_view_changed(*win.timeline.view_range())
    for _ in range(6):
        app.processEvents()
    time.sleep(0.15)
    out = d / "37-timeline-zoom.png"
    win.grab().save(str(out))
    print(f"  时间轴（放大）→ {out.name} {out.stat().st_size} 字节")
except Exception as e:                             # noqa: BLE001
    print("缩放截图跳过:", e)

win.close()
print("完成")