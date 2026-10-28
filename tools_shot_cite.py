"""截图：问模型页的引用带页码/时间码（不发模型请求，直接渲染检索结果）。"""
import os
import sys
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)   # offscreen 无字体，中文会变方块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from dsh_app import chat
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine
from dsh_app.ui.main_window import MainWindow

probe = Path("data/probe")
db = DB(probe / "chat_shot.db")
eng = Engine(db, Settings.load())
for f in ("notes3.pdf", "deck.pptx", "笔记.md"):
    p = probe / f
    if p.exists():
        r = eng.kb_import(str(p))
        print(f"导入 {f}: chunks={r.get('chunks')} located={r.get('located')}")

app = QApplication.instance() or QApplication([])
st = Settings.load()
st.llm.preload_on_start = False          # 截图不需要预热云端模型
win = MainWindow(st, db)
win.resize(1596, 1000)
win.show()
app.processEvents()
import time
# 注意：time.sleep 会阻塞事件循环，QTimer 永远不触发，必须边等边处理事件
for _ in range(20):
    app.processEvents()
    time.sleep(0.05)

win.tabs.setCurrentIndex(3)      # 问模型
hits = chat.kb_search(db, "秩 线性无关 特征值", k=4)
print("检索命中:", [(h["doc"], h["label"]) for h in hits])
win._append_chat("user", "秩和线性无关是什么关系？")
win._append_chat("assistant",
                 "秩就是线性无关列的最大个数。换句话说，把矩阵的列看成一组向量，"
                 "能挑出多少个互相独立的就是秩。", hits)
app.processEvents()

out = Path("docs/screenshots/19-kb-cite.png")
win.grab().save(str(out))
print("已保存:", out, out.stat().st_size, "字节")