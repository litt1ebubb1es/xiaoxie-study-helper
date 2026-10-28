"""把「设置页底部」的两块新东西单独拍下来：自动抓取体检 + 知识库自动归档。

为什么单独写：真机窗口里滚到这两块要精确点击，盲点会误碰其它控件
（之前就误点到「按记录调阈值」把阈值写进了配置）。这里直接构造界面、
把滚动条推到底、grab 出来，不碰任何真实状态。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if "--show" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from dsh_app.config import Settings  # noqa: E402
from dsh_app.db import DB  # noqa: E402
from dsh_app.ui.main_window import MainWindow  # noqa: E402

OUT = ROOT / "data" / "probe" / "settings"
PAGES = 5          # 设置页在最后


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.db").unlink(missing_ok=True)      # 每次从干净库开始，数字才可比
    app = QApplication(sys.argv[:1])
    db = DB(OUT / "probe.db")
    win = MainWindow(Settings.load(), db)
    win.resize(1456, 939)
    # 造两条丢弃记录 + 一节看过的课，让两块都有内容可看
    db.add_capture_discard("msedge", "哔哩哔哩·网课", "首页自动播放", 6.9, 8.0, "too_short")
    db.add_capture_discard("msedge", "哔哩哔哩·网课", "广告前奏", 11.8, 8.0, "too_short")
    cid = db.ensure_course("哔哩哔哩·网课")
    db.create_session(cid, "第一章：随机事件及其概率（二）", audio_path="a.flac",
                      source="webvideo")
    sid = db.list_sessions()[0]["id"]
    db.update_session(sid, duration=164.0)
    db.replace_segments(sid, [
        {"t_start": float(i * 25), "t_end": float(i * 25 + 25),
         "text": "样本空间的划分、条件概率、全概率公式与贝叶斯公式的推导过程。"}
        for i in range(7)])
    win.engine.kb_ingest_session(sid)
    win._refresh_discard_info()
    win._refresh_kb_auto_info()

    win.tabs.setCurrentIndex(PAGES)
    app.processEvents()
    win._settings_scroll.verticalScrollBar().setValue(
        win._settings_scroll.verticalScrollBar().maximum())
    for _ in range(30):
        app.processEvents()
    win._settings_scroll.verticalScrollBar().setValue(
        win._settings_scroll.verticalScrollBar().maximum())
    app.processEvents()

    shot = OUT / "settings-bottom.png"
    win._settings_scroll.widget().grab().save(str(shot))
    print("丢弃提示:", win.lbl_discard.text())
    print("归档提示:", win.lbl_kb_auto.text())
    print("截图:", shot)
    ok = ("2 次" in win.lbl_discard.text() and "已归档 1 节" in win.lbl_kb_auto.text())
    print("RESULT:", "PASS" if ok else "FAIL")
    win.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())