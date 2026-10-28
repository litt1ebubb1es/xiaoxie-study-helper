"""拍「问模型」小窗的截图（真平台插件，不用 offscreen：offscreen 没有中文字体）。

用法：./.venv/Scripts/python.exe tools_shot_chatwin.py
产物：docs/screenshots/34-chat-window.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)

from PySide6.QtCore import Qt                       # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication          # noqa: E402

from dsh_app.config import Settings                  # noqa: E402
from dsh_app.db import DB                            # noqa: E402
from dsh_app.pipeline import Engine                  # noqa: E402
from dsh_app.ui import theme                         # noqa: E402
from dsh_app.ui.chat_window import ChatWindow        # noqa: E402


def fake_screenshot() -> QImage:
    """画一张像"课件截图"的图：白底 + 一行公式 + 一个高亮框。"""
    img = QImage(420, 200, QImage.Format_RGB32)
    img.fill(QColor("#FFF3EA"))   # 用淡橙底，缩略图才不会看成一白条
    p = QPainter(img)
    p.setPen(QColor("#191C1F"))
    f = p.font()
    f.setPixelSize(19)
    p.setFont(f)
    p.drawText(24, 52, "F(x) = P(X ≤ x)")
    p.setPen(QColor("#5B646E"))
    p.drawText(24, 96, "右连续；左极限 = P(X < x)")
    p.setPen(QColor("#F2621B"))
    p.drawRect(18, 26, 250, 44)
    p.fillRect(0, 0, 420, 12, QColor("#F2621B"))
    p.end()
    return img


def main() -> None:
    app = QApplication.instance() or QApplication([])
    theme.apply_font(app)
    settings = Settings.load()
    probe = ROOT / "data" / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    db = DB(probe / "shot_chatwin.db")
    eng = Engine(db, settings)

    msgs = [
        {"role": "user", "content": "这页的右连续到底在说什么？"},
        {"role": "assistant",
         "content": "「右连续」是说从右边逼近 x 时极限等于 F(x)。\n"
                    "因为定义里写的是 P(X≤x)，端点那一点的概率质量被算进去了，"
                    "所以左边可能跳一下，右边必须接得上。",
         "sources": [{"doc": "概率论讲义.pdf", "source_ref": "第2章 · 第 14 页",
                      "score": "0.82", "snippet": "分布函数右连续，左极限为 P(X<x)。"}]},
    ]
    win = ChatWindow(eng, settings, msgs, session_id="", on_close=None)
    win.chk_top.setChecked(True)          # 展示"置顶"打开的样子
    win.add_image(fake_screenshot(), "课件截图.png")
    win.lbl_state.setText("已粘贴一张截图，直接点发送即可")
    win.resize(560, 660)
    win.show()
    for _ in range(12):
        app.processEvents()

    out = ROOT / "docs" / "screenshots" / "34-chat-window.png"
    win.grab().save(str(out))
    print(f"问模型小窗 → {out.name} {out.stat().st_size} 字节")
    win.close()


if __name__ == "__main__":
    main()