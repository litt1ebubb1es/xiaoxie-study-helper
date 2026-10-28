"""整屏截图（含弹出菜单、对话框这些独立顶层窗口）。

`tools_shot_live.py` 用 PrintWindow 只拍主窗口，QMenu 是独立的顶层窗口，
拍不到；要看菜单必须整屏抓。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots/39-screen.png")
    app = QGuiApplication(sys.argv[:1])
    screen = app.primaryScreen()
    shot = screen.grabWindow(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    shot.save(str(out))
    print(f"已保存 {out} {shot.width()}x{shot.height()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())