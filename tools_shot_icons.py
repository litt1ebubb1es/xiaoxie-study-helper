"""把图标集排成一张对照图，方便看"整套图形是不是一个体系"。

用法：./.venv/Scripts/python.exe tools_shot_icons.py
产物：docs/screenshots/35-icons.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)

from PySide6.QtCore import Qt                                # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter     # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

from dsh_app.ui import icons, theme                           # noqa: E402


def main() -> None:
    app = QApplication.instance() or QApplication([])
    theme.apply_font(app)
    names = icons.names()
    cols, cell, pad = 6, 104, 16
    rows = (len(names) + cols - 1) // cols
    img = QImage(cols * cell + pad * 2, rows * cell + pad * 2 + 44, QImage.Format_RGB32)
    img.fill(QColor(theme.SURFACE))
    p = QPainter(img)
    f = QFont()
    f.setPixelSize(13)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(theme.TEXT))
    p.drawText(pad, pad + 14, f"图标集 {len(names)} 个（24×24 网格，线宽 1.7）")
    small = QFont()
    small.setPixelSize(11)
    p.setFont(small)
    for i, n in enumerate(names):
        x = pad + (i % cols) * cell
        y = pad + 40 + (i // cols) * cell
        pm = icons.get(n, theme.TEXT, 30).pixmap(30, 30)
        p.drawPixmap(x + 33, y + 8, pm)
        p.setPen(QColor(theme.TEXT_3))
        p.drawText(x, y + 60, cell, 16, Qt.AlignHCenter, n)
    p.end()
    out = ROOT / "docs" / "screenshots" / "35-icons.png"
    img.save(str(out))
    print(f"图标对照图 → {out.name} {out.stat().st_size} 字节")


if __name__ == "__main__":
    main()