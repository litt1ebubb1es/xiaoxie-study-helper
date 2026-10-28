"""生成 QSS 用的勾选标记（QCheckBox 选中态的白色对勾）。

为什么要有这个脚本：QSS 不能画图标，`::indicator:checked` 只能"填一块颜色"。
之前选中态就是一个实心橙方块，看不出"勾上了没有"。Qt 支持 `image: url(...)`，
所以这里把对勾渲染成 PNG 存成资源，主题里引用它。

只在改配色/尺寸时重跑一次；运行时不生成、不依赖 QApplication。

    ./.venv/Scripts/python.exe tools_make_check_icon.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QPointF, Qt                    # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

SIZE = 16
OUT = Path(__file__).resolve().parent / "dsh_app" / "ui" / "assets" / "check.png"


def main() -> int:
    app = QApplication.instance() or QApplication([])      # noqa: F841
    img = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(255, 255, 255))
    pen.setWidthF(1.9)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    # 对勾：左下 → 中下 → 右上，坐标按 16px 方格调过
    p.drawPolyline([QPointF(3.6, 8.4), QPointF(6.6, 11.4), QPointF(12.4, 4.9)])
    p.end()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not img.save(str(OUT)):
        print("保存失败:", OUT)
        return 1
    print(f"已写出 {OUT} ({OUT.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())