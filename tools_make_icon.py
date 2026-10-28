"""把吉祥物导出成 Windows 图标文件（.ico），供任务栏 / 快捷方式使用。

为什么要自己拼 ICO：Qt 的图片插件能读 .ico，但写不一定带对应插件（本机实测没有），
而 Windows 图标格式其实很简单——头 + 目录 + 若干 PNG 帧。Vista 之后允许目录项
直接内嵌 PNG，所以这里用 QPixmap 渲染每一档尺寸，再按规范拼容器。

用法：
    python tools_make_icon.py                # 写 docs/icon/xiaoxie.ico
    python tools_make_icon.py --sizes 16,32,256
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

if "--show" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT_DIR = ROOT / "docs" / "icon"


def png_bytes(pixmap) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    return bytes(buf.data())


def build_ico(frames: dict[int, bytes]) -> bytes:
    """frames: {边长: PNG 字节} → ICO 容器。"""
    entries, blobs = b"", b""
    offset = 6 + 16 * len(frames)
    for size in sorted(frames):
        data = frames[size]
        w = 0 if size >= 256 else size          # 0 表示 256
        entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return struct.pack("<HHH", 0, 1, len(frames)) + entries + blobs


def main() -> int:
    app = QApplication(sys.argv[:1])                     # noqa: F841  需要 QGuiApplication
    from dsh_app.ui import icons

    sizes = SIZES
    for arg in sys.argv:
        if arg.startswith("--sizes"):
            raw = arg.split("=", 1)[1] if "=" in arg else ""
            if raw:
                sizes = tuple(int(x) for x in raw.split(",") if x.strip())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {s: png_bytes(icons.app_badge(s)) for s in sizes}
    ico = build_ico(frames)
    out = OUT_DIR / "xiaoxie.ico"
    out.write_bytes(ico)

    # 顺手留一张 256 的 PNG，README/文档里直接用
    icons.app_badge(256).save(str(OUT_DIR / "xiaoxie-256.png"))
    print(f"写好了 {out}（{len(ico)} 字节，含 {len(sizes)} 档：{sorted(sizes)}）")
    print(f"预览图 {OUT_DIR / 'xiaoxie-256.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())