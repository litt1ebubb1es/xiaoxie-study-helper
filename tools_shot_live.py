"""截图：抓**正在运行**的主窗口（PrintWindow），证明改动在真进程里生效。

和 tools_shot_theme.py 的区别：那个是自己建一个窗口渲染（离屏/新进程），
这个是用 Win32 `PrintWindow` 抓用户屏幕上那个真窗口——用来确认"装好了、跑起来了"。

用法：
    ./.venv/Scripts/python.exe tools_shot_live.py            # 抓主窗口
    ./.venv/Scripts/python.exe tools_shot_live.py --list     # 只列出窗口
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

from PySide6.QtGui import QImage

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
user32.SetProcessDPIAware()

OUT_DIR = Path("docs/screenshots")
TITLE_HINTS = ("小蟹", "网课")


def find_window() -> tuple[int, str]:
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if any(h in buf.value for h in TITLE_HINTS):
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else (0, "")


def grab(hwnd: int, path: Path) -> bool:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    # 2 = PW_RENDERFULLCONTENT：不这样抓不到 Qt 的合成内容（会得到全黑）
    ok = user32.PrintWindow(hwnd, mem, 2)

    class BMIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    bmi = BMIH()
    bmi.biSize = ctypes.sizeof(BMIH)
    bmi.biWidth, bmi.biHeight = w, -h          # 负高度 = 自上而下，省一次翻转
    bmi.biPlanes, bmi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    img = QImage(buf, w, h, QImage.Format_RGB32).copy()
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(img.save(str(path)))


def main() -> int:
    hwnd, title = find_window()
    if not hwnd:
        print("没找到运行中的主窗口（先启动 run_app.py）")
        return 1
    print(f"窗口：{title}")
    if "--list" in sys.argv:
        return 0
    out = OUT_DIR / "38-live-window.png"
    ok = grab(hwnd, out)
    print(("已保存 " if ok else "抓取失败 ") + str(out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())