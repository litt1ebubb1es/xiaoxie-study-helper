"""在**正在运行**的主窗口里点一个坐标，用来给真实交互留下证据。

为什么不用 Qt 测试：离屏测试能证明信号连对了，但证明不了"真窗口里点得到"。
这个脚本用 Win32 把窗口提到前台，把鼠标移到坐标上点一下，然后再抓一张图。

坐标是相对**窗口左上角**的像素（和 tools_shot_live.py 抓出来的图一一对应）。

用法：
    python tools_live_click.py 334 882              # 点一下
    python tools_live_click.py 334 882 --wait 3     # 点完等 3 秒再截图
    python tools_live_click.py --dump               # 只打印窗口位置
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

TITLE_HINTS = ("小蟹", "网课")
OUT = Path("docs/screenshots/38-live-window.png")

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


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


def rect_of(hwnd: int) -> tuple[int, int, int, int]:
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def main() -> int:
    hwnd, title = find_window()
    if not hwnd:
        print("没找到运行中的主窗口")
        return 1
    x, y, w, h = rect_of(hwnd)
    print(f"窗口 {title!r} @ ({x},{y}) {w}x{h}")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "--dump" in sys.argv:
        return 0
    dx, dy = int(args[0]), int(args[1])
    wait = 3.0
    if "--wait" in sys.argv:
        wait = float(sys.argv[sys.argv.index("--wait") + 1])

    user32.ShowWindow(hwnd, 9)              # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.35)
    # 客户区起点和窗口边框有差：这里按整窗坐标算，和截图一致
    sx, sy = x + dx, y + dy
    user32.SetCursorPos(sx, sy)
    time.sleep(0.15)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.06)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    print(f"点击 ({sx},{sy}) = 窗口内 ({dx},{dy})")
    time.sleep(wait)

    from PySide6.QtGui import QGuiApplication, QImage  # 只为存图

    app = QGuiApplication.instance() or QGuiApplication([])
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 2)

    class BMIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    bmi = BMIH()
    bmi.biSize = ctypes.sizeof(BMIH)
    bmi.biWidth, bmi.biHeight = w, -h
    bmi.biPlanes, bmi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    QImage(buf, w, h, QImage.Format_RGB32).copy().save(str(OUT))
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    print("已保存", OUT)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())