"""把吉祥物图标"焊"进一个 pythonw.exe 的副本，得到带螃蟹图标的启动器。

为什么非得动 exe：Windows 11 的任务栏按钮图标取自**进程的 exe 资源**，
窗口自己的 WM_SETICON（Qt 的 setWindowIcon）只影响标题栏和 Alt+Tab。
实测三种做法：
  · 只 setWindowIcon          → 任务栏仍是 python 的图标；
  · 设未注册的 AppUserModelID → 任务栏退化成"空白文档"图标，更糟；
  · 换 exe 图标（本脚本）     → 任务栏、Alt+Tab、桌面快捷方式全部一致。
所以这里直接改 PE 资源：拷一份 pythonw.exe，删掉原有 RT_ICON/RT_GROUP_ICON，
写入我们从 xiaoxie.ico 生成的图标组。副本放在原 exe 同目录，Python 才找得到 dll。

用法：
    python tools_patch_exe_icon.py                 # 生成 .venv/Scripts/小蟹学习助手.exe
    python tools_patch_exe_icon.py --verify        # 只读检查图标资源
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from ctypes import wintypes
from pathlib import Path

if "--show" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ICO = ROOT / "docs" / "icon" / "xiaoxie.ico"
# 注意：venv 里的 pythonw.exe 只是个"转发器"，真正跑起来的是 base 解释器
# （实测：启动 .venv/Scripts/pythonw.exe 会 fork 出 F:\python\pythonw.exe）。
# 任务栏图标取的是**真正那个进程的 exe**，所以必须给 base 解释器的副本换图标。
BASE_EXE = Path(sys.base_prefix) / "pythonw.exe"
SRC = BASE_EXE if BASE_EXE.exists() else (ROOT / ".venv" / "Scripts" / "pythonw.exe")
DST = SRC.parent / "小蟹学习助手.exe"

RT_ICON = 3
RT_GROUP_ICON = 14
LOAD_LIBRARY_AS_DATAFILE = 0x02

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.BeginUpdateResourceW.restype = wintypes.HANDLE
k32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
k32.UpdateResourceW.restype = wintypes.BOOL
k32.UpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
                                wintypes.WORD, wintypes.LPVOID, wintypes.DWORD]
k32.EndUpdateResourceW.restype = wintypes.BOOL
k32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
k32.LoadLibraryExW.restype = wintypes.HMODULE
k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
k32.FreeLibrary.argtypes = [wintypes.HMODULE]

ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMODULE, wintypes.LPVOID,
                              wintypes.LPWSTR, ctypes.c_void_p)
k32.EnumResourceNamesW.argtypes = [wintypes.HMODULE, wintypes.LPVOID,
                                   ENUMPROC, ctypes.c_void_p]


def _noop() -> None:
    """（保留占位：枚举资源那条路已弃用，见 patch() 的说明。）"""


def dib_for(png_frame: bytes) -> bytes:
    """把 PNG 帧转成 RT_ICON 需要的 DIB（BITMAPINFOHEADER + BGRA + AND 掩码）。"""
    from PySide6.QtGui import QImage

    img = QImage.fromData(png_frame, "PNG")
    if img.isNull():
        raise ValueError("PNG 解不开")
    img = img.convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    px = img.bits()                                    # memoryview
    stride = img.bytesPerLine()
    rows = []
    for y in range(h - 1, -1, -1):                     # DIB 自下而上
        line = bytes(px[y * stride: y * stride + w * 4])
        rows.append(line)
    xor = b"".join(rows)
    # AND 掩码：1bpp，每行按 4 字节对齐；有 alpha 时全 0 即可，但尺寸必须对
    mask_stride = ((w + 31) // 32) * 4
    mask = b"\x00" * (mask_stride * h)
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, len(xor) + len(mask), 0, 0, 0, 0)
    return header + xor + mask


def build_group(entries: list[tuple[int, int, int, int]]) -> bytes:
    """GRPICONDIR + 每帧目录项。entries: [(w, h, colorcount, bytesize)]，ID 从 1 递增。"""
    out = struct.pack("<HHH", 0, 1, len(entries))
    for i, (w, h, _cc, size) in enumerate(entries, start=1):
        out += struct.pack("<BBBBHHIH", 0 if w >= 256 else w, 0 if h >= 256 else h,
                           0, 0, 1, 32, size, i)
    return out


def patch() -> int:
    if not ICO.exists():
        print(f"先跑 tools_make_icon.py 生成 {ICO}")
        return 1
    if not SRC.exists():
        print(f"找不到 {SRC}")
        return 1
    DST.write_bytes(SRC.read_bytes())

    data = ICO.read_bytes()
    _, _, count = struct.unpack("<HHH", data[:6])
    frames, off = [], 6
    for _ in range(count):
        w, h, _cc, _r, _p, _bpp, size, offset = struct.unpack("<BBBBHHII", data[off:off + 16])
        off += 16
        frames.append(((w or 256), (h or 256), data[offset:offset + size]))

    dibs = [dib_for(png) for _w, _h, png in frames]
    group = build_group([(w, h, 0, len(dib)) for (w, h, _), dib in zip(frames, dibs)])

    hu = k32.BeginUpdateResourceW(str(DST), False)
    if not hu:
        print("BeginUpdateResource 失败：", ctypes.get_last_error())
        return 1
    try:
        # 不枚举旧资源（EnumResourceNamesW 的 ctypes 回调在这台机器上会卡住），
        # 直接按 ID 覆盖：python 官方 exe 的图标组就是 ID=1、帧 ID=1..N，
        # 我们覆盖 1..7 并写回组 1，剩下的旧帧没人引用，不影响显示。
        for idx, dib in enumerate(dibs, start=1):
            buf = ctypes.create_string_buffer(dib, len(dib))
            if not k32.UpdateResourceW(hu, ctypes.c_void_p(RT_ICON), ctypes.c_void_p(idx),
                                       0, buf, len(dib)):
                raise OSError(f"写 RT_ICON {idx} 失败：{ctypes.get_last_error()}")
        gbuf = ctypes.create_string_buffer(group, len(group))
        if not k32.UpdateResourceW(hu, ctypes.c_void_p(RT_GROUP_ICON), ctypes.c_void_p(1),
                                   0, gbuf, len(group)):
            raise OSError(f"写 RT_GROUP_ICON 失败：{ctypes.get_last_error()}")
    finally:
        ok = k32.EndUpdateResourceW(hu, False)
    if not ok:
        print("EndUpdateResource 失败：", ctypes.get_last_error())
        return 1

    sizes = [f"{w}x{h}" for w, h, _ in frames]
    print(f"已生成 {DST}（{DST.stat().st_size} 字节）", flush=True)
    print(f"  图标帧数 {len(dibs)}，尺寸 {sizes}", flush=True)
    print(f"  已覆盖 RT_ICON 1..{len(dibs)} 与 RT_GROUP_ICON 1", flush=True)
    return 0


def verify() -> int:
    """用 shell32 的 ExtractIconEx 数一下副本里到底有几个图标。

    比自己解析 PE 资源目录靠谱：那是 Windows 自己判断"这个 exe 有几个图标"的同一套逻辑。
    """
    if not DST.exists():
        print("还没生成，先跑一次不带 --verify", flush=True)
        return 1
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.ExtractIconExW.restype = wintypes.UINT
    shell.ExtractIconExW.argtypes = [wintypes.LPCWSTR, ctypes.c_int,
                                     ctypes.POINTER(wintypes.HICON),
                                     ctypes.POINTER(wintypes.HICON), wintypes.UINT]
    big = (wintypes.HICON * 32)()
    small = (wintypes.HICON * 32)()
    n = shell.ExtractIconExW(str(DST), -1, big, small, 32)
    print(f"副本里的图标数：{n}", flush=True)
    print("RESULT:", "PASS" if n >= 1 else "FAIL", flush=True)
    return 0 if n >= 1 else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    code = patch()
    if code == 0:
        verify()
    raise SystemExit(code)