"""测量录制期间的真实内存 / CPU / 磁盘开销。"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time
from pathlib import Path

from dsh_app.capture import LoopbackRecorder, list_loopback_devices, probe_loopback


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem() -> dict:
    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]
    c = PROCESS_MEMORY_COUNTERS()
    c.cb = ctypes.sizeof(c)
    h = k32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
    return {
        "rss_mb": c.WorkingSetSize / 1048576,
        "private_mb": c.PagefileUsage / 1048576,
        "peak_rss_mb": c.PeakWorkingSetSize / 1048576,
    }


def pick_live_device() -> dict | None:
    for d in list_loopback_devices():
        if probe_loopback(int(d["index"]), 0.8).get("ok"):
            return d
    return None


def main() -> None:
    import tempfile

    dev = pick_live_device()
    print("设备:", dev["name"] if dev else "无可用设备")
    if not dev:
        return
    out = Path(tempfile.mkdtemp()) / "bench.wav"

    m0 = mem()
    t0 = time.time()
    c0 = time.process_time()
    rec = LoopbackRecorder(out, device_name=dev["name"], detect_repeats=True)
    rec.start()
    samples = []
    for i in range(24):
        time.sleep(1.0)
        m = mem()
        samples.append((round(time.time() - t0, 1), m["rss_mb"], m["private_mb"],
                        out.stat().st_size if out.exists() else 0))
    wall = time.time() - t0
    cpu = time.process_time() - c0
    rec.stop()
    m1 = mem()

    print(f"\n录制 {wall:.1f}s｜采到音频 {rec.duration:.1f}s｜指纹事件 {len(rec.live_audio)} 缓冲块")
    print(f"内存  起始 RSS {m0['rss_mb']:.1f} MB → 结束 {m1['rss_mb']:.1f} MB "
          f"(增长 {m1['rss_mb'] - m0['rss_mb']:+.1f} MB)")
    print(f"      私有内存 {m0['private_mb']:.1f} → {m1['private_mb']:.1f} MB "
          f"(增长 {m1['private_mb'] - m0['private_mb']:+.1f} MB)")
    print(f"CPU   占用 {cpu / wall * 100:.1f}%（单核百分比）")
    size = out.stat().st_size
    print(f"磁盘  {size / 1048576:.2f} MB / {wall:.0f}s = "
          f"{size / wall / 1024:.0f} KB/s → 45 分钟约 {size / wall * 2700 / 1048576:.0f} MB，"
          f"90 分钟约 {size / wall * 5400 / 1048576:.0f} MB")

    mid = samples[len(samples) // 2:]
    if mid:
        print("\n逐秒 RSS:", " ".join(f"{x[1]:.0f}" for x in mid), "MB（后半段，看是否持续上涨）")
    print("WAV 头:", open(out, "rb").read(44)[:4], "采样率", rec.rate, "声道", rec.channels)


if __name__ == "__main__":
    main()