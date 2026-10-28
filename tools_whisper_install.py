"""命令行封装：下载 faster-whisper 模型到本机 data/models。

与界面按钮**共用同一份实现**（dsh_app.whisper_setup.download_model），
避免"命令行能下、界面下不了"这种两套逻辑的老毛病。

用法：python tools_whisper_install.py [small medium ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh_app import whisper_setup as ws  # noqa: E402


def main() -> None:
    sizes = sys.argv[1:] or [ws.DEFAULT_SIZE]
    ok_all = True
    last = ""
    def progress(msg: str) -> None:
        nonlocal last
        if msg != last:
            print(msg, flush=True)
            last = msg

    for s in sizes:
        ok, msg = ws.download_model(s, progress=progress)
        print(f"[{s}] {'OK' if ok else 'FAIL'} {msg}", flush=True)
        ok_all = ok_all and ok
    st = ws.status()
    print("STATUS", {"deps_ok": st["deps_ok"], "installed": st["installed_sizes"],
                     "ready": st["ready"]}, flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()