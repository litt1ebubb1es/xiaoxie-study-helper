"""诊断：逐个 WASAPI 环回设备试读，判断谁真能出数据。"""
from __future__ import annotations

import queue
import threading
import time

import pyaudiowpatch as pyaudio


def try_device(p: pyaudio.PyAudio, dev: dict, seconds: float = 1.2) -> dict:
    ch = min(2, int(dev["maxInputChannels"]))
    rate = int(dev["defaultSampleRate"])
    block = max(256, int(rate * 0.04))
    info = {"name": dev["name"][:34], "idx": dev["index"], "ch": ch, "rate": rate}
    try:
        st = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                    input_device_index=int(dev["index"]), frames_per_buffer=block)
    except Exception as e:
        info.update(ok=False, err=f"{type(e).__name__}: {e}")
        return info

    q: queue.Queue = queue.Queue()

    def reader():
        n = 0
        while True:
            try:
                raw = st.read(block, exception_on_overflow=False)
            except Exception as e:  # noqa: BLE001
                q.put(("err", f"{type(e).__name__}: {e}"))
                return
            n += len(raw)
            if n > 0 and q.empty():
                q.put(("data", n))
            if n > block * ch * 2 * 5:
                return

    th = threading.Thread(target=reader, daemon=True)
    t0 = time.time()
    th.start()
    got = None
    try:
        got = q.get(timeout=seconds)
    except queue.Empty:
        got = ("timeout", 0)
    el = time.time() - t0
    info.update(elapsed=round(el, 2), result=got[0], bytes=got[1])

    # 先让读线程停下，再关流
    try:
        st.stop_stream()
    except Exception:
        pass
    time.sleep(0.05)
    try:
        st.close()
    except Exception:
        pass
    info["ok"] = got[0] == "data"
    return info


def main() -> None:
    p = pyaudio.PyAudio()
    try:
        try:
            dflt = p.get_default_wasapi_loopback()
            print("默认环回:", dflt["name"], "idx", dflt["index"])
        except Exception as e:
            dflt = None
            print("默认环回取不到:", e)
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            out = p.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
            print("默认输出:", out["name"], "idx", out["index"])
        except Exception as e:
            print("默认输出取不到:", e)

        print("\n逐设备试读：")
        for dev in p.get_loopback_device_info_generator():
            r = try_device(p, dev)
            mark = "OK  " if r["ok"] else "FAIL"
            print(f"  [{mark}] idx={r['idx']:3d} {r['name']:36s} {r['rate']}Hz "
                  f"ch={r['ch']} {r['result']:8s} bytes={r.get('bytes', 0):6d} "
                  f"{r.get('err', '')}")
    finally:
        p.terminate()


if __name__ == "__main__":
    main()