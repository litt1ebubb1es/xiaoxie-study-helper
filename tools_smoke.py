"""端到端冒烟测试：不用任何模型，验证采集 → 转写 → 分析 → 诊断的真实链路。

1. WASAPI 系统回环录 3 秒（不需要有声音在播，验证设备打开/写入/关闭）
2. 用 ffmpeg 造一个带音轨的 mp4，验证"视频导入 → 16k 单声道 wav"
3. 用侧车字幕验证真实转写路径（有 srt 时不依赖任何在线模型）
4. 音频指纹：喂一段"重复内容"的合成音频，验证回退/重复播放能被推断出来
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# 脚本就在仓库根目录：parent 就是仓库根（之前多退了一层，产物被写到盘符根目录）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dsh_app import capture as cap  # noqa: E402
from dsh_app.config import Settings, ensure_dirs  # noqa: E402
from dsh_app.db import DB  # noqa: E402
from dsh_app.pipeline import Engine  # noqa: E402

OUT = ROOT / "data" / "smoke"
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> int:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. 回环录音 --------------------------------------------------
    # 本机默认输出是蓝牙耳机：没在播放时环回设备不出数据（这是正常物理现象）。
    # 所以这里先找一个"当前真的在出数据"的设备做录音断言，再单独验证
    # "无数据设备不会把录音线程焊死"。
    devs = cap.list_loopback_devices()
    check("枚举回环设备", len(devs) > 0, f"{len(devs)} 个，例如 {devs[0]['name'][:28] if devs else '-'}")

    live = None
    for d in devs:
        pr = cap.probe_loopback(int(d["index"]), 0.8)
        if pr.get("ok"):
            live = d
            break
    if live is None:
        check("回环录音 2 秒", True, "跳过：当前没有在出数据的环回设备（蓝牙未播放属正常）")
    else:
        wav = OUT / "rec.wav"
        rec = cap.LoopbackRecorder(wav, device_name=live["name"], detect_repeats=True)
        try:
            rec.start()
            time.sleep(2.0)
            rec.stop()
            dur = cap.audio_duration(wav)
            check("回环录音 2 秒", wav.exists() and dur > 1.0,
                  f"{dur:.2f}s，rate={rec.rate}，设备「{rec.device_label[:22]}」")
        except Exception as e:
            check("回环录音 2 秒", False, f"{type(e).__name__}: {e}")

    # 无数据设备：必须 2 秒内正常返回（不卡死、不抛异常、idle 在涨）
    try:
        silent = None
        for d in devs:
            if live is None or d["index"] != live["index"]:
                pr = cap.probe_loopback(int(d["index"]), 0.6)
                if not pr.get("ok"):
                    silent = d
                    break
        if silent is None:
            check("无音频流设备不卡死", True, "跳过：所有设备都在出数据")
        else:
            t0 = time.time()
            r2 = cap.LoopbackRecorder(OUT / "silent.wav", device_name=silent["name"])
            r2.start()
            time.sleep(1.6)
            r2.stop()
            el = time.time() - t0
            check("无音频流设备不卡死",
                  el < 3.0 and r2.duration == 0 and not r2.receiving and r2.idle_seconds > 1.0,
                  f"{el:.1f}s 返回，duration={r2.duration:.1f}s idle={r2.idle_seconds:.1f}s "
                  f"设备「{silent['name'][:18]}」")
    except Exception as e:
        check("无音频流设备不卡死", False, f"{type(e).__name__}: {e}")

    # ---- 2. 媒体导入（视频 → 16k wav）--------------------------------
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        mp4 = OUT / "lesson.mp4"
        cmd = [exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=10:d=12",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
               "-shortest", "-ac", "2", str(mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        check("生成测试 mp4", mp4.exists() and mp4.stat().st_size > 1000,
              f"{mp4.stat().st_size // 1024} KB" if mp4.exists() else r.stderr[-160:])
        dst = OUT / "lesson16k.wav"
        cap.import_media(mp4, dst)
        check("mp4 → 16k 单声道 wav", dst.exists() and cap.audio_duration(dst) > 10,
              f"{cap.audio_duration(dst):.1f}s")
        audio = cap.read_wav_mono16k(dst)
        check("读取为 numpy 单声道", audio.ndim == 1 and audio.size > 16000 * 10, f"{audio.size} 采样")
    except Exception as e:
        check("媒体导入", False, f"{type(e).__name__}: {e}")

    # ---- 3. 侧车字幕 → 真实转写路径 ----------------------------------
    try:
        srt = OUT / "lesson16k.srt"
        lines = []
        t = 0.0
        texts = [
            "这节课我们来看随机变量的定义。", "定义有两个必要条件，缺一不可。",
            "很多人会把充分条件和必要条件搞混。", "接下来做一个简短的推导，两边同时取对数。",
            "这里有一个特别容易出错的地方，梯度为零并不意味着最优解。",
            "还要检查边界条件，这是一个反例。", "我们用一个具体数字跑一遍，初始值零点一。",
            "迭代十次收敛到零点八七，误差量级百分之一。", "小结一下，把定义和推导分开理解。",
        ]
        for i, tx in enumerate(texts):
            a, b = t, t + 1.2
            lines.append(f"{i+1}\n{_ts(a)} --> {_ts(b)}\n{tx}\n")
            t = b + 0.4
        srt.write_text("\n".join(lines), encoding="utf-8")

        settings = Settings()
        settings.asr.backend = "sidecar"
        db = DB(OUT / "smoke.db")
        eng = Engine(db, settings)
        cid = db.ensure_course("冒烟课程", subject="数学")
        sid = db.create_session(cid, "第 1 讲", str(OUT / "lesson16k.wav"), source="file")
        res, note = eng.transcribe(sid, OUT / "lesson16k.wav")
        check("侧车字幕转写", res.is_real and len(res.segments) >= 8,
              f"{res.backend}，{len(res.segments)} 段｜{note[:40]}")
        db.add_event(sid, 3.0, "seek_back", {"distance_s": 35}, 1.0, "manual")
        db.add_event(sid, 6.5, "manual_mark", {"note": "冒烟"}, 1.0, "manual")
        ana = eng.analyze_session(sid)
        check("分析产出窗口与回看点", len(ana.windows) > 0 and len(ana.knots) > 0,
              f"{len(ana.windows)} 窗口 / {len(ana.knots)} 回看点")
        if ana.knots:
            k = ana.knots[0]
            check("回看点带可解释依据", bool(k.evidence), k.evidence[:48])
        d = eng.diagnosis(sid)
        check("漏洞地图生成", d.get("n_knots", 0) > 0, f"plan={len(d['plan'])}")
        db.close()
    except Exception as e:
        check("侧车字幕转写", False, f"{type(e).__name__}: {e}")

    # ---- 4. 音频指纹 --------------------------------------------------
    try:
        det = cap.AudioRepeatDetector(rate=16000)
        sr = 16000
        rng = np.random.default_rng(7)
        # 8 秒 A，8 秒 B，再重复 A → 应在重复处命中
        toneA = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 8) / sr)).astype(np.float32)
        toneB = (0.3 * np.sin(2 * np.pi * 900 * np.arange(sr * 8) / sr)).astype(np.float32)
        events = []
        for block, label in ((toneA, "A1"), (toneB, "B"), (toneA, "A2")):
            for i in range(0, len(block), sr // 2):
                events += det.feed(block[i:i + sr // 2])
        check("指纹检出重复段落", any(e["kind"] in ("replay", "seek_back") for e in events),
              f"{len(events)} 个事件：{[e['kind'] for e in events[:4]]}")
        det2 = cap.AudioRepeatDetector(rate=16000)
        noise = rng.normal(0, 0.1, sr * 12).astype(np.float32)
        ev2 = []
        for i in range(0, len(noise), sr // 2):
            ev2 += det2.feed(noise[i:i + sr // 2])
        check("指纹不误报随机噪声", len(ev2) == 0, f"{len(ev2)} 个误报")
    except Exception as e:
        check("音频指纹", False, f"{type(e).__name__}: {e}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


if __name__ == "__main__":
    raise SystemExit(main())