"""音频文件读写收口。

为什么要这一层：录音存的是 WAV（`wave` 模块写的），但课后要把它转成 16k 单声道
FLAC —— 一节 90 分钟的课，WAV 是 890 MB，FLAC 大约 95 MB，而转写只需要 16k 单声道。
可老代码里"算时长""读波形""补算回溯"全是 `wave.open`，只认 WAV。与其在每个调用点
加 `if 后缀 != .wav`，不如把"读音频"这件事统一到这里，上层继续调原来的函数名。

另一件事：**不能整段读进内存**。90 分钟 44.1k 立体声转 float32 是 2 GB，
转码必须分块，所以 `compress_to_flac` 是一块一块读、一块一块写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

TARGET_RATE = 16000          # 转写/特征分析只用这个采样率
FLAC_SUBTYPE = "PCM_16"
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus", ".aiff", ".aif"}


def _sf():
    import soundfile as sf

    return sf


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """改采样率。有 scipy 用多相滤波（质量好、快），没有就线性插值。"""
    if src == dst or x.size == 0:
        return x
    from math import gcd

    g = gcd(int(src), int(dst))
    try:
        from scipy.signal import resample_poly

        return resample_poly(x, dst // g, src // g).astype(np.float32)
    except Exception:                      # noqa: BLE001 —— 没装 scipy 也得能跑
        n = int(len(x) * dst / src)
        if n <= 1:
            return x
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float32)


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def info(path: str | Path) -> dict:
    """采样率、帧数、时长、声道、容器格式。读不出来就返回空值，不抛。"""
    p = Path(path)
    try:
        with _sf().SoundFile(str(p)) as f:
            rate = int(f.samplerate)
            frames = int(f.frames)
            return {"rate": rate, "frames": frames, "channels": int(f.channels),
                    "duration": frames / rate if rate else 0.0,
                    "format": str(f.format or "").lower()}
    except Exception:                       # noqa: BLE001
        return {"rate": 0, "frames": 0, "channels": 0, "duration": 0.0, "format": ""}


def duration(path: str | Path) -> float:
    """时长（秒）。WAV 走 wave 模块最快，其它格式交给 soundfile。"""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.suffix.lower() == ".wav":
        import wave

        try:
            with wave.open(str(p), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate() or 1)
        except Exception:                   # noqa: BLE001
            pass
    return float(info(p)["duration"])


def iter_mono_blocks(path: str | Path, *, block_s: float = 20.0,
                     overlap_s: float = 0.5) -> Iterator[np.ndarray]:
    """按块吐单声道 float32（原始采样率）。

    每块多读 `overlap_s` 的尾巴，供调用方在重采样后丢掉开头重叠段用——
    这样分块重采样的接缝落在"上一块的内部"，不会有咔哒声。
    """
    p = Path(path)
    with _sf().SoundFile(str(p)) as f:
        rate = int(f.samplerate)
        total = int(f.frames)
        step = max(1, int(block_s * rate))
        tail = max(0, int(overlap_s * rate))
        pos = 0
        while pos < total:
            # 有一类容器（尤其 FLAC）不允许 seek 到文件尾之后：
            # 循环条件卡住 pos，再夹一次，读到末尾就自然结束。
            f.seek(min(pos, max(0, total - 1)))
            data = f.read(frames=step + tail, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            yield data.mean(axis=1).astype(np.float32)
            pos += step


def read_mono(path: str | Path, *, rate: int | None = TARGET_RATE,
              start: float = 0.0, max_seconds: float | None = None) -> np.ndarray:
    """读一段单声道波形。`rate=None` 表示保持原采样率（播放时用）。"""
    p = Path(path)
    with _sf().SoundFile(str(p)) as f:
        f_rate = int(f.samplerate)
        total = int(f.frames)
        if start > 0:
            f.seek(min(int(start * f_rate), max(0, total - 1)))
        frames = -1 if max_seconds is None else int(max_seconds * f_rate)
        data = f.read(frames=frames, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if len(data) else np.zeros(0, np.float32)
    if rate and mono.size:
        mono = resample(mono, f_rate, int(rate))
    return mono


def read_mono16k(path: str | Path, *, max_seconds: float | None = None) -> np.ndarray:
    return read_mono(path, rate=TARGET_RATE, max_seconds=max_seconds)


def read_segment(path: str | Path, t0: float, t1: float) -> tuple[np.ndarray, int]:
    """取 [t0, t1) 这段，保持原采样率——重新转写/校对某一段时直接用它。"""
    span = max(0.0, float(t1) - float(t0))
    with _sf().SoundFile(str(path)) as f:
        rate = int(f.samplerate)
        total = int(f.frames)
        f.seek(max(0, min(int(t0 * rate), max(0, total - 1))))
        data = f.read(frames=int(span * rate), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if len(data) else np.zeros(0, np.float32)
    return mono, rate


def write_wav(path: str | Path, mono: np.ndarray, rate: int = TARGET_RATE) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(mono, -1.0, 1.0)
    _sf().write(str(p), data.astype(np.float32), int(rate), subtype="PCM_16")
    return p


def compress_to_flac(src: str | Path, dst: str | Path, *,
                     rate: int = TARGET_RATE, block_s: float = 20.0) -> Path:
    """WAV（任意采样率/声道）→ 16k 单声道 FLAC。分块流式，内存占用与文件长度无关。"""
    sf = _sf()
    s, d = Path(src), Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    src_rate = int(info(s)["rate"]) or TARGET_RATE
    tmp = d.with_suffix(d.suffix + ".part")
    with sf.SoundFile(str(tmp), mode="w", samplerate=int(rate), channels=1,
                      format="FLAC", subtype=FLAC_SUBTYPE) as out:
        first = True
        for block in iter_mono_blocks(s, block_s=block_s):
            # 块内自己重采样；非首块丢掉开头重叠的那 0.5 秒（它在上一块尾巴里已经写过）
            mono = resample(block, src_rate, int(rate))
            if not first:
                skip = int(0.5 * rate)
                mono = mono[skip:]
            first = False
            if mono.size:
                out.write(mono.astype(np.float32))
    tmp.replace(d)
    return d