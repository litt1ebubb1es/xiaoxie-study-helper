"""采集层：系统回环录音 + 媒体导入 + 音频指纹重复检测（方案 4.2 / 4.2 末段）。

三个设计取舍：

1. **系统回环而非逐客户端适配**（ADR-005）。浏览器里的网课和客户端里的网课，
   在操作系统看来都只是"一个进程在向声卡送音频"。采一次即覆盖全部。
2. **没有播放器事件时，用音频指纹推断"回退/重复播放"**。这是对方案 4.2 表格
   "播放行为需推断"的正面回答：同一个声学内容在几分钟内再次出现，只有两种可能——
   用户回退了，或平台自动重播了。两种都属于"这里值得标记"。
3. **只做外部观测**（方案 4.2 末段 / 7.2）：不注入进程、不挂接内部函数、不接管播放。
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import audio_io


# --------------------------------------------------------------------------
# 设备枚举
# --------------------------------------------------------------------------
def list_loopback_devices() -> list[dict]:
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        return []
    out: list[dict] = []
    p = pyaudio.PyAudio()
    try:
        out_name = ""
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            dflt = p.get_device_info_by_index(int(wasapi.get("defaultOutputDevice", -1)))
            out_name = str(dflt.get("name", ""))
        except Exception:
            out_name = ""
        for lb in p.get_loopback_device_info_generator():
            nm = str(lb["name"])
            # 默认标记：环回设备名里含默认输出设备名（比 index+1 的算术假设可靠）
            out.append({
                "index": int(lb["index"]),
                "name": nm,
                "channels": int(lb["maxInputChannels"]),
                "rate": int(lb["defaultSampleRate"]),
                "default": bool(out_name) and out_name.lower() in nm.lower(),
            })
    except Exception:
        pass
    finally:
        p.terminate()
    return out


# --------------------------------------------------------------------------
# 音频指纹：判断"这段内容刚才出现过"
# --------------------------------------------------------------------------
class AudioRepeatDetector:
    """把音频切成 1 秒帧，取 12 个频带的对数能量作为指纹。

    命中条件：与 [now-300s, now-12s] 内的某帧余弦相似度 ≥ 阈值，且连续 2 帧命中。
    输出两类事件：大幅回退（匹配点明显更早）→ seek_back；小幅重复 → replay。
    """

    def __init__(self, rate: int = 16000, n_bands: int = 12, sim: float = 0.985,
                 min_lookback_s: float = 12.0, max_lookback_s: float = 300.0,
                 cooldown_s: float = 20.0):
        self.rate = rate
        self.n_bands = n_bands
        self.sim = sim
        self.min_lookback_s = min_lookback_s
        self.max_lookback_s = max_lookback_s
        self.cooldown_s = cooldown_s
        self.frames: list[tuple[float, np.ndarray]] = []
        self._pending = np.zeros(0, dtype=np.float32)
        self._t = 0.0
        self._hits = 0
        self._last_hit_t = -1e9
        self._edges = np.fft.rfftfreq(1024, 1.0 / rate)
        self._band_idx = np.array_split(np.arange(len(self._edges)), n_bands)

    def feed(self, mono: np.ndarray, rate: int | None = None) -> list[dict]:
        r = rate or self.rate
        if r != self.rate:
            mono = _resample(mono, r, self.rate)
        self._pending = np.concatenate([self._pending, mono.astype(np.float32)])
        out: list[dict] = []
        n = self.rate
        while len(self._pending) >= n:
            chunk, self._pending = self._pending[:n], self._pending[n:]
            vec = self._fingerprint(chunk)
            out += self._match(vec)
        return out

    def _fingerprint(self, chunk: np.ndarray) -> np.ndarray:
        if not np.any(chunk):
            return np.zeros(self.n_bands, dtype=np.float32)
        # 3 段平均，降低相位敏感性
        parts = [chunk[i:i + 1024] for i in range(0, len(chunk) - 1024 + 1, 512)]
        if not parts:
            parts = [np.pad(chunk, (0, 1024 - len(chunk)))]
        spec = np.mean([np.abs(np.fft.rfft(p * np.hanning(len(p)))) for p in parts], axis=0)
        bands = np.array([float(np.sum(spec[idx])) for idx in self._band_idx]) + 1e-9
        v = np.log(bands)
        v = v - v.mean()
        nv = float(np.linalg.norm(v))
        return (v / nv).astype(np.float32) if nv > 1e-9 else np.zeros_like(v)

    def _match(self, vec: np.ndarray) -> list[dict]:
        out: list[dict] = []
        now = self._t
        self._t += 1.0
        if not np.any(vec):
            self.frames.append((now, vec))
            self._hits = 0
            return out
        hit = None
        for tt, vv in reversed(self.frames):
            if now - tt < self.min_lookback_s:
                continue
            if now - tt > self.max_lookback_s:
                break
            if float(np.dot(vec, vv)) >= self.sim:
                hit = tt
                break
        if hit is not None:
            self._hits += 1
        else:
            self._hits = 0
        if self._hits >= 2 and now - self._last_hit_t >= self.cooldown_s:
            self._last_hit_t = now
            dist = now - hit
            if dist >= 10.0:
                out.append({"kind": "seek_back", "t": now,
                            "payload": {"distance_s": round(dist, 1),
                                        "matched_t": round(hit, 1)},
                            "confidence": 0.75, "source": "audio_fp"})
            else:
                out.append({"kind": "replay", "t": now,
                            "payload": {"matched_t": round(hit, 1)},
                            "confidence": 0.7, "source": "audio_fp"})
            self._hits = 0
        self.frames.append((now, vec))
        if len(self.frames) > int(self.max_lookback_s * 2):
            self.frames = self.frames[-int(self.max_lookback_s * 2):]
        return out


class AudioPauseDetector:
    """靠"声音断掉"判断用户暂停/离开，同样不需要看屏幕。

    依据只有两条，都在音频里：
    1. 播放器暂停后，WASAPI 环回仍然在出数据，但全是 0（数字静音）；
    2. 设备整体空闲时干脆不出数据——这条由录制器的 idle_seconds 兜住。

    输出 pause（短停顿）或 overstay（长时间停在那里）。"失焦"这类判断做不到，
    因为不看窗口状态，所以不冒充 blur_pause。
    """

    def __init__(self, min_silence_s: float = 3.0, long_silence_s: float = 20.0,
                 eps: float = 1e-4):
        self.min_silence_s = float(min_silence_s)
        self.long_silence_s = float(long_silence_s)
        self.eps = float(eps)
        self._sil_start: float | None = None
        self._sil_end = 0.0

    def feed(self, mono: np.ndarray, t_end: float, rate: int) -> list[dict]:
        if mono is None or mono.size == 0:
            return []
        dur = float(len(mono)) / float(rate or 1)
        t_start = max(0.0, float(t_end) - dur)
        if float(np.max(np.abs(mono))) <= self.eps:
            if self._sil_start is None:
                self._sil_start = t_start
            self._sil_end = float(t_end)
            return []
        return self._close()

    def flush(self) -> list[dict]:
        return self._close()

    def _close(self) -> list[dict]:
        if self._sil_start is None:
            return []
        start, length = self._sil_start, max(0.0, self._sil_end - self._sil_start)
        self._sil_start = None
        if length < self.min_silence_s:
            return []
        if length >= self.long_silence_s:
            return [{"kind": "overstay", "t": start,
                     "payload": {"pause_s": round(length, 1),
                                 "reason": "长时间没有声音（暂停或离开）"},
                     "confidence": 0.6, "source": "audio_pause"}]
        return [{"kind": "pause", "t": start,
                 "payload": {"pause_s": round(length, 1)},
                 "confidence": 0.7, "source": "audio_pause"}]


def detect_audio_events(path, min_silence_s: float = 3.0,
                        long_silence_s: float = 20.0) -> list[dict]:
    """对已经存在的录音补做一遍「回溯 / 暂停」检测。

    在线录制时这两件事本来就在跑；但导入的文件、或者上次在线没跑到的情况，
    需要能事后补算。纯读文件，不联网、不调用模型。

    读的时候不假设格式：压缩后的 FLAC、导入的 m4a 都要能补算（老版本只认 WAV）。
    """
    p = Path(path)
    if not p.exists() or not audio_io.is_audio(p):
        return []
    events: list[dict] = []
    try:
        rate = int(audio_io.info(p)["rate"]) or 0
        if not rate:
            return []
        rep = AudioRepeatDetector(rate=rate)
        pau = AudioPauseDetector(min_silence_s, long_silence_s)
        done = 0.0
        for arr in audio_io.iter_mono_blocks(p, block_s=1.0, overlap_s=0.0):
            done += len(arr) / float(rate)
            try:
                events += rep.feed(arr, rate)
            except Exception:  # noqa: BLE001
                pass
            events += pau.feed(arr, done, rate)
        events += pau.flush()
    except Exception:  # noqa: BLE001
        return events
    return events


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """改采样率。实现放在 audio_io —— 那里还要给文件级重采样用，不能两处各一份。"""
    return audio_io.resample(x, src, dst)


# --------------------------------------------------------------------------
# 系统回环录音
# --------------------------------------------------------------------------
@dataclass
class RecorderEvent:
    t: float
    kind: str
    payload: dict = field(default_factory=dict)


class LoopbackRecorder:
    """WASAPI 系统回环录音（回调式）。

    为什么用回调而不是阻塞 read()：
    实测本机蓝牙耳机（EDIFIER W820NB）的环回设备在**没有音频播放时完全不出数据**，
    阻塞式 read() 会一直等下去，录音线程焊死、界面永远 0:00，用户以为程序坏了。
    回调式驱动下，没有数据就不触发回调，主流程照常运行，并能在界面上给出
    "当前设备没有音频流"的明确提示（蓝牙设备需要先开始播放）。
    """

    def __init__(self, out_path: str | Path, device_name: str = "",
                 channels: int = 2, detect_repeats: bool = True,
                 detect_pauses: bool = True, pause_min_s: float = 3.0,
                 pause_long_s: float = 20.0):
        self.out_path = Path(out_path)
        self.device_name = device_name
        self.channels_pref = channels
        self.detect_repeats = detect_repeats
        self.detect_pauses = detect_pauses
        self.rate = 0
        self.channels = 0
        self.duration = 0.0
        self.level = 0.0                 # 最近一块的 RMS（0..1），给电平条和"静音判定"用
        self.level_at = 0.0              # 上面这个值的采样时间，用来判断它是不是陈的
        self.error = ""
        self.warning = ""
        self.started_at = 0.0
        self.last_data_at = 0.0
        self._thread: threading.Thread | None = None
        self._watch: threading.Thread | None = None
        self._stop = threading.Event()
        self._wf: wave.Wave_write | None = None
        self._pa = None
        self._stream = None
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=256)
        self._samples = 0
        self._lock = threading.Lock()
        self.detector = AudioRepeatDetector(rate=16000)
        self.pause_detector = AudioPauseDetector(pause_min_s, pause_long_s)
        self._stall_from = 0.0
        self.live_audio: list[np.ndarray] = []
        self.on_level: Callable[[float], None] | None = None
        self.on_event: Callable[[RecorderEvent], None] | None = None
        self.device_label = ""

    # -- 设备选择 -------------------------------------------------------
    def _pick_device(self) -> dict:
        import pyaudiowpatch as pyaudio
        p = self._pa
        assert p is not None
        wanted = (self.device_name or "").strip().lower()
        devices = list(p.get_loopback_device_info_generator())
        if not devices:
            raise RuntimeError("系统里没有可用的 WASAPI 回环设备")
        if wanted:
            for d in devices:
                if wanted in str(d["name"]).lower():
                    return d
        try:
            return p.get_default_wasapi_loopback()
        except Exception:
            return devices[0]

    # -- 生命周期 -------------------------------------------------------
    def start(self) -> None:
        import pyaudiowpatch as pyaudio

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._pa = pyaudio.PyAudio()
        dev = self._pick_device()
        self.rate = int(dev["defaultSampleRate"])
        self.channels = min(2, int(dev["maxInputChannels"]))
        self.device_label = str(dev["name"])
        self._wf = wave.open(str(self.out_path), "wb")
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.rate)
        block = max(256, int(self.rate * 0.04))
        last_err = None
        for _attempt in range(2):
            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16, channels=self.channels, rate=self.rate,
                    input=True, input_device_index=int(dev["index"]),
                    frames_per_buffer=block, stream_callback=self._callback,
                )
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.6)
        if self._stream is None:
            try:
                self._wf.close()
            except Exception:
                pass
            try:
                self._pa.terminate()
                self._pa = None
            except Exception:
                pass
            raise RuntimeError(
                f"无法打开回环设备「{dev['name']}」：{last_err}。"
                f"请确认该输出设备可用，或换一个输出设备后重试。"
            )
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._writer, name="loopback-writer", daemon=True)
        self._thread.start()
        if self.detect_pauses:
            self._watch = threading.Thread(target=self._watch_stall, name="pause-watch",
                                           daemon=True)
            self._watch.start()

    def _watch_stall(self) -> None:
        """设备整体不出数据时也当成"用户没在听"。

        蓝牙/HDMI 这类设备暂停后常常一个字节都不给，静音检测就没机会触发；
        这里用采样数停止增长来判断。恢复出声后自动闭合并补发事件。
        """
        while not self._stop.wait(1.0):
            dur = self.duration
            idle = self.idle_seconds
            if dur <= 0 or not self.receiving:
                continue
            if idle >= self.pause_detector.min_silence_s:
                if self._stall_from <= 0:
                    self._stall_from = max(0.0, dur - idle)
                continue
            if self._stall_from > 0:
                length = max(0.0, time.time() - self.last_data_at)
                start, self._stall_from = self._stall_from, 0.0
                length = max(length, self.pause_detector.min_silence_s)
                kind = ("overstay" if length >= self.pause_detector.long_silence_s
                        else "pause")
                ev = {"kind": kind, "t": start,
                      "payload": {"pause_s": round(length, 1),
                                  "reason": "设备没有音频流（暂停或静音）"},
                      "confidence": 0.6, "source": "audio_pause"}
                if self.on_event:
                    try:
                        self.on_event(RecorderEvent(ev["t"], ev["kind"], ev["payload"]))
                    except Exception:
                        pass

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        try:
            self._q.put_nowait(bytes(in_data))
        except queue.Full:
            pass  # 宁可丢一块，也不能阻塞音频线程
        return (None, pyaudio.paContinue)

    def _writer(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                raw = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_block(raw)
            except Exception as e:  # noqa: BLE001
                self.error = f"写盘失败：{type(e).__name__}: {e}"
                break

    def _write_block(self, raw: bytes) -> None:
        if not raw:
            return
        if self._wf is not None:
            self._wf.writeframes(raw)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1 and arr.size % self.channels == 0:
            arr = arr.reshape(-1, self.channels).mean(axis=1)
        self.last_data_at = time.time()
        with self._lock:
            self._samples += arr.size
            self.duration = self._samples / float(self.rate)
        lvl = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        self.level = lvl
        self.level_at = self.last_data_at
        if self.on_level:
            try:
                self.on_level(lvl)
            except Exception:
                pass
        mono16 = _resample(arr, self.rate, 16000)
        with self._lock:
            self.live_audio.append(mono16.copy())
            if len(self.live_audio) > 400:
                self.live_audio = self.live_audio[-200:]
        if self.detect_repeats:
            for ev in self.detector.feed(arr, self.rate):
                if self.on_event:
                    try:
                        self.on_event(RecorderEvent(ev["t"], ev["kind"], ev["payload"]))
                    except Exception:
                        pass
        if self.detect_pauses:
            for ev in self.pause_detector.feed(arr, self.duration, self.rate):
                if self.on_event:
                    try:
                        self.on_event(RecorderEvent(ev["t"], ev["kind"], ev["payload"]))
                    except Exception:
                        pass

    # -- 状态 -----------------------------------------------------------
    @property
    def idle_seconds(self) -> float:
        """距最近一次收到音频数据的秒数（从未收到则从启动算起）。"""
        base = self.last_data_at or self.started_at
        return max(0.0, time.time() - base) if base else 0.0

    @property
    def receiving(self) -> bool:
        return self.last_data_at > 0

    def stop(self) -> Path:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        # 录到最后一秒还在静音时，那段也要补一条事件，否则"最后暂停了"会丢
        try:
            for ev in self.pause_detector.flush():
                if self.on_event:
                    self.on_event(RecorderEvent(ev["t"], ev["kind"], ev["payload"]))
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            if self._wf is not None:
                self._wf.close()
        except Exception:
            pass
        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        self._pa = None
        self._thread = None
        return self.out_path

    # -- 取最近的音频 ---------------------------------------------------
    def drain_audio(self) -> np.ndarray:
        with self._lock:
            if not self.live_audio:
                return np.zeros(0, dtype=np.float32)
            out = np.concatenate(self.live_audio)
            self.live_audio = []
        return out


def probe_loopback(index: int, seconds: float = 0.9) -> dict:
    """试读一个环回设备：判断它当前是否真的在产出音频。

    蓝牙耳机在没播放时不出数据，这个探测让用户能自己看出来该选哪个设备。
    """
    try:
        import pyaudiowpatch as pyaudio
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    p = pyaudio.PyAudio()
    info: dict = {"ok": False, "bytes": 0, "elapsed": 0.0}
    stream = None
    try:
        dev = p.get_device_info_by_index(int(index))
        ch = min(2, int(dev["maxInputChannels"]))
        rate = int(dev["defaultSampleRate"])
        block = max(256, int(rate * 0.04))
        got = queue.Queue()

        def cb(in_data, frame_count, time_info, status):
            if not got.empty():
                return (None, pyaudio.paContinue)
            got.put(bytes(in_data))
            return (None, pyaudio.paContinue)

        t0 = time.time()
        stream = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                        input_device_index=int(index), frames_per_buffer=block,
                        stream_callback=cb)
        try:
            data = got.get(timeout=seconds)
            info.update(ok=bool(data), bytes=len(data or b""))
        except queue.Empty:
            info.update(ok=False, bytes=0)
        info["elapsed"] = round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        p.terminate()
    return info


# --------------------------------------------------------------------------
# 媒体导入（视频/音频 → 16k 单声道 wav）
# --------------------------------------------------------------------------
def import_media(src: str | Path, dst: str | Path) -> Path:
    """用 imageio-ffmpeg 自带的 ffmpeg 提取音轨，不要求系统安装 ffmpeg。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"找不到 ffmpeg：{e}")
    cmd = [exe, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not dst.exists():
        tail = (r.stderr or "")[-800:]
        raise RuntimeError(f"音轨提取失败：{tail}")
    return dst


def audio_duration(path: str | Path) -> float:
    # 课后音频会被压成 FLAC，所以读时长不能只认 WAV：统一交给 audio_io
    return audio_io.duration(path)


def read_wav_mono16k(path: str | Path) -> np.ndarray:
    return audio_io.read_mono16k(path)


def write_wav(path: str | Path, mono16k: np.ndarray) -> Path:
    return audio_io.write_wav(path, mono16k, 16000)


def split_wav(path: str | Path, chunk_seconds: float = 120.0,
              out_dir: str | Path | None = None) -> list[tuple[float, Path]]:
    """长音频分段（方案 5.1：精转模型有时长限制，需要分段处理）。"""
    src = Path(path)
    out_dir = Path(out_dir or src.parent / f"{src.stem}_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = read_wav_mono16k(src)
    n = int(16000 * chunk_seconds)
    parts: list[tuple[float, Path]] = []
    for i in range(0, max(1, len(audio)), n):
        seg = audio[i:i + n]
        if len(seg) < 16000 * 0.3:
            continue
        p = write_wav(out_dir / f"{src.stem}_{len(parts):03d}.wav", seg)
        parts.append((i / 16000.0, p))
    return parts