"""感知层：转写后端（方案 4.4 / 5.1）。

四种后端，全部对外暴露同一份 `(segments, words)` 结构，上层不感知差异：

| 后端 | 用途 | 依赖 |
|------|------|------|
| openai_compat | 云端或本地 OpenAI 兼容转写服务（whisper / FunASR-server / vLLM-audio） | 网络或本地端口 |
| faster_whisper | 纯本地离线转写，随包可选 | faster-whisper + 模型文件 |
| sidecar | 读取与音频同名的 .srt/.vtt/.txt（课程自带字幕、你自己整理的字幕） | 无 |
| demo | 无可用的真实后端时的确定性演示流，**输出不是真实转写**，UI 明确标注 | 无 |

双遍转写（方案 5.1）在这里体现为：`transcribe(..., precise=False)` 走实时模型，
`precise=True` 走精转模型；两遍结果都带词级时间戳，融合与对齐在 pipeline 里完成。
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class ASRSegment:
    t_start: float
    t_end: float
    text: str
    words: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    speaker: str = ""


@dataclass
class ASRResult:
    segments: list[ASRSegment]
    backend: str
    is_real: bool = True
    note: str = ""
    language: str = "zh"

    def as_dicts(self) -> list[dict]:
        return [
            {
                "t_start": s.t_start, "t_end": s.t_end, "text": s.text,
                "words": s.words, "confidence": s.confidence, "speaker": s.speaker,
            }
            for s in self.segments
        ]


class ASRError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# 1) OpenAI 兼容转写
# --------------------------------------------------------------------------
class OpenAICompatASR:
    def __init__(self, base_url: str, api_key: str, model: str, language: str = "zh",
                 timeout_s: float = 600.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "未配置 base_url"
        from urllib.parse import urlparse
        host = (urlparse(self.base_url).hostname or "").lower()
        is_local = host in ("127.0.0.1", "localhost", "::1")
        if not self.api_key and not is_local:
            return False, "缺 api_key（云端转写需要鉴权）"
        try:
            import socket
            from urllib.parse import urlparse
            u = urlparse(self.base_url)
            host = u.hostname or ""
            port = u.port or (443 if u.scheme == "https" else 80)
            socket.create_connection((host, port), timeout=4).close()
            return True, f"{host}:{port} 可达"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def transcribe(self, audio_path: str | Path, precise: bool = False) -> ASRResult:
        from openai import OpenAI

        p = Path(audio_path)
        if not p.exists():
            raise ASRError(f"音频不存在：{p}")
        client = OpenAI(base_url=self.base_url, api_key=self.api_key or "sk-local",
                        timeout=self.timeout_s)
        with p.open("rb") as fh:
            try:
                resp = client.audio.transcriptions.create(
                    model=self.model,
                    file=fh,
                    response_format="verbose_json",
                    language=self.language,
                    timestamp_granularities=["segment", "word"],
                )
            except TypeError:
                fh.seek(0)
                resp = client.audio.transcriptions.create(
                    model=self.model, file=fh, response_format="verbose_json",
                    language=self.language,
                )
        data = resp if isinstance(resp, dict) else resp.model_dump()
        segs: list[ASRSegment] = []
        words_all = data.get("words") or []
        for i, s in enumerate(data.get("segments") or []):
            ws = [w for w in words_all
                  if float(w.get("start", -1)) >= float(s.get("start", 0)) - 0.01
                  and float(w.get("end", 1e9)) <= float(s.get("end", 0)) + 0.01]
            segs.append(ASRSegment(
                t_start=float(s.get("start", 0.0)),
                t_end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip(),
                words=[{"t_start": float(w.get("start", 0)), "t_end": float(w.get("end", 0)),
                        "text": str(w.get("word", "")).strip()} for w in ws],
                confidence=1.0 - float(s.get("no_speech_prob", 0.0) or 0.0),
            ))
        if not segs and data.get("text"):
            dur = _audio_duration(p) or 60.0
            segs = _split_by_punctuation(str(data["text"]), 0.0, dur)
        if not segs:
            raise ASRError("转写服务返回空结果")
        return ASRResult(segments=segs, backend=f"openai_compat:{self.model}")


def _split_by_punctuation(text: str, t0: float, t1: float) -> list[ASRSegment]:
    """没有时间戳时的兜底：按标点切句并按字数比例分配时间。"""
    pieces = [p.strip() for p in re.split(r"(?<=[。！？；!?;])", text) if p.strip()]
    if not pieces:
        return []
    total = sum(len(p) for p in pieces)
    out: list[ASRSegment] = []
    cur = t0
    for p in pieces:
        d = (t1 - t0) * len(p) / total
        out.append(ASRSegment(t_start=cur, t_end=cur + d, text=p))
        cur += d
    return out


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        import subprocess
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mi * 60 + s
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------
# 2) 本地 faster-whisper（可选）
# --------------------------------------------------------------------------
class FasterWhisperASR:
    def __init__(self, model_size: str = "", device: str = "auto",
                 compute_type: str = "auto", language: str = "zh"):
        # model_size 可以是尺寸名（small）也可以是本机目录 —— 由 whisper_setup 解析
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    def available(self) -> tuple[bool, str]:
        from .. import whisper_setup as ws

        ok, msg = ws.deps_status()
        if not ok:
            return False, "未安装 faster-whisper（设置页有「安装依赖」按钮）"
        if not ws.status()["any_model"]:
            return False, f"{msg}｜本机还没有任何 Whisper 模型（设置页点「下载模型」）"
        return True, msg

    def transcribe(self, audio_path: str | Path, precise: bool = False) -> ASRResult:
        from .. import whisper_setup as ws

        rows, info, how = ws.transcribe_retry(
            self.model_size, audio_path, language=self.language,
            prefer_gpu=(self.device != "cpu"), want_words=True,
            word_timestamps=True, vad_filter=True)
        out = [ASRSegment(t_start=r["t_start"], t_end=r["t_end"], text=r["text"],
                          words=r["words"], confidence=r["confidence"])
               for r in rows if r["text"]]
        if not out:
            raise ASRError("faster-whisper 未产出分段")
        return ASRResult(segments=out, backend=f"faster_whisper:{how}")


# --------------------------------------------------------------------------
# 3) 同名字幕侧车
# --------------------------------------------------------------------------
class SidecarASR:
    SUFFIXES = (".srt", ".vtt", ".txt", ".md")

    def __init__(self, audio_path: str | Path, duration_hint: float | None = None):
        self.audio_path = Path(audio_path)
        self.duration_hint = duration_hint

    def find(self) -> Path | None:
        # 直接传进来的就是字幕/文本文件（没有音频的会话走这条路）
        if self.audio_path.suffix.lower() in self.SUFFIXES and self.audio_path.is_file():
            return self.audio_path
        for suf in self.SUFFIXES:
            p = self.audio_path.with_suffix(suf)
            if p.exists():
                return p
        return None

    def available(self) -> tuple[bool, str]:
        p = self.find()
        return (True, f"找到字幕 {p.name}") if p else (False, "同目录没有 .srt/.vtt/.txt")

    def transcribe(self, audio_path: str | Path | None = None, precise: bool = False) -> ASRResult:
        p = self.find()
        if not p:
            raise ASRError("没有可用的字幕文件")
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix.lower() in (".srt", ".vtt"):
            segs = _parse_srt(raw)
        else:
            dur = (self.duration_hint or _audio_duration(self.audio_path)
                   or max(60.0, len(raw) / 4.5))
            segs = _split_by_punctuation(raw.replace("\n", ""), 0.0, dur)
        if not segs:
            raise ASRError("字幕解析为空")
        return ASRResult(segments=segs, backend=f"sidecar:{p.name}")


def _parse_srt(raw: str) -> list[ASRSegment]:
    blocks = re.split(r"\n\s*\n", raw.strip())
    out: list[ASRSegment] = []
    for b in blocks:
        m = re.search(
            r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})",
            b)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        text = "\n".join(b.splitlines()[1:]).strip() if "-->" in b.splitlines()[0] else \
            "\n".join(l for l in b.splitlines() if "-->" not in l).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            out.append(ASRSegment(t_start=t0, t_end=t1, text=text))
    return out


# --------------------------------------------------------------------------
# 4) 演示后端（明确标注为非真实转写）
# --------------------------------------------------------------------------
_DEMO_SECTIONS = [
    ("引入", ["这一节我们讲随机变量和它的分布",
              "先从样本空间说起，样本空间是全部可能结果的集合",
              "随机变量是指把样本空间映射到实数轴的一个函数"]),
    ("概念定义", ["把随机变量记作 X，分布函数定义为 F(x) 等于 P(X 小于等于 x)",
                  "分布函数一定是右连续的",
                  "期望是指随机变量的加权平均",
                  "方差的定义是偏离均值的平方的期望"]),
    ("推导", ["由期望的线性性，可以把求和拆开",
              "两边同时提出常数因子",
              "于是得到了方差的另一个表达式",
              "这个递推关系对离散型和连续型都成立"]),
    ("易错点", ["这里有一个特别容易出错的地方",
                "分布函数在跳跃点处取的是右极限",
                "很多人把条件概率和联合概率搞混",
                "条件概率记作 P(A|B)，它要求 B 的概率不为零"]),
    ("数值例子", ["我们用一个具体的例子跑一遍",
                  "假设 X 服从参数为二的泊松分布",
                  "算出来期望和方差都等于二",
                  "再用切比雪夫不等式估一个上界"]),
    ("小结", ["小结一下",
              "这一节的重点是把分布函数和概率密度函数分开理解",
              "下一节讲大数定律和中心极限定理"]),
]


class DemoASR:
    """确定性演示转写。**不是真实语音识别结果**，仅用于无任何后端时跑通链路。"""

    def __init__(self, seed: int = 20261001, duration_hint: float = 0.0):
        self.seed = seed
        self.duration_hint = duration_hint

    def available(self) -> tuple[bool, str]:
        return True, "演示后端（文本为合成内容，非真实转写）"

    def transcribe(self, audio_path: str | Path | None = None, precise: bool = False) -> ASRResult:
        rng = random.Random(self.seed + (1 if precise else 0))
        dur = self.duration_hint or (_audio_duration(Path(audio_path)) if audio_path else 0.0) or 480.0
        segs: list[ASRSegment] = []
        t = 2.0
        round_i = 0
        while t < dur - 3:
            name, lines = _DEMO_SECTIONS[round_i % len(_DEMO_SECTIONS)]
            round_i += 1
            for line in lines:
                dense = name in ("概念定义", "推导", "易错点")
                txt = line
                if dense and rng.random() < 0.5:
                    txt = rng.choice(["这里涉及", "注意"]) + line
                length = max(2.0, len(txt) / (4.6 if dense else 5.4))
                if t + length > dur:
                    length = max(1.5, dur - t - 0.5)
                if length <= 0.8:
                    break
                # 长停顿：密集段之间插入 1.6–3 秒空白
                segs.append(ASRSegment(t_start=t, t_end=t + length, text=txt,
                                       confidence=0.0))
                t += length + (rng.uniform(0.25, 0.7) if not dense else rng.uniform(1.6, 3.2))
        if not segs:
            segs = [ASRSegment(t_start=0.0, t_end=dur, text="（无音频内容）", confidence=0.0)]
        return ASRResult(segments=segs, backend="demo", is_real=False,
                         note="演示后端：文本为合成内容，不是真实转写")


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
_SUBTITLE_SUFFIXES = (".srt", ".vtt", ".txt", ".md")


def build_candidates(settings, audio_path: str | Path | None = None,
                     precise: bool = False) -> list[tuple[Any, str]]:
    """按代价从低到高列出所有可尝试的转写后端。

    顺序：侧车字幕（零成本）→ 本地 Whisper（离线免费）→ 云端（快但要联网）
    → 演示（仅显式允许时）。

    为什么必须返回**列表**：云端转写的 available() 只能验证到"主机可达"，
    真调用才发现端点没有 /audio/transcriptions（实测 404）。所以兜底要发生在
    **调用失败之后**，而不是选后端的时候。
    """
    a = settings.asr
    backend = (a.backend or "auto").lower()
    out: list[tuple[Any, str]] = []

    p = Path(audio_path) if audio_path else None
    has_media = bool(p and p.is_file() and p.suffix.lower() not in _SUBTITLE_SUFFIXES)

    if backend in ("auto", "sidecar") and audio_path:
        out.append((SidecarASR(audio_path), "同名字幕"))
    if has_media and backend in ("auto", "local_whisper", "faster_whisper"):
        try:
            from .. import whisper_setup as ws

            if ws.status()["any_model"] and ws.deps_status()[0]:
                out.append((FasterWhisperASR(
                    model_size=getattr(a, "local_size", ""),
                    device=getattr(a, "local_device", "auto"),
                    language=a.language), "本地 Whisper"))
        except Exception:  # noqa: BLE001
            pass
    if has_media and backend in ("auto", "cloud", "local_openaicompat"):
        base = a.base_url or (settings.llm.base_url if backend == "auto" else "")
        key = a.api_key or settings.llm.api_key
        model = a.precise_model if precise else a.realtime_model
        if base:
            out.append((OpenAICompatASR(base, key, model, a.language), "云端转写"))
    if getattr(a, "allow_demo", False):
        out.append((DemoASR(a.offline_seed), "演示转写"))
    return out


def build_asr(settings, audio_path: str | Path | None = None, precise: bool = False):
    """按配置挑一个可用的后端，返回 (backend, 说明)。"""
    cands = build_candidates(settings, audio_path, precise)
    notes = []
    for c, hint in cands:
        ok, msg = c.available()
        notes.append(f"{hint}({type(c).__name__})={msg}")
        if ok:
            return c, " | ".join(notes)
    raise ASRError(
        "没有可用的转写后端：" + (" | ".join(notes) or "没有任何候选")
        + "。请到设置页填 api_key，或给这一节配一份同名字幕（.srt），"
          "也可以在设置里显式打开「允许演示转写」，"
          "或者装本地 Whisper（设置页「本地转写」一栏）。")
