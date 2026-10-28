"""管线编排：录制 → 转写 → 负荷计算 → 回看点判定 → 入库 → 重讲 / 诊断。

这一层是六层架构的粘合处，不包含算法细节：
认知算法在 cognitive/，转写在 asr/，生成在 review/，采集在 capture/。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import capture as cap
from .asr import ASRResult, DemoASR, build_asr
from .cognitive import (
    AnalysisResult,
    Evt,
    Knot,
    TermLexicon,
    WeightState,
    analyze,
    calibrate_step,
)
from .config import AUDIO_DIR, Settings, ensure_dirs
from .db import DB
from .llm import LLMClient, build_llm
from .review import STRATEGY_MAP, STRATEGIES, build_prompt, fallback_outline, recommend

BACKEND_NOTE_KEY = "asr_last_note"


# --------------------------------------------------------------------------
# 实时分析（用户裁决：听课中可以做分析）
# --------------------------------------------------------------------------
class LiveAnalyzer:
    """滚动窗口实时打分。

    方案原文要求"回退后 3 秒内出现标记"但同时说"听课中不做分析"——两者冲突。
    用户裁决允许听课中分析，因此这里做轻量滚动估计：
    每 `min_s` 秒用最近 `window_s` 秒的转写重算一次 L，与刚收到的行为事件融合。
    它对延迟友好（不落库、不调用模型），代价是 L 用课程前段的运行统计近似 z-score。
    """

    def __init__(self, cfg, lexicon: TermLexicon | None = None):
        self.cfg = cfg
        self.lexicon = lexicon
        self.segments: list[dict] = []
        self.events: list[Evt] = []
        self._last_run = 0.0
        self.last_result: AnalysisResult | None = None

    def push_segments(self, segs: Iterable[dict]) -> None:
        self.segments.extend(segs)

    def push_event(self, kind: str, t: float, payload: dict | None = None,
                   confidence: float = 1.0, source: str = "manual") -> None:
        self.events.append(Evt(t=t, kind=kind, payload=payload or {},
                               confidence=confidence, source=source))

    def maybe_run(self, now: float, force: bool = False) -> list[Knot]:
        if not self.cfg.live_enabled:
            return []
        if not self.segments:
            return []
        if not force and now - self._last_run < self.cfg.step_s:
            return []
        self._last_run = now
        tail_start = max(0.0, now - self.cfg.window_s * 3)
        tail = [s for s in self.segments if float(s["t_end"]) >= tail_start]
        if not tail:
            return []
        evs = [e for e in self.events if e.t >= tail_start]
        res = analyze(tail, evs, self.cfg, lexicon=self.lexicon)
        self.last_result = res
        return res.knots


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------
@dataclass
class RewordResult:
    text: str
    strategy: str
    strategy_name: str
    degraded: bool = False
    error: str = ""


class _Unconfigured:
    """占位模型：configured=False，让 learn_terms 直接走确定性抽取。"""

    configured = False

    def complete_json(self, *a, **kw):      # noqa: D401
        raise RuntimeError("未配置模型")

    def complete_text(self, *a, **kw):
        raise RuntimeError("未配置模型")

    def chat(self, *a, **kw):
        return "", "未配置模型"


class Engine:
    def __init__(self, db: DB, settings: Settings):
        self.db = db
        self.settings = settings
        self.llm: LLMClient = build_llm(settings)
        self.weights: WeightState = WeightState.from_json(
            db.get_kv("weights"), settings.cognitive)
        self.recorder: cap.LoopbackRecorder | None = None
        self._live: LiveAnalyzer | None = None
        self._record_session_id: str = ""
        self._audio_events: list[dict] = []
        self.on_level: "Callable[[float], None] | None" = None   # UI 电平条挂这里

    def capture_level(self) -> tuple[float, float]:
        """当前录音电平与它的新鲜度：`(rms, 距今秒数)`。

        网页视频自动抓取要靠它判断"页面是不是真的没声音了"。没有录音、
        或者电平是很久以前那块数据的，都返回一个很大的"年龄"，让调用方
        知道这个数不能当证据。
        """
        rec = self.recorder
        if rec is None or not rec.level_at:
            return 0.0, 1e9
        return float(rec.level), max(0.0, time.time() - float(rec.level_at))

    def is_capture_quiet(self, threshold: float = 0.005, stale_s: float = 3.0) -> bool:
        """录音里现在是安静的吗？（网页视频自动抓取用这个决定停不停）

        两种情况都算安静：
        1. 电平是新鲜的、而且确实很低（页面被静音、视频暂停）；
        2. **根本收不到数据**——WASAPI 回环在没有任何程序出声时不给回调，
           这时文件也不再增长，就是"没声音"。

        第 2 条一开始写反了（判不了就当有声音），结果是浏览器停播之后
        会话永远不结束，一节录到天荒地老。真正防"切碎片"的不是这里，
        而是调用方的 `stop_after_s`：要连续安静 20 秒才停。
        """
        rec = self.recorder
        if rec is None:
            return True
        level, age = self.capture_level()
        if age > stale_s:
            return True
        return level < threshold

    # -- 预热 -----------------------------------------------------------
    def preload_llm(self) -> tuple[bool, str]:
        if self.settings.llm.backend == "off":
            return False, "生成层已关闭"
        return self.llm.preload()

    # -- 录制 -----------------------------------------------------------
    def start_web_capture(self, meta: dict) -> str:
        """网页视频自动抓取的开始：按站点归课、页面标题当节名。

        为什么不按标题归课：标题每集都不一样，一学期下来会散成几十门"课"。
        站点是稳定的那一维（同一站的网课放一起，找的时候按站找）。
        """
        course = str(meta.get("course") or "").strip() or "网页视频·网课"
        title = str(meta.get("page_title") or "").strip()             or str(meta.get("site") or "").strip() or "网页视频"
        if self.recording:
            return self._record_session_id or ""
        return self.start_recording(course, title, source="webvideo")

    def stop_web_capture(self, reason: str = "silence") -> dict:
        """自动抓取的收尾：停录，把音频交给同一条分析链。"""
        if not self.recording:
            return {"ok": False, "error": "当前没有在录制"}
        sid, path, dur = self.stop_recording()
        # 记成会话事件（不是 feedback：feedback 挂在回看点上，这里说的是"这节是怎么录到的"）
        try:
            self.db.add_event(sid, float(dur or 0.0), "web_capture_stop",
                              {"reason": reason, "duration": round(float(dur or 0), 1)},
                              1.0, "capture_watch")
        except Exception:
            pass
        return {"ok": True, "session_id": sid, "audio": str(path or ""),
                "duration": float(dur or 0.0), "reason": reason}

    def start_recording(self, course_title: str, session_title: str,
                        device_name: str = "", source: str = "loopback") -> str:
        ensure_dirs()
        cid = self.db.ensure_course(course_title)
        audio_path = AUDIO_DIR / f"{int(time.time())}_loopback.wav"
        sid = self.db.create_session(cid, session_title, str(audio_path), source=source)
        self.recorder = cap.LoopbackRecorder(
            audio_path, device_name or self.settings.capture.device_name,
            self.settings.capture.channels,
            detect_pauses=self.settings.capture.detect_pauses,
            pause_min_s=self.settings.capture.pause_min_s,
            pause_long_s=self.settings.capture.pause_long_s,
        )
        self._audio_events = []
        self.recorder.on_event = self._on_repeat_event
        self.recorder.on_level = self._on_level
        self.recorder.start()
        self._record_session_id = sid
        self._live = LiveAnalyzer(self.settings.cognitive)
        return sid

    def _on_repeat_event(self, ev: cap.RecorderEvent) -> None:
        self._audio_events.append({"kind": ev.kind, "t": ev.t, "payload": ev.payload,
                                   "confidence": 0.75, "source": "audio_fp"})
        if self._live:
            self._live.push_event(ev.kind, ev.t, ev.payload, 0.75, "audio_fp")

    def _on_level(self, level: float) -> None:
        """录音线程每块回调一次。只在有 UI 挂钩时转发，不影响无界面路径。"""
        cb = self.on_level
        if cb is not None:
            try:
                cb(float(level))
            except Exception:              # noqa: BLE001
                pass

    def mark_manual(self, note: str = "") -> None:
        """手动标记：优先级高于任何自动信号（方案 3.2）。"""
        t = self.recorder.duration if self.recorder else 0.0
        self._audio_events.append({"kind": "manual_mark", "t": t,
                                   "payload": {"note": note}, "confidence": 1.0,
                                   "source": "manual"})
        if self._live:
            self._live.push_event("manual_mark", t, {"note": note}, 1.0, "manual")
        if self._record_session_id:
            self.db.add_event(self._record_session_id, t, "manual_mark",
                              {"note": note}, 1.0, "manual")

    def stop_recording(self) -> tuple[str, Path | None, float]:
        sid = self._record_session_id
        path = None
        dur = 0.0
        if self.recorder:
            path = self.recorder.stop()
            dur = self.recorder.duration
        if sid:
            self.db.update_session(sid, duration=dur, audio_path=str(path or ""))
            for ev in self._audio_events:
                self.db.add_event(sid, ev["t"], ev["kind"], ev["payload"],
                                  ev.get("confidence", 1.0), ev.get("source", "audio_fp"))
        self.recorder = None
        self._record_session_id = ""
        self._live = None
        return sid, path, dur

    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def live_tick(self) -> list[Knot]:
        if not self._live or not self.recorder:
            return []
        return self._live.maybe_run(self.recorder.duration)

    # -- 转写 -----------------------------------------------------------
    def transcribe(self, session_id: str, audio_path: str | Path | None = None,
                   precise: bool = False, asr=None) -> tuple[ASRResult, str]:
        from .asr import build_candidates

        row = self.db.get_session(session_id)
        if row is None:
            raise RuntimeError(f"会话不存在：{session_id}")
        raw_path = audio_path or row["audio_path"] or ""
        path = Path(raw_path) if str(raw_path).strip() else None
        # Path("") 会变成 "."，SidecarASR 拿去打开目录会 PermissionError —— 这里必须收敛成 None
        if path is not None and not path.is_file():
            path = None

        if asr is not None:
            cands: list[tuple[Any, str]] = [(asr, "指定后端")]
        else:
            cands = build_candidates(self.settings, path, precise)
        if not cands:
            msg = ("没有可用的转写后端。请到设置页配一项：填 api_key / 装本地 Whisper"
                   "（「本地转写」一栏）/ 放一份同名字幕（.srt）/ "
                   "或在设置里显式打开「允许演示转写」（合成文本，仅用于看流程）")
            self.db.update_session(session_id, state="error", error=msg)
            raise RuntimeError(msg)

        # 逐个真调用：available() 只能验到"主机可达"，云端端点没转写接口要调用时才暴露
        result = None
        errors: list[str] = []
        used = ""
        for backend, hint in cands:
            try:
                if path and path.exists():
                    result = backend.transcribe(path, precise=precise)
                else:
                    try:
                        result = backend.transcribe(None, precise=precise)
                    except TypeError as e:
                        raise RuntimeError("该转写后端需要音频文件，但会话没有音频") from e
                used = hint
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{hint}→{type(e).__name__}: {str(e)[:110]}")
                continue
        if result is None:
            msg = "所有转写后端都失败：" + " ｜ ".join(errors)
            self.db.update_session(session_id, state="error", error=msg)
            raise RuntimeError(msg)

        self.db.replace_segments(session_id, result.as_dicts(), pass_no=2 if precise else 1)
        if result.is_real:
            self.db.update_session(session_id, state="transcribed", error="")
        else:
            self.db.update_session(session_id, state="transcribed")
        note = result.note
        if errors:
            # 把第一个失败原因也带上：将来"为什么没用云端"能直接查出来
            note = f"{note}（前面 {len(errors)} 个后端失败：{errors[0][:90]}）"
        self.db.set_kv(BACKEND_NOTE_KEY, f"{result.backend}｜{used}｜{note}")
        dur = max((s.t_end for s in result.segments), default=0.0)
        self.db.update_session(session_id, duration=dur)
        return result, note

    # -- 分析 -----------------------------------------------------------
    def import_transcript(self, path: str | Path, course_title: str,
                          session_title: str,
                          duration_hint: float | None = None) -> dict:
        """零录制路径：只用字幕/文本建立可分析的会话。

        .srt/.vtt 自带时间戳 → 时间轴是真的；
        .txt/.md 没有时间戳 → 必须给课程时长，按字数比例估算时间轴（会在返回里标 estimated）。
        """
        from .asr import _parse_srt, _split_by_punctuation

        src = Path(path)
        raw = src.read_text(encoding="utf-8", errors="replace")
        suffix = src.suffix.lower()
        estimated = False
        if suffix in (".srt", ".vtt"):
            segs = _parse_srt(raw)
        else:
            if not duration_hint or duration_hint <= 0:
                raise ValueError("纯文本没有时间戳，需要先给出课程时长（分钟）")
            text = "".join(raw.split())
            segs = _split_by_punctuation(text, 0.0, float(duration_hint))
            estimated = True
        if not segs:
            raise ValueError(f"没能从 {src.name} 里解析出任何句子")
        ensure_dirs()
        cid = self.db.ensure_course(course_title)
        sid = self.db.create_session(cid, session_title, "", source="subtitle")
        self.db.replace_segments(sid, [
            {"t_start": x.t_start, "t_end": x.t_end, "text": x.text,
             "confidence": 0.9 if not estimated else 0.5} for x in segs
        ], pass_no=1)
        dur = max(x.t_end for x in segs)
        self.db.update_session(sid, duration=dur, state="transcribed", error="")
        self.db.set_kv(BACKEND_NOTE_KEY,
                       f"字幕导入：{src.name}" + ("（时间轴按字数估算）" if estimated else ""))
        return {"session_id": sid, "segments": len(segs), "duration": dur,
                "estimated_timeline": estimated, "source": src.name}

    def probe_url(self, url: str, sessdata: str = "") -> dict:
        from . import source_url as su

        p = su.probe(url, sessdata)
        return {"ok": p.ok, "platform": p.platform, "title": p.title,
                "duration": p.duration, "n_pages": p.n_pages,
                "has_subtitle": p.has_subtitle, "langs": p.subtitle_langs,
                "note": p.note, "error": p.error,
                "audio_mb_estimate": round(
                    su.estimate_audio_mb(p.duration) if p.duration else 0.0, 1)}

    def import_url(self, url: str, course_title: str, session_title: str,
                   sessdata: str = "", prefer: str = "auto",
                   progress: Callable[[str], None] | None = None) -> dict:
        """从链接导入：先试字幕（零转写），不行再只下音频转写。

        prefer: auto | subtitle | audio
        返回里 source 说明到底走了哪条路，界面必须如实展示。
        """
        from . import source_url as su

        def say(msg: str) -> None:
            if progress:
                progress(msg)

        ensure_dirs()
        p = su.probe(url, sessdata)
        if not p.ok:
            raise RuntimeError(p.error or "链接探测失败")
        title = p.title or session_title or "未命名"
        segs: list[dict] = []
        how = ""

        want_sub = prefer in ("auto", "subtitle")
        want_audio = prefer in ("auto", "audio")
        if want_sub:
            try:
                if p.platform == "bilibili":
                    segs, how = su.bili_subtitle_segments(url, sessdata)
                else:
                    say("正在取字幕…")
                    segs, how = su.ytdlp_subtitle_segments(url, cookie_file="")
                say(f"拿到字幕：{how}")
            except Exception as e:  # noqa: BLE001
                say(f"取字幕失败：{str(e)[:120]}")
                if prefer == "subtitle":
                    raise
        if not segs:
            if not want_audio:
                raise RuntimeError("没有字幕，且你选择了只取字幕")
            # 先确认转写后端真的可用，别下了几十上百 MB 才发现没法转
            from .asr import build_asr

            try:
                build_asr(self.settings)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"没有可用的转写后端，先不下载音频：{str(e)[:120]}") from e
            say("没有可用字幕，改为只下载音轨（不下载画面）…")
            path, how = su.download_audio(url, AUDIO_DIR)
            say(how)
            cid = self.db.ensure_course(course_title)
            sid = self.db.create_session(cid, session_title or title, str(path),
                                        source="url_audio")
            self.db.update_session(sid, duration=p.duration)
            result, note = self.transcribe(sid)
            return {"session_id": sid, "segments": len(result.segments),
                    "duration": float(max((x.t_end for x in result.segments),
                                          default=0.0)),
                    "estimated_timeline": False, "source": "下载音轨 + 转写",
                    "how": how, "title": title, "needs_asr": True}

        cid = self.db.ensure_course(course_title)
        sid = self.db.create_session(cid, session_title or title, "",
                                    source="url_subtitle")
        self.db.replace_segments(sid, segs, pass_no=1)
        if p.platform == "bilibili":
            try:
                from . import source_url as _su

                bvid = _su.bvid_of(url)
                view = _su._bili_view(bvid, sessdata)
                self.db.set_kv(f"url_ref:{sid}",
                               {"bvid": bvid, "cid": int(view["cid"])})
            except Exception:  # noqa: BLE001
                pass     # 记不住就退化：仍可分析文本，只是没有画面取证
        # 时长取字幕结尾与视频真实时长中较大的那个：字幕可能只覆盖一部分，
        # 用真实时长更诚实（界面显示的是这节课多长，不是字幕到哪）。
        dur = max(float(x["t_end"]) for x in segs)
        if p.duration:
            dur = max(dur, float(p.duration))
        self.db.update_session(sid, duration=dur, state="transcribed", error="")
        self.db.set_kv(BACKEND_NOTE_KEY, f"链接字幕导入：{how}")
        return {"session_id": sid, "segments": len(segs), "duration": dur,
                "estimated_timeline": False, "source": "链接字幕", "how": how,
                "title": title, "needs_asr": False}

    def session_url_ref(self, session_id: str) -> dict:
        return self.db.get_kv(f"url_ref:{session_id}", {}) or {}

    def visual_evidence(self, session_id: str, times: Sequence[float],
                        sessdata: str = "") -> list[dict]:
        """按时间点取画面（B站故事板，不下载视频）。返回 [{t, png, error}]。"""
        from . import source_url as su

        ref = self.session_url_ref(session_id)
        if not ref.get("bvid"):
            raise RuntimeError("这个会话没有关联视频链接，取不到画面（只有链接导入的会话有）")
        sb = su.bili_storyboard(ref["bvid"], int(ref["cid"]), sessdata)
        out = []
        for t in times:
            try:
                png = su.storyboard_frame_png(sb, float(t), sessdata)
                out.append({"t": float(t), "png": png, "error": ""})
            except Exception as e:  # noqa: BLE001
                out.append({"t": float(t), "png": b"", "error": str(e)[:120]})
        return out

    def attach_visual_evidence(self, session_id: str, sessdata: str = "",
                               max_knots: int = 6) -> dict:
        """给回看点补画面：取该时刻的帧 → 视觉模型识别 → 作为屏幕证据入库。

        为什么要做：回看点只看"说了什么"容易误判（比如老师在写推导，说话很少）。
        补一张当时的画面，能直接看到是不是板书/公式/PPT。
        """
        from .screen import build_extractor

        knots = self.db.get_knots(session_id) or []
        if not knots:
            return {"ok": False, "reason": "这一节还没有回看点"}
        ext = build_extractor(self.settings)
        ok, why = ext.available()
        if not ok:
            return {"ok": False, "reason": f"视觉模型不可用：{why}"}
        picked = knots[:max_knots]
        times = [float(k.t_start + k.t_end) / 2 for k in picked]
        frames = self.visual_evidence(session_id, times, sessdata)
        results = []
        for k, fr in zip(picked, frames):
            if fr["error"] or not fr["png"]:
                results.append({"t": fr["t"], "ok": False,
                                "reason": fr["error"] or "没取到画面"})
                continue
            res = ext.extract(fr["png"])
            if res.error:
                results.append({"t": fr["t"], "ok": False, "reason": res.error})
                continue
            text = (res.text or "").strip()
            if not text:
                results.append({"t": fr["t"], "ok": False,
                                "reason": "画面里没有识别到文字"})
                continue
            self.db.add_screen_note(session_id, fr["t"], text, res.kind or "slide",
                                    res.model, source="storyboard")
            results.append({"t": fr["t"], "ok": True, "chars": len(text),
                            "kind": res.kind, "text": text})
        good = [r for r in results if r["ok"]]
        return {"ok": bool(good), "tried": len(results), "saved": len(good),
                "results": results,
                "reason": "" if good else "画面都没识别出文字"}

    def add_manual_mark(self, session_id: str, t: float, note: str = "") -> int:
        """零录制也能给行为信号：用户自己说"这一分钟我没听懂"。"""
        return self.db.add_event(session_id, max(0.0, float(t)), "manual_mark",
                                 {"note": note, "manual": True}, confidence=1.0,
                                 source="manual")

    # -- 外部课程导入（B 站等） -------------------------------------------
    def import_series(self, course_title: str, items: Sequence[dict],
                      source: str = "bili") -> dict:
        """把一门课的分集清单建成"待录的节"。

        每一节只存**元信息**（标题、时长、原始链接），不存视频、不转写——
        视频还在 B 站服务器上，我们只负责"到点去听"。这样才敢一次导入 20 集：
        没录的节占几百字节，不是几百兆。

        去重按 video_url：同一集导两次不该出现两行。
        """
        course = (course_title or "").strip() or "网页视频·网课"
        cid = self.db.ensure_course(course)
        try:
            have = {str(r["video_url"] or "") for r in
                    self.db.query("SELECT video_url FROM sessions WHERE course_id=?", (cid,))}
        except Exception:                              # noqa: BLE001
            have = set()
        created, skipped = 0, 0
        for it in items or []:
            url = str(it.get("url") or "").strip()
            title = str(it.get("title") or "").strip() or "未命名"
            if not url or url in have:
                skipped += 1
                continue
            sid = self.db.create_session(cid, title, source=source)
            self.db.update_session(sid, video_url=url,
                                   duration=float(it.get("duration") or 0),
                                   state="imported")
            have.add(url)
            created += 1
        return {"ok": True, "course_id": cid, "created": created, "skipped": skipped}

    def watched_sessions(self) -> list[dict]:
        """已经录到音频、值得进知识库的节（没录的不算：空的不能算知识）。"""
        out = []
        for r in self.db.query(
                "SELECT id,title,state,duration FROM sessions "
                "WHERE audio_path<>'' ORDER BY created_at DESC"):
            if int(r["duration"] or 0) <= 0:
                continue
            n = self.db.query("SELECT COUNT(*) c FROM segments WHERE session_id=?",
                              (r["id"],))
            if not n or int(n[0]["c"] or 0) == 0:
                continue
            out.append({"id": r["id"], "title": r["title"], "state": r["state"]})
        return out

    # -- 知识库（用户导入的资料 + 每节课自动归档） -------------------------
    def kb_ingest_session(self, sid: str) -> dict:
        """把一节**看过的课**归档进知识库：转写正文 + 回看点，都带时间码可定位。

        为什么必须有这一步：知识库以前只装用户上传的讲义，问模型时"课里讲过什么"
        只能靠临时检索回看点上下文，答完就散了。归档之后，同一门课里"第 2 讲说过
        的那句话"能被第 5 讲的问题检索到——时间码就是它的出处。

        幂等：path 固定成 `session:<sid>`，重跑先删旧的再写新的。
        """
        from . import kb as kbmod

        row = self.db.get_session(sid)
        if not row:
            return {"ok": False, "reason": "没有这一节"}
        segs = [dict(r) for r in self.db.get_segments(sid)]
        if not segs:
            return {"ok": False, "reason": "这一节还没有转写文本"}
        dur = float(row["duration"] or 0)
        title = str(row["title"] or "未命名")
        course_id = str(row["course_id"] or "")

        blocks: list[dict] = []
        # 开头一块：这一节讲了什么（回看点清单 + 术语），检索"哪一讲讲过 X"时最常命中它
        knots = [dataclasses.asdict(k) for k in self.db.get_knots(sid)]
        head = [f"《{title}》课堂记录（时长 {_hms(int(dur))}，共 {len(segs)} 段转写）"]
        if knots:
            head.append("本节回看点：" + "；".join(
                f"{_hms(int(k['t_start']))} {k['evidence'] or '卡住'}" for k in knots[:8]))
        blocks.append({"text": "\n".join(head), "source_ref": "00:00:00",
                       "section": "本节概要"})
        # 正文：每 60 秒左右一块，时间码就是出处
        buf: list[str] = []
        start_t = float(segs[0]["t_start"])
        for sgm in segs:
            buf.append(str(sgm["text"] or ""))
            span = float(sgm["t_end"]) - start_t
            if span >= 60.0:
                blocks.append({"text": " ".join(buf), "source_ref": _hms(int(start_t)),
                               "section": title})
                buf = []
                start_t = float(sgm["t_end"])
        if buf:
            blocks.append({"text": " ".join(buf), "source_ref": _hms(int(start_t)),
                           "section": title})
        # 回看点单独成块：它的措辞是"你在这里卡住了"，检索"我哪里没听懂"最容易命中
        for k in knots:
            body = f"{_hms(int(k['t_start']))} {k['evidence'] or ''}"
            if k.get("excerpt"):
                body += "\n原话：" + str(k["excerpt"])
            if k.get("score"):
                body += "\n判定：" + f"{k['grade']}（{float(k['score']):.2f}）"
            blocks.append({"text": body, "source_ref": _hms(int(k["t_start"])),
                           "section": f"回看点 {_hms(int(k['t_start']))}"})

        chunks = kbmod.chunk_blocks(blocks)
        if not chunks:
            return {"ok": False, "reason": "切不出可检索的块"}
        path_key = f"session:{sid}"
        old = self.db.kb_doc_by_path(path_key)
        if old is not None:
            self.db.delete_kb_doc(int(old["id"]))
        chars = sum(len(b["text"]) for b in blocks)
        doc_id = self.db.add_kb_doc(
            f"{title}（课堂记录）", path_key,
            f"来自课堂转写·{len(segs)} 段·{len(knots)} 个回看点", chars, chunks,
            course_id=course_id)
        try:
            self.db.set_kv("kb_sig", self.kb_signature()["sig"])
        except Exception:                              # noqa: BLE001
            pass
        return {"ok": True, "doc_id": doc_id, "chunks": len(chunks), "chars": chars}

    def kb_ingest_all(self) -> dict:
        """把所有"看过的课"批量归档（设置页一个按钮，导入历史也一起补齐）。"""
        done, failed = 0, 0
        for s in self.watched_sessions():
            try:
                if self.kb_ingest_session(s["id"]).get("ok"):
                    done += 1
                else:
                    failed += 1
            except Exception:                          # noqa: BLE001
                failed += 1
        return {"ok": True, "done": done, "failed": failed}

    # -- 知识库（用户导入的资料） -----------------------------------------
    def kb_import(self, path: str, course_id: str = "", title: str = "") -> dict:
        from . import kb as kbmod

        blocks, note = kbmod.read_document_blocks(path)
        if not blocks:
            return {"ok": False, "reason": "文件里没有可用的文字内容"}
        # 在每个定位块（页/小节/字幕条）内部再切块，块继承定位信息。
        # 跨页的块无法回答「这句话在第几页」，所以这里按块切而不是按全文切。
        chunks = kbmod.chunk_blocks(blocks)
        if not chunks:
            return {"ok": False, "reason": "文件里没有可用的文字内容"}
        text = "\n\n".join(b["text"] for b in blocks)
        name = title or Path(path).stem
        # 绑到不存在的课程 = 这份资料在"问模型"里永远检索不到（按课程过滤）。
        # 这属于静默失效，宁可当作全局知识库并明确告诉用户。
        warn = ""
        if course_id and not self.db.get_course(course_id):
            warn = f"没有课程 {course_id}，这份资料已作为全局知识库导入"
            course_id = ""
        doc_id = self.db.add_kb_doc(name, str(path), note, len(text), chunks,
                                    course_id=course_id)
        located = sum(1 for c in chunks if c.get("source_ref"))
        try:
            self.db.set_kv("kb_sig", self.kb_signature()["sig"])
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "id": doc_id, "title": name, "chars": len(text),
                "chunks": len(chunks), "blocks": len(blocks), "located": located,
                "note": note, "warning": warn}

    def kb_docs(self, course_id: str | None = None) -> list[dict]:
        return [{"id": int(r["id"]), "title": r["title"],
                 "course_id": r["course_id"], "chars": int(r["chars"] or 0),
                 "chunks": int(r["n_chunks"] or 0), "note": r["note"],
                 "path": r["path"]} for r in self.db.list_kb_docs(course_id)]

    def kb_search(self, query: str, course_id: str = "", k: int = 5) -> list[dict]:
        from .chat import kb_search

        return kb_search(self.db, query, course_id, k)

    # -- 知识库索引签名：改了切块参数到底要不要重建 ----------------------
    def kb_signature(self) -> dict:
        """当前索引"是什么参数、装了什么"的指纹。

        为什么需要：切块参数（600/60）、BM25 参数、以及每份资料的块数都会影响检索结果。
        没有指纹时，用户调了参数只能靠"感觉不对"来判断要不要重建——
        这正是当初 `kb_import` 改按页切之后最容易踩的静默失效。
        """
        from . import kb as kbmod

        docs = self.db.query(
            "SELECT id, title, n_chunks, chars FROM kb_docs ORDER BY id") or []
        payload = {
            "chunk": [kbmod.MAX_CHUNK, kbmod.OVERLAP],
            "docs": [[int(d["id"]), int(d["n_chunks"] or 0)] for d in docs],
        }
        sig = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return {"sig": sig, "chunk": payload["chunk"], "docs": len(docs),
                "chunks": sum(int(d["n_chunks"] or 0) for d in docs)}

    def kb_index_status(self) -> dict:
        """当前指纹 vs 上次记录的指纹。stale=True 就该提示重建。"""
        cur = self.kb_signature()
        stored = self.db.get_kv("kb_sig", "") or ""
        return {**cur, "stored": stored, "stale": bool(stored) and stored != cur["sig"],
                "has_index": cur["chunks"] > 0}

    def rebuild_kb_index(self) -> dict:
        """按当前参数重切所有资料。读不回原文件的那份会明确报出来，不静默跳过。"""
        from . import kb as kbmod

        rebuilt, missing, failed = 0, [], []
        for d in self.db.query("SELECT id, title, path FROM kb_docs ORDER BY id") or []:
            p = Path(str(d["path"] or ""))
            try:
                if not p.exists():
                    missing.append(str(d["title"]))
                    continue
                blocks, _note = kbmod.read_document_blocks(p)
                chunks = kbmod.chunk_blocks(blocks)
                if not chunks:
                    failed.append(str(d["title"]))
                    continue
                self.db.replace_kb_chunks(int(d["id"]), chunks)
                rebuilt += 1
            except Exception as e:  # noqa: BLE001
                failed.append(f"{d['title']}（{type(e).__name__}）")
        sig = self.kb_signature()
        self.db.set_kv("kb_sig", sig["sig"])
        return {"ok": True, "rebuilt": rebuilt, "missing": missing, "failed": failed,
                "sig": sig["sig"], "chunks": sig["chunks"]}

    # -- 对话 -------------------------------------------------------------
    def ask(self, question: str, session_id: str = "", use_kb: bool = True,
            use_context: bool = True, images: Sequence[str] = (),
            files: Sequence[tuple[str, str]] = ()) -> dict:
        """问一次。images 是 data URL（截图），files 是 [(文件名, 正文)]。"""
        from .chat import ask as _ask

        return _ask(self.db, self.llm, session_id, question,
                    use_kb=use_kb, use_context=use_context,
                    images=images, files=files)

    def chat_history(self, session_id: str, limit: int = 50) -> list[dict]:
        return [{"role": r["role"], "content": r["content"],
                 "at": r["created_at"]} for r in self.db.get_chat(session_id, limit)]

    def clear_chat(self, session_id: str) -> None:
        self.db.clear_chat(session_id)

    # -- 屏幕内容（F3） ---------------------------------------------------
    def capture_screen_note(self, session_id: str, image_bytes: bytes | None = None,
                            t: float | None = None, text: str | None = None,
                            kind: str = "", source: str = "vision") -> dict:
        """抓一帧屏幕 → 视觉模型识别 → 落库。

        text 直接给出时不调模型（用户在界面里改了识别结果，或手工补一条）。
        """
        row = self.db.get_session(session_id)
        if row is None:
            raise RuntimeError(f"会话不存在：{session_id}")
        model = ""
        latency = 0.0
        if text is None:
            from .screen import build_extractor, grab_primary_screen

            img = image_bytes if image_bytes is not None else grab_primary_screen()
            ext = build_extractor(self.settings)
            res = ext.extract(img)
            model = res.model
            latency = res.latency_s
            if res.error:
                raise RuntimeError(f"视觉模型识别失败：{res.error}")
            text = res.text
            kind = kind or res.kind
        text = (text or "").strip()
        if not text:
            return {"ok": False, "reason": "画面里没有识别到课程文字（可能是纯视频/黑屏）",
                    "model": model, "latency_s": latency}
        if t is None:
            t = float(row["duration"] or 0.0) or (
                max((float(r["t"]) for r in self.db.get_screen_notes(session_id)), default=0.0)
                + 60.0)
        nid = self.db.add_screen_note(session_id, float(t), text,
                                      kind or "slide", model, source)
        dur = max(float(t) + 30.0, float(row["duration"] or 0.0))
        self.db.update_session(session_id, duration=dur)
        return {"ok": True, "id": nid, "t": float(t), "text": text,
                "kind": kind or "slide", "model": model, "latency_s": latency,
                "chars": len(text)}

    def _screen_segments(self, session_id: str) -> list[dict]:
        """把屏幕笔记转成"带时间跨度的内容段"。

        每条的跨度 = 到下一条的间隔，也就是这页在屏幕上停留了多久。
        这个跨度会被 features 用成"长停留"信号（屏幕内容没有语音停顿这回事）。
        """
        notes = list(self.db.get_screen_notes(session_id))
        if not notes:
            return []
        out: list[dict] = []
        for i, r in enumerate(notes):
            t0 = float(r["t"])
            if i + 1 < len(notes):
                t1 = float(notes[i + 1]["t"])
            else:
                t1 = t0 + 120.0
            if t1 - t0 < 5.0:
                t1 = t0 + 5.0
            out.append({"t_start": t0, "t_end": t1, "text": str(r["text"]),
                        "speaker": "", "confidence": 0.85, "_kind": "screen"})
        return out

    def _backfill_audio_events(self, session_id: str, row) -> None:
        """给已有录音补算「回溯 / 暂停」事件，只算一次。

        在线录制时这两个检测器在跑；但导入的文件、或者当时设备抽风没跑到的情况，
        事后得能补上——否则"你回退了 40 秒"这种最强证据会整段缺失。
        纯读 WAV，不联网、不调模型，45 分钟音频大约 2–4 秒。
        """
        from . import capture as cap

        path = (row["audio_path"] or "") if row is not None else ""
        if not path or not Path(path).exists():
            return
        if self.db.get_kv(f"evt_audio:{session_id}", 0):
            return
        try:
            extra = cap.detect_audio_events(
                path,
                min_silence_s=self.settings.capture.pause_min_s,
                long_silence_s=self.settings.capture.pause_long_s,
            )
        except Exception:  # noqa: BLE001
            return
        seen = {(r["kind"], round(float(r["t"]) / 3.0))
                for r in self.db.get_events(session_id)}
        for ev in extra:
            key = (ev["kind"], round(float(ev["t"]) / 3.0))
            if key in seen:
                continue
            seen.add(key)
            self.db.add_event(session_id, float(ev["t"]), ev["kind"],
                              ev.get("payload") or {}, float(ev.get("confidence", 0.7)),
                              ev.get("source") or "audio_fp")
        self.db.set_kv(f"evt_audio:{session_id}", 1)

    def analyze_session(self, session_id: str) -> AnalysisResult:
        row = self.db.get_session(session_id)
        if row is None:
            raise RuntimeError(f"会话不存在：{session_id}")
        course = self.db.query("SELECT * FROM courses WHERE id=?", (row["course_id"],))
        subject = course[0]["subject"] if course else "通用"
        segs = [dict(r) for r in self.db.get_segments(session_id)]
        self._backfill_audio_events(session_id, row)
        events = [
            Evt(t=float(r["t"]), kind=r["kind"],
                payload=_safe_json(r["payload_json"]), confidence=float(r["confidence"] or 1.0),
                source=r["source"] or "manual")
            for r in self.db.get_events(session_id)
        ]
        screen_segs = self._screen_segments(session_id)
        kind = "speech"
        if screen_segs and segs:
            kind = "mixed"
        elif screen_segs:
            kind = "screen"
        if screen_segs:
            # 屏幕内容是同一时间轴上的另一路输入：一起喂给特征计算，
            # 但不能覆盖语音段（所以是"追加"而不是"替换"）
            segs = list(segs) + screen_segs
            segs.sort(key=lambda x: float(x["t_start"]))
        res = analyze(segs, events, self.settings.cognitive, subject=subject or "通用",
                      include_personal=self.settings.ui.show_personal_quadrant,
                      content_kind=kind)
        self.db.replace_windows(session_id, res.windows)
        # 纠正过的回看点不该被"重新分析"复活：先记下来，按区间把新算出的同位置回看点
        # 重新标成 dismissed（用户已经判过"这里不算"）。
        dismissed_spans = [
            (float(k["t_start"]), float(k["t_end"]))
            for k in self.db.query(
                "SELECT t_start,t_end FROM knots WHERE session_id=? AND status='dismissed'",
                (session_id,))
        ]
        knot_ids = self.db.replace_knots(session_id, res.knots)
        if dismissed_spans:
            for kid, k in zip(knot_ids, res.knots):
                mid = (float(k.t_start) + float(k.t_end)) / 2.0
                if any(a - 5.0 <= mid <= b + 5.0 for a, b in dismissed_spans):
                    self.db.set_knot_status(int(kid), "dismissed")
        self._link_knowledge(row, res, knot_ids)
        self.db.update_session(session_id, state="analyzed", error="")
        self._finalize_audio(session_id)
        return res

    # ------------------------------------------------------------------
    # 音频收尾：压缩 / 清理
    # ------------------------------------------------------------------
    def _finalize_audio(self, session_id: str) -> dict:
        """分析完成后处理音频文件：要么压缩，要么删除。

        一节 90 分钟的系统回环录音，44.1k 立体声 WAV 是 890 MB。转写只需要 16k
        单声道，压成 FLAC 约 95 MB，而且压缩后仍能重新转写、重新分析。
        任何一步失败都只记录并保留原文件——不能因为"省空间"把证据弄丢。
        """
        from . import audio_io

        row = self.db.get_session(session_id)
        if row is None:
            return {"ok": False, "reason": "no-session"}
        raw = Path(row["audio_path"] or "")
        if not raw or not raw.exists():
            return {"ok": False, "reason": "no-audio"}
        cfg = self.settings.capture
        try:
            if not getattr(cfg, "save_raw_audio", True):
                size = raw.stat().st_size
                raw.unlink()
                self.db.update_session(session_id, audio_path="")
                return {"ok": True, "action": "deleted", "bytes": size}
            rate = int(getattr(cfg, "audio_flac_rate", 16000) or 16000)
            if (getattr(cfg, "audio_compress", True) and raw.suffix.lower() == ".wav"
                    and audio_io.info(raw)["rate"] > rate):
                dst = raw.with_suffix(".flac")
                audio_io.compress_to_flac(raw, dst, rate=rate)
                before, after = raw.stat().st_size, dst.stat().st_size
                self.db.update_session(session_id, audio_path=str(dst))
                # 删原件这一步单独处理：录音的句柄偶尔还没松手（Windows 上
                # WinError 32）。压缩已经成功、路径也已经指向 flac，这时候
                # 报一条"音频处理失败"只会让用户以为录音坏了。
                for _ in range(3):
                    try:
                        raw.unlink()
                        break
                    except OSError:
                        time.sleep(0.2)
                return {"ok": True, "action": "compressed", "bytes": after,
                        "saved": max(0, before - after), "path": str(dst),
                        "raw_left": raw.exists()}
        except Exception as e:             # noqa: BLE001
            self.db.update_session(session_id, error=f"音频处理失败（保留原文件）：{e}"[:400])
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "action": "kept"}

    # ------------------------------------------------------------------
    # 回看点纠正
    # ------------------------------------------------------------------
    def dismiss_knot(self, knot_id: int, note: str = "用户认为这里不算回看点") -> None:
        """人工否决一个回看点：状态置 dismissed + 留一条反馈。

        为什么不删行：误判的位置和特征恰恰是以后调权重的样本，删了就没了。
        """
        self.db.set_knot_status(int(knot_id), "dismissed")
        self.db.add_feedback(int(knot_id), "dismissed", note)

    def restore_knot(self, knot_id: int) -> None:
        self.db.set_knot_status(int(knot_id), "pending")

    def confirm_knot(self, knot_id: int) -> None:
        """确认"确实卡住"：不改状态（还是待解决），只记一条正反馈。"""
        self.db.add_feedback(int(knot_id), "confirm_useful", "用户确认这是真回看点")

    def _link_knowledge(self, session_row, res: AnalysisResult, knot_ids: Sequence[int]) -> None:
        """把回看点挂到知识点，并建立共现边（供 3.5 漏洞地图的复习顺序使用）。

        知识点**不再直接取词表**：词表里的「定义 / 必要条件 / 边界条件」是元词，
        描述"这是哪种命题"，不描述"这一节教了什么"。它们出现在字面里恰好说明附近有术语，
        所以仍用于术语密度计分（features.py），但不能当知识点。
        真正的术语走 ``terms`` 那一层：定义句式 + 术语后缀 + 学科词表，有模型时再加一层提名（需原文可查）。
        """
        if not res.windows:
            return
        tr_text = "".join(w.text for w in res.windows)
        per_knot: dict[int, str] = {}
        for kid, knot in zip(knot_ids, res.knots):
            per_knot[int(kid)] = (knot.excerpt or "") + " " + " ".join(
                w.text for w in res.windows if knot.t_start <= w.center <= knot.t_end)
        origin = self._apply_terms(session_row, tr_text, per_knot)
        return origin

    def _apply_terms(self, session_row, course_text: str, per_knot: dict) -> str:
        """抽出术语 → 挂到回看点 → 建共现边。返回术语来源（llm / rule）。"""
        from . import terms as terms_mod

        learn = terms_mod.learn_terms(self.llm, course_text, max_terms=16)
        ranked = [t["term"] for t in learn.get("terms") or [] if t.get("term")]
        origin = str(learn.get("source") or "rule")
        if not ranked:
            return origin
        cid = session_row["course_id"]
        sid = session_row["id"]
        # 「第一次出现的时间」要真的按转写算：先出现的术语才是后面的前置，
        # 全填 0 会让每个知识点都变成彼此的前置（互相锁死，顺序就没意义了）。
        segs = self.db.get_segments(sid) or []

        def _first_mention(term: str) -> float:
            for sg in segs:
                if term in str(sg["text"] or ""):
                    return float(sg["t_start"] or 0)
            return 0.0

        touched: set[int] = set()
        # 一个术语最多挂 2 个回看点，而且挂在"讲得最多"的那几个上。
        # 否则每个回看点都会挂同一批词，"漏洞地图"看起来全是同一个知识点。
        for term in ranked:
            hits = sorted(((kid, text.count(term)) for kid, text in per_knot.items()
                           if term in text), key=lambda x: -x[1])[:2]
            if not hits:
                continue
            kp = self.db.upsert_knowledge_point(cid, term, _first_mention(term), sid,
                                                origin=origin)
            touched.add(int(kp))
            for kid, cnt in hits:
                self.db.link_knot_point(int(kid), kp, float(cnt))
        # 共现边：这一节碰到的术语 ↔ 全课其他知识点，就够表示"它们一起出现过"。
        # 不做全量两两配对——100 个知识点就是 9900 条边，而且每分析一节都要重写一遍。
        ids = [int(r["id"]) for r in self.db.get_knowledge_points(cid)]
        for a in touched:
            for b in ids:
                if a != b:
                    self.db.add_edge(a, b, "co_occur", 1.0)
        # 每次重链接都顺手清掉"既没被验证过、也没挂上任何回看点"的知识点：
        # 否则规则升级后，旧词表垃圾会永远留在表里（它只是不再被引用）
        self.db.prune_orphan_points(cid)
        self.db.retire_orphan_points(cid)
        return origin

    def rebuild_knowledge(self, course_id: str | None = None, use_llm: bool = True) -> dict:
        """不重听音频，只按现存窗口/回看点重建知识点。词表升级后必须能一键重算。"""
        from . import terms as terms_mod

        courses = [course_id] if course_id else [
            str(r["id"]) for r in self.db.query("SELECT id FROM courses ORDER BY id") or []]
        out = []
        for cid in courses:
            sessions = self.db.list_sessions(cid) or []
            before_names = {str(r["name"]) for r in self.db.get_knowledge_points(cid)}
            terms_seen: dict[str, str] = {}
            for sess in sessions:
                sid_ = str(sess["id"])
                segs = self.db.get_segments(sid_) or []
                knots = self.db.get_knots(sid_) or []
                if not segs or not knots:
                    continue
                # windows 表只存分数，正文在 segments 里：重建时按时间段取原话
                def _text_in(a: float, b: float) -> str:
                    return " ".join(str(s["text"] or "") for s in segs
                                    if float(s["t_end"]) >= a and float(s["t_start"]) <= b)
                text = _text_in(0.0, 1e9)
                per_knot = {int(k.id): (k.excerpt or "") + " " + _text_in(float(k.t_start),
                                                                        float(k.t_end))
                            for k in knots}
                if use_llm and getattr(self.llm, "configured", False):
                    origin = self._apply_terms(sess, text, per_knot)
                else:
                    saved, self.llm = self.llm, _Unconfigured()
                    try:
                        origin = self._apply_terms(sess, text, per_knot)
                    finally:
                        self.llm = saved
                terms_seen[str(sess["id"])] = origin
            self.db.prune_orphan_points(cid)
            self.db.retire_orphan_points(cid)
            after_names = {str(r["name"]) for r in self.db.get_knowledge_points(cid)}
            gone = before_names - after_names
            # 退役的不算删除：它在库里还在，只是不再参与"该学什么"
            retired_now = {str(r["name"]) for r in
                           self.db.get_knowledge_points(cid, include_retired=True)
                           if int(r["retired"] or 0)}
            out.append({"course_id": cid, "sessions": len(sessions),
                        "origin": terms_seen,
                        "pruned": sorted(gone - retired_now),
                        "retired": sorted(gone & retired_now),
                        "remaining": sorted(after_names)})
        return {"ok": True, "courses": out}

    # -- 掌握度：回看点 → 知识点 → 会不会 ---------------------------------
    def record_kp_attempt(self, kp_id: int, correct: float, source: str = "manual",
                          knot_id: int | None = None) -> dict:
        """记一次"这个知识点会不会"的证据，并按近因加权重算掌握度。

        掌握度不直接累加，而是每次都从 kp_attempts 全历史重算——
        这样规则改了（权重、封顶）旧数据也能按新规则解释，不会留下一堆口径不一的结果。
        """
        from .cognitive import mastery as kpm

        row = self.db.get_kp(kp_id)
        if row is None:
            raise RuntimeError(f"知识点不存在：{kp_id}")
        self.db.add_kp_attempt(kp_id, float(correct), source, knot_id)
        hist = [float(r["correct"]) for r in self.db.get_kp_attempts(kp_id, limit=50)]
        kp_type = str(row["kp_type"] or "concept")
        if source in kpm.QUALITATIVE_PASS_ACTIONS:
            # 用户明确说"本来就会 / 讲明白了"：定性通过，直接记 1.0
            score = 1.0
        else:
            score = kpm.compute_mastery(hist)
        gate = kpm.gate_threshold(kp_type)

        # 记忆稳定度：从上次证据到现在隔了多久，决定这次涨多少
        now = time.time()
        prev_at = float(row["last_evidence_at"] or 0)
        prev_s = float(row["stability"] or 0) or kpm.DEFAULT_STABILITY_DAYS
        elapsed_d = (now - prev_at) / 86400.0 if prev_at else 0.0
        if source in kpm.QUALITATIVE_PASS_ACTIONS:
            # 定性通过直接给一个长间隔：用户说会就是会，别第二天又来问
            stability = min(kpm.MAX_STABILITY_DAYS, max(prev_s * 2.5, 7.0))
        else:
            stability = kpm.next_stability(prev_s, float(correct) >= 0.5, elapsed_d)
        due_at = now + kpm.due_in_days(stability) * 86400.0
        self.db.set_kp_mastery(kp_id, score, len(hist), gate,
                               stability=stability, due_at=due_at)
        return {"kp_id": int(kp_id), "name": row["name"], "mastery": round(score, 3),
                "attempts": len(hist), "gate": gate,
                "stability_days": round(stability, 2),
                "due_in_h": round(kpm.due_in_days(stability) * 24, 1),
                "status": kpm.status_of(score, len(hist), kp_type)}

    def record_feedback(self, knot_id: int, action: str, note: str = "") -> dict:
        """用户对回看点的处置 → 同时更新回看点状态与相关知识点的掌握度。"""
        from .cognitive import mastery as kpm

        self.db.add_feedback(int(knot_id), action, note)
        status = "mastered" if action == "mastered" else (
            "explained" if action in ("confirm_useful", "replay_opened") else "pending")
        self.db.set_knot_status(int(knot_id), status)
        correct = kpm.ACTION_CORRECT.get(action)
        updated = []
        if correct is not None:
            for kp in self.db.get_knot_points(int(knot_id)):
                updated.append(self.record_kp_attempt(int(kp["id"]), correct,
                                                      source=action, knot_id=int(knot_id)))
        return {"action": action, "status": status, "kp": updated}

    def knot_solved(self, knot_id: int) -> bool:
        """回看点算不算"补上了"：挂在它上面的知识点全部过门槛。

        没有任何知识点挂着的回看点（术语没识别出来）一律算未解决——
        宁可让它继续待在清单上，也不能假装解决了。
        """
        from .cognitive import mastery as kpm

        kps = self.db.get_knot_points(int(knot_id))
        if not kps:
            return False
        return all(kpm.is_mastered(float(k["mastery"] or 0), str(k["kp_type"] or "concept"))
                   for k in kps)

    def kp_overview(self, course_id: str) -> list[dict]:
        """漏洞地图用的知识点清单：掌握度从低到高，未验证的排在"还不会"之后。"""
        from .cognitive import mastery as kpm

        out = []
        for r in self.db.get_knowledge_points(course_id):
            attempts = int(r["attempts"] or 0)
            mastery = float(r["mastery"] or 0)
            kp_type = str(r["kp_type"] or "concept")
            knots = self.db.kp_knots(int(r["id"]))
            out.append({
                "id": int(r["id"]), "name": r["name"],
                "mastery": round(mastery, 3), "attempts": attempts,
                "gate": float(r["gate"] or kpm.gate_threshold(kp_type)),
                "status": kpm.status_of(mastery, attempts, kp_type),
                "knots": len(knots),
                "sessions": len(json.loads(r["session_ids"] or "[]")),
            })
        order = {"weak": 0, "learning": 1, "unknown": 2, "mastered": 3}
        # 复习队列：掌握度会过期，界面要能看出"这个该复习了"
        try:
            ret = {q["id"]: q for q in self.review_queue(course_id, limit=200)}
        except Exception:  # noqa: BLE001
            ret = {}
        for item in out:
            r = ret.get(item["id"])
            if r is not None:
                item.update(retention=r["retention"], overdue=r["overdue"],
                            stability_days=r["stability_days"],
                            due_in_h=r["due_in_h"], elapsed_h=r["elapsed_h"])
        # 排序必须在合并留存之后：该复习的排最前。
        # 它的掌握度还是 100%，只按"掌握度/状态"排会把它压到最底下，
        # 等于把最该做的事藏起来。
        out.sort(key=lambda x: (0 if x.get("overdue") else 1,
                                order.get(x["status"], 9),
                                x.get("retention") if x.get("retention") is not None else 1.0,
                                -x["knots"], x["name"]))
        return out

    # -- 复习队列：掌握度会过期 ------------------------------------------
    def review_queue(self, course_id: str, limit: int = 8) -> list[dict]:
        """该复习的知识点，留存低的排前面。

        为什么要有这一层：掌握度是"现在会不会"，不是"永远会不会"。
        没有排程时，一个月前补上的知识点和昨天补上的看起来一模一样——
        这是"看起来在学、其实在漏"的最常见形态。
        """
        from .cognitive import mastery as kpm

        now = time.time()
        out = []
        for r in self.db.get_knowledge_points(course_id):
            attempts = int(r["attempts"] or 0)
            if attempts <= 0:
                continue          # 没测过的东西谈不上遗忘
            kp_type = str(r["kp_type"] or "concept")
            stability = float(r["stability"] or 0) or kpm.DEFAULT_STABILITY_DAYS
            last = float(r["last_evidence_at"] or 0) or float(r["due_at"] or 0)
            elapsed_d = (now - last) / 86400.0 if last else 0.0
            r_now = kpm.retrievability(elapsed_d, stability)
            due_d = kpm.due_in_days(stability)
            out.append({
                "id": int(r["id"]), "name": r["name"],
                "mastery": round(float(r["mastery"] or 0), 3),
                "retention": round(r_now, 3),
                "stability_days": round(stability, 2),
                "elapsed_h": round(elapsed_d * 24, 1),
                "due_in_h": round(max(0.0, due_d - elapsed_d) * 24, 1),
                "overdue": bool(kpm.review_due(r_now, due_d, elapsed_d)),
                "status": kpm.status_of(float(r["mastery"] or 0), attempts, kp_type),
                "knots": len(self.db.kp_knots(int(r["id"]))),
                "locked": float(r["mastery"] or 0) < kpm.DEFAULT_GATE,
            })
        out.sort(key=lambda x: (not x["overdue"], x["retention"], x["name"]))
        return out[:limit]

    def learning_order(self, course_id: str) -> list[dict]:
        """按依赖关系排出"该先补哪个知识点"。

        知识点之间的边（co_occur / prereq）不是装饰：早出现、且至今没补上的术语，
        就是后面那个知识点讲不通的原因。没有这一层，"下一步学什么"只能按名字排序。
        """
        from .cognitive import mastery as kpm

        rows = {int(r["id"]): r for r in self.db.get_knowledge_points(course_id)}
        if not rows:
            return []
        edges = [e for e in self.db.query(
            "SELECT src_id, dst_id, kind, weight FROM knowledge_edges") or []
            if int(e["src_id"]) in rows and int(e["dst_id"]) in rows]
        first_t = {i: float(r["first_seen_t"] or 0) for i, r in rows.items()}
        mastery = {i: float(r["mastery"] or 0) for i, r in rows.items()}

        prereq: dict[int, set[int]] = {i: set() for i in rows}
        for e in edges:
            a, b = int(e["src_id"]), int(e["dst_id"])
            # 更早出现、而且现在还没补上的，才算 b 的前置
            if a != b and first_t[a] < first_t[b] and mastery[a] < kpm.DEFAULT_GATE:
                prereq[b].add(a)

        order = sorted(rows, key=lambda i: (round(mastery[i], 3), first_t[i], rows[i]["name"]))
        # 一遍插入式修正：前置得排在它后面那个知识点之前
        for _ in range(len(order)):
            moved = False
            for idx, kp in enumerate(order):
                for p in prereq[kp]:
                    pj = order.index(p)
                    if pj > idx:
                        order.insert(idx, order.pop(pj))
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break

        out = []
        for i in order:
            r = rows[i]
            out.append({
                "id": i, "name": r["name"],
                "mastery": round(mastery[i], 3),
                "verified": int(r["attempts"] or 0) > 0,
                "locked": mastery[i] < kpm.DEFAULT_GATE,
                "prereqs": [rows[p]["name"] for p in sorted(prereq[i])],
                "blocked_by": [rows[p]["name"] for p in sorted(prereq[i])
                               if mastery[p] < kpm.DEFAULT_GATE],
                "knots": len(self.db.kp_knots(i)),
            })
        return out

    def gate_lock(self, course_id: str) -> list[dict]:
        """门槛没过、因此**只能重讲不能往下走**的知识点。

        「硬门槛」的意思是：这不是建议，是闸门。掌握度没过 90%，
        就不该让它出现在"去做题/开新节"的选项里。
        """
        out = [x for x in self.learning_order(course_id) if x["locked"]]
        # 顺序不重排，只在同一层级内微调：
        # ① 测过但没过的优先（离门槛更近，用户也有记忆在）；② 前置已经补上的优先。
        out.sort(key=lambda x: (not x["verified"], bool(x["blocked_by"])))
        return out

    def forced_strategy(self, kp_name: str, attempts: int, mastery: float) -> dict:
        """闸门锁死时，重讲策略由证据决定，不交给模型挑。

        为什么会这样：让模型"自己选一个讲法"时，它偏爱挑听起来高级的那种，
        而眼下缺的是最基础的那一层。规则化以后至少是可解释、可复现的。
        """
        from .cognitive import mastery as kpm

        if attempts <= 0:
            key, why = "prerequisite", "还没有任何证据，先讲清它从哪儿来、依赖什么"
        elif mastery < kpm.DEFAULT_GATE * 0.5:
            key, why = "analogy", "离门槛还很远（不到一半），先用类比把基本形状立住"
        elif mastery < kpm.DEFAULT_GATE:
            key, why = "layered", "只差最后一点，分层复述把细节补齐"
        else:
            key, why = "numeric", "已经过门槛，重讲只是为了加深，用具体数字走一遍"
        strat = STRATEGY_MAP.get(key)
        return {"key": key, "name": (strat.name if strat else key), "why": why}

    def _gate_strategy_for_knot(self, knot_id: int) -> str | None:
        """这个回看点挂着的知识点里只要有一个被闸门锁住，就返回强制策略。"""
        rows = self.db.query(
            """SELECT kp.name AS name, kp.attempts AS attempts, kp.mastery AS mastery
               FROM knot_points kp2 JOIN knowledge_points kp ON kp.id = kp2.kp_id
               WHERE kp2.knot_id=? ORDER BY kp2.weight DESC""", (int(knot_id),))
        from .cognitive import mastery as kpm

        for r in rows or []:
            if float(r["mastery"] or 0) < kpm.DEFAULT_GATE:
                return self.forced_strategy(r["name"], int(r["attempts"] or 0),
                                            float(r["mastery"] or 0))["key"]
        return None

    def next_action(self, course_id: str) -> dict:
        """下一步该干什么。顺序是硬的：闸门没过 → 重讲，而不是做题。

        1. 门槛没过、而且还没重讲过 → 重讲（策略由闸门定）
        2. 门槛没过但已经重讲过 → 出小测验证（重讲本身不算证据）
        3. 过了门槛但留存掉到线下的 → 复习
        4. 都干净 → 开新的一节
        """
        from .cognitive import mastery as kpm

        locked = self.gate_lock(course_id)
        if locked:
            pick = locked[0]
            knot_id = self._first_knot_for_kp(pick["id"])
            explained = False
            if knot_id is not None:
                row = self.db.query("SELECT status FROM knots WHERE id=?", (int(knot_id),))
                explained = bool(row and str(row[0]["status"]) == "explained")
            strat = self.forced_strategy(pick["name"], 1 if pick["verified"] else 0,
                                         pick["mastery"])
            if explained:
                return {"action": "quiz", "kp": pick, "knot_id": knot_id,
                        "locked": True, "strategy": strat,
                        "why": f"「{pick['name']}」已经重讲过了，但还没有证据——"
                               f"出 3 道小测才能算数（门槛 {kpm.DEFAULT_GATE * 100:.0f}%）"}
            why = (f"「{pick['name']}」"
                   + (f"只有 {pick['mastery'] * 100:.0f}%" if pick["verified"] else "还没测过")
                   + f"，没过 {kpm.DEFAULT_GATE * 100:.0f}% 的门槛：先重讲这段，"
                   + f"不要跳到做题或新内容")
            if pick["blocked_by"]:
                why += f"；它依赖「{'、'.join(pick['blocked_by'][:2])}」，建议一起补"
            elif not pick["prereqs"]:
                why += "；它不依赖别的知识点，从这里开始最省力"
            return {"action": "replay", "kp": pick, "knot_id": knot_id,
                    "locked": True, "strategy": strat, "why": why}

        queue = self.review_queue(course_id, limit=20)
        overdue = [q for q in queue if q["overdue"]]
        if overdue:
            pick = overdue[0]
            return {"action": "quiz", "kp": pick, "locked": False,
                    "why": f"「{pick['name']}」留存已掉到 {pick['retention'] * 100:.0f}%，"
                           f"再放下去就要重新学了",
                    "knot_id": self._first_knot_for_kp(pick["id"])}
        weak = [q for q in queue if not kpm.is_mastered(q["mastery"])]
        if weak:
            pick = weak[0]
            return {"action": "quiz", "kp": pick, "locked": True,
                    "why": f"「{pick['name']}」只有 {pick['mastery'] * 100:.0f}%，"
                           f"还没过 {kpm.DEFAULT_GATE * 100:.0f}% 的门槛",
                    "knot_id": self._first_knot_for_kp(pick["id"])}
        return {"action": "new", "kp": None, "locked": False,
                "why": "当前没有到期该复习、也没有没过门槛的知识点；可以开新的一节",
                "knot_id": None}

    def _first_knot_for_kp(self, kp_id: int) -> int | None:
        rows = self.db.kp_knots(int(kp_id))
        return int(rows[0]["id"]) if rows else None

    # -- 重讲 -----------------------------------------------------------
    def reword(self, knot_id: int, strategy_key: str | None = None) -> RewordResult:
        knots = {k.id: k for k in self.db.get_knots()}
        knot = knots.get(knot_id)
        if knot is None:
            return RewordResult("", "", "", True, "回看点不存在")
        # 闸门优先：门槛没过时讲法由证据决定，模型推荐排后面
        key = strategy_key or self._gate_strategy_for_knot(knot_id)             or recommend(knot.excerpt, knot.evidence_json)
        strat = STRATEGY_MAP.get(key) or STRATEGIES[0]
        row = self.db.get_session(knot.session_id)
        segs = self.db.get_segments(knot.session_id)
        before = " ".join(r["text"] for r in segs if knot.t_start - 300 <= r["t_end"] <= knot.t_start)[-400:]
        target = " ".join(r["text"] for r in segs if knot.t_start <= r["t_end"] and r["t_start"] <= knot.t_end)
        after = " ".join(r["text"] for r in segs if knot.t_end <= r["t_start"] <= knot.t_end + 300)[:400]
        history = self._history_terms(row)
        course = self.db.query("SELECT title FROM courses WHERE id=?", (row["course_id"],)) if row else []
        system, user = build_prompt(strat, target, before, after, history,
                                    course[0]["title"] if course else "")
        # 走统一规则：空返回/只回推理内容时先降档重问一次，再不行才给结构化提纲降级
        call = self.llm.complete_text(system, user, max_tokens=900)
        text, err = call.text, call.error
        if err or not text:
            # 降级提纲也算"重讲过一遍"：用户手里有可照着走的三步模板，
            # 就该进入"出小测验证"，而不是卡在重讲上出不去。
            # 但反馈里要写清是降级（replay_degraded），别把降级当成真讲过。
            self.db.add_feedback(knot_id, "replay_degraded", strat.key)
            self.db.set_knot_status(knot_id, "explained")
            return RewordResult(fallback_outline(strat, target, history), strat.key,
                                strat.name, True, err)
        self.db.add_feedback(knot_id, "replay_opened", strat.key)
        self.db.set_knot_status(knot_id, "explained")
        return RewordResult(text, strat.key, strat.name, False, "")

    def _history_terms(self, session_row) -> list[str]:
        if not session_row:
            return []
        rows = self.db.query(
            """SELECT kp.name AS name, COUNT(*) AS c FROM knot_points kp2
               JOIN knowledge_points kp ON kp.id = kp2.kp_id
               JOIN knots k ON k.id = kp2.knot_id
               WHERE kp.course_id = ? AND k.session_id != ?
               GROUP BY kp.name ORDER BY c DESC LIMIT 10""",
            (session_row["course_id"], session_row["id"]),
        )
        return [r["name"] for r in rows]

    # -- 小测：回看点 → 3 道 4 选 1 → 真实证据 -----------------------------
    def knot_source_text(self, knot_id: int, before_s: float = 300.0) -> dict:
        """取出题要用的原文：这一段的转写 + 同窗口的屏幕文字。

        出题只能基于"这一段到底讲了什么"，不许拿整节课或者别处的资料凑——
        题干跑偏比题目少更糟。
        """
        knots = {k.id: k for k in self.db.get_knots()}
        knot = knots.get(int(knot_id))
        if knot is None:
            return {"ok": False, "reason": "回看点不存在", "text": ""}
        segs = self.db.get_segments(knot.session_id)
        target = " ".join(r["text"] for r in segs
                          if knot.t_start <= r["t_end"] and r["t_start"] <= knot.t_end)
        before = " ".join(r["text"] for r in segs
                          if knot.t_start - before_s <= r["t_end"] <= knot.t_start)
        notes = []
        try:
            for r in self.db.query(
                    """SELECT text FROM screen_notes
                       WHERE session_id=? AND t BETWEEN ? AND ?
                       ORDER BY t""",
                    (knot.session_id, knot.t_start - 10, knot.t_end + 10)):
                if r["text"]:
                    notes.append(str(r["text"]))
        except Exception:  # noqa: BLE001  老库可能没有这张表
            pass
        text = target.strip() or knot.excerpt or ""
        return {"ok": bool(text), "text": text, "before": before, "notes": notes,
                "excerpt": knot.excerpt or "", "session_id": knot.session_id,
                "t_start": knot.t_start, "t_end": knot.t_end}

    def make_quiz(self, knot_id: int, k: int = 3) -> dict:
        """给一个回看点出题。返回 {ok, items, dropped, error, attempts, degraded}。

        出处（evidence）必须能在这一段原文里查到；查不到的题当场作废，
        不写掌握度——宁可少一道，也不拿噪声当证据。
        """
        from . import quiz as qz

        src = self.knot_source_text(knot_id)
        if not src.get("ok"):
            return {"ok": False, "items": [], "dropped": [], "attempts": 0,
                    "error": src.get("reason") or "这一段没有可用原文，出不了题",
                    "degraded": True}
        course_id = ""
        row = self.db.get_session(src["session_id"])
        if row is not None:
            course_id = str(row["course_id"] or "")
        snippets: list[str] = []
        if course_id:
            try:
                snippets = [h["text"] for h in self.kb_search(src["text"][:120], course_id, k=2)]
            except Exception:  # noqa: BLE001
                snippets = []
        system, user = qz.build_prompt(src["text"], src.get("excerpt") or "",
                                       snippets, src.get("notes") or [],
                                       src.get("before") or "")
        call = self.llm.complete_json(system, user, expect_key="questions", max_tokens=1200)
        if not call.ok:
            return {"ok": False, "items": [], "dropped": [], "attempts": call.attempts,
                    "error": call.error or "模型没有给出可用的题", "degraded": True}
        items, dropped = qz.parse_questions(call.payload, src["text"])
        if not items:
            return {"ok": False, "items": [], "dropped": dropped, "attempts": call.attempts,
                    "error": "题都不合规（题干/选项/出处校验没过）", "degraded": True}
        # 挂上一个知识点，答对答错才知道该更新谁的掌握度
        kps = self.db.get_knot_points(int(knot_id))
        kp_id = int(kps[0]["id"]) if kps else None
        for it in items:
            it.kp_id = kp_id
        ids = self.db.add_quiz_items(src["session_id"], int(knot_id),
                                     [i.to_row() for i in items])
        for it, iid in zip(items, ids):
            it.item_id = iid
        return {"ok": True, "items": [i.public() for i in items], "dropped": dropped,
                "attempts": call.attempts, "degraded": call.hardened,
                "void": [i.public() for i in items if i.void], "error": ""}

    def answer_quiz(self, item_id: int, choice_index: int) -> dict:
        """判一道题。4 选 1 是确定性判分；题目作废则不写掌握度。"""
        from . import quiz as qz

        rows = self.db.query("SELECT * FROM quiz_items WHERE id=?", (int(item_id),))
        if not rows:
            return {"ok": False, "error": "题目不存在", "counts": False}
        r = rows[0]
        item = qz.QuizItem(
            prompt=str(r["prompt"]),
            choices=json.loads(r["choices_json"] or "[]"),
            correct_index=int(r["correct_index"] or 0),
            evidence=str(r["evidence"] or ""),
            evidence_ok=bool(r["evidence_ok"]),
            void_reason=str(r["void_reason"] or ""),
            kp_id=int(r["kp_id"]) if r["kp_id"] is not None else None,
            item_id=int(r["id"]),
        )
        verdict = qz.grading_verdict(item, choice_index)
        self.db.answer_quiz_item(int(item_id), int(choice_index))
        updated = None
        if verdict["counts"] and item.kp_id:
            updated = self.record_kp_attempt(item.kp_id, 1.0 if verdict["correct"] else 0.0,
                                             source="quiz", knot_id=None)
        return {"ok": True, "item_id": int(item_id), **verdict,
                "correct_index": item.correct_index, "evidence": item.evidence,
                "kp": updated}

    def import_exercises(self, path: str | Path, course_id: str) -> dict:
        """导入老师发的练习册（xlsx/csv）。

        导入的题不查原文（答案是文件给的，文件就是权威），但**出题来源要标出来**：
        模型题靠原文回查防幻觉，导入题靠文件答案，两者不能混在一起算"出处可查"。
        """
        from . import exercises as ex

        rows, errors = ex.load_file(path)
        if not rows:
            return {"ok": False, "imported": 0, "skipped": 0, "errors": errors,
                    "error": "一条题也没解析出来"}
        kps = {str(r["name"]).strip(): int(r["id"])
               for r in self.db.get_knowledge_points(course_id)}
        matched, unmatched = 0, set()
        now = time.time()
        for r in rows:
            kp_id = kps.get(r.kp_name.strip()) if r.kp_name else None
            if kp_id:
                matched += 1
            elif r.kp_name:
                unmatched.add(r.kp_name.strip())
            self.db.execute(
                """INSERT INTO exercises(course_id,kp_id,kp_name,prompt,choices_json,
                                         answer_index,explain,kind,source,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (course_id, kp_id, r.kp_name, r.prompt,
                 json.dumps(r.choices, ensure_ascii=False), r.answer_index,
                 r.explain, r.kind, "import", now))
        return {"ok": True, "imported": len(rows), "skipped": len(errors),
                "errors": errors, "matched": matched,
                "unmatched": sorted(unmatched), "error": ""}

    def exercises_overview(self, course_id: str) -> dict:
        rows = self.db.query("SELECT * FROM exercises WHERE course_id=?", (course_id,)) or []
        return {"total": len(rows),
                "answered_kp": sum(1 for r in rows if r["kp_id"] is not None),
                "unmatched": sorted({str(r["kp_name"]) for r in rows
                                     if r["kp_id"] is None and r["kp_name"]}),
                "items": [{"id": int(r["id"]), "kp_name": str(r["kp_name"] or ""),
                           "prompt": str(r["prompt"]), "explain": str(r["explain"] or ""),
                           "choices": json.loads(r["choices_json"] or "[]"),
                           "answer_index": int(r["answer_index"] or 0)} for r in rows]}

    def quiz_from_exercises(self, course_id: str, session_id: str, k: int = 3,
                            kp_id: int | None = None) -> dict:
        """用导入的练习出题。没做过的优先；和模型题走同一套确定性判分。"""
        sql = "SELECT * FROM exercises WHERE course_id=?"
        args: list[Any] = [course_id]
        if kp_id is not None:
            sql += " AND kp_id=?"
            args.append(int(kp_id))
        rows = self.db.query(sql + " ORDER BY id", args) or []
        if not rows:
            return {"ok": False, "items": [], "error": "还没有导入的练习，先导入 xlsx/csv",
                    "source": "import"}
        # 这次已经出过的题先跳过，避免同一份练习反复问同一道
        done = {str(r["prompt"]) for r in (self.db.get_quiz_items(session_id) or [])
                if str(r["source"] or "llm") == "import"}
        pool = [r for r in rows if str(r["prompt"]) not in done] or list(rows)
        picked = pool[:max(1, min(int(k), len(pool)))]
        items, ids = [], []
        for r in picked:
            it = {"prompt": str(r["prompt"]),
                  "choices": json.loads(r["choices_json"] or "[]"),
                  "correct_index": int(r["answer_index"] or 0),
                  "evidence": str(r["explain"] or ""),
                  "evidence_ok": True,          # 答案来自文件，不靠原文回查
                  "void_reason": "",
                  "kp_id": int(r["kp_id"]) if r["kp_id"] is not None else None,
                  "source": "import"}
            items.append(it)
        ids = self.db.add_quiz_items(session_id, None, items)
        out = []
        for it, iid in zip(items, ids):
            out.append({**it, "item_id": iid})
        return {"ok": True, "items": out, "error": "", "source": "import",
                "void": [], "dropped": [], "attempts": 0, "degraded": False}

    def quiz_overview(self, session_id: str) -> dict:
        """这一节的小测情况 + 挂着的知识点，界面上一眼能看出"测没测过"。"""
        stats = self.db.quiz_stats(session_id)
        rows = self.db.get_quiz_items(session_id)
        return {"stats": stats,
                "items": [{"item_id": int(r["id"]), "prompt": str(r["prompt"]),
                           "answered_index": r["answered_index"],
                           "void": bool(r["void_reason"])} for r in rows]}

    # -- 待答问题队列：听课攒、课后答（ADR-004）-------------------------
    def queue_question(self, session_id: str, text: str,
                       t_sec: float | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "问题是空的"}
        if t_sec is None:
            t_sec = self.current_position(session_id)
        excerpt = ""
        segs = self.db.get_segments(session_id)
        if segs:
            near = min(segs, key=lambda r: abs(float(r["t_start"]) - float(t_sec)))
            excerpt = str(near["text"] or "")[:300]
        qid = self.db.add_question(session_id, text, float(t_sec or 0), excerpt)
        return {"ok": True, "id": qid, "t_sec": float(t_sec or 0), "excerpt": excerpt}

    def current_position(self, session_id: str) -> float:
        """没在录音时用最后一个事件/最后一段转写当"现在"——队列要带时间感。"""
        rows = self.db.query(
            "SELECT MAX(t_start) t FROM segments WHERE session_id=?", (session_id,))
        cand = float(rows[0]["t"] or 0) if rows else 0.0
        try:
            k = self.db.query("SELECT MAX(t_end) t FROM knots WHERE session_id=?", (session_id,))
            cand = max(cand, float(k[0]["t"] or 0) if k else 0.0)
        except Exception:  # noqa: BLE001
            pass
        return cand

    def answer_question(self, qid: int) -> dict:
        """课后集中答：用这一节的回看点上下文 + 知识库把问题答掉。"""
        rows = self.db.query("SELECT * FROM pending_questions WHERE id=?", (int(qid),))
        if not rows:
            return {"ok": False, "error": "问题不存在"}
        r = rows[0]
        session_id = str(r["session_id"])
        question = str(r["text"])
        ctx = []
        if r["excerpt"]:
            ctx.append(f"【课堂原文（第 {_hms(float(r['t_sec'] or 0))} 处）】\n{r['excerpt']}")
        try:
            knots = [k for k in self.db.get_knots(session_id)
                     if abs(float(k.t_start) - float(r["t_sec"] or 0)) <= 180]
            if knots:
                ctx.append("【附近的回看点】\n" + "\n".join(
                    f"- {_hms(k.t_start)} {k.excerpt[:80]}" for k in knots[:4]))
        except Exception:  # noqa: BLE001
            pass
        try:
            # 课堂上下文并进问题本身：这一节的回看点与原文就在手边，没必要让它再猜一遍
            full_q = question if not ctx else (question + "\n\n" + "\n\n".join(ctx))
            res = self.ask(full_q, session_id=session_id,
                           use_kb=True, use_context=True)
            answer = res.get("answer") or ""
            err = res.get("error") or ""
        except Exception as e:  # noqa: BLE001
            answer, err = "", f"{type(e).__name__}: {e}"
        if not answer:
            return {"ok": False, "error": err or "没答出来", "id": int(qid)}
        self.db.set_question_status(int(qid), "answered", answer)
        return {"ok": True, "id": int(qid), "answer": answer}

    def answer_all_questions(self, session_id: str, limit: int = 8) -> dict:
        rows = self.db.get_questions(session_id, "pending")
        done, failed = 0, []
        for r in rows[:limit]:
            res = self.answer_question(int(r["id"]))
            if res.get("ok"):
                done += 1
            else:
                failed.append({"id": int(r["id"]), "error": res.get("error") or ""})
        return {"ok": True, "answered": done, "failed": failed,
                "remaining": max(0, len(rows) - done)}

    # -- 覆盖度：不许把"抽到了"说成"全覆盖" ------------------------------
    def capture_coverage(self, session_id: str) -> dict:
        """这一节的画面/文字证据到底覆盖了多少。

        借 DeepTutor `services/rag/visual_coverage.py` 的原则：
        **产出的资产数 ≠ 覆盖率**。界面必须说清楚"有几个回看点没画面"，
        否则用户会以为那段本来就没内容。
        """
        knots = self.db.get_knots(session_id) if session_id else []
        frames: list = []
        try:
            frames = self.db.query(
                "SELECT t FROM screen_notes WHERE session_id=? ORDER BY t",
                (session_id,)) if session_id else []
        except Exception:  # noqa: BLE001
            frames = []
        marks = [float(r["t"] or 0) for r in frames]
        covered = []
        for k in knots:
            if any(abs(m - float(k.t_start)) <= 45 for m in marks):
                covered.append(int(k.id))
        return {"knots": len(knots), "frames": len(marks),
                "covered": len(covered),
                "missing": max(0, len(knots) - len(covered)),
                "missing_ids": [int(k.id) for k in knots if int(k.id) not in covered]}

    # -- 笔记导出 -------------------------------------------------------
    def export_notes(self, session_id: str, path: str | os.PathLike | None = None) -> dict:
        """把这一节的东西装订成一份 markdown：回看点 / 掌握度 / 重讲 / 小测 / 待答问题。"""
        from .cognitive import mastery as kpm

        row = self.db.get_session(session_id)
        if row is None:
            return {"ok": False, "error": "会话不存在"}
        diary = self.diagnosis(session_id)
        knots = self.db.get_knots(session_id)
        kp_rows = {int(r["id"]): r for r in self.db.get_knowledge_points(str(row["course_id"]))}
        cov = self.capture_coverage(session_id)
        stats = self.db.quiz_stats(session_id)
        questions = self.db.get_questions(session_id, None)

        L: list[str] = []
        title = str(row["title"] or "未命名")
        L.append(f"# {title}")
        L.append("")
        L.append(f"- 时长：{_hms(float(row['duration'] or 0))}")
        L.append(f"- 回看点：{len(knots)} 个；画面覆盖：{cov['covered']}/{cov['knots']}"
                 + ("" if not cov["missing"] else f"（{cov['missing']} 个回看点没有画面证据）"))
        if stats["total"]:
            L.append(f"- 小测：{stats['answered']}/{stats['total']} 已答，"
                     f"答对 {stats['correct']}")
        L += ["", "## 回看点与掌握度", ""]
        if not knots:
            L.append("（这一节没有回看点）")
        for k in knots:
            solved = self.knot_solved(int(k.id))
            L.append(f"### {_hms(k.t_start)}　{k.level}　{k.score:.2f}"
                     f"　{'✅ 已补上' if solved else '⚠️ 未解决'}")
            L.append("")
            if k.excerpt:
                L.append(f"> {k.excerpt.strip()}")
                L.append("")
            # 人话解释存在 knots.evidence（界面那张表「为什么标记这里」用的就是它）
            if k.evidence:
                L.append(f"为什么标这里：{k.evidence}")
                L.append("")
            pts = self.db.get_knot_points(int(k.id))
            for p in pts:
                name = str(p["name"])
                m = float(p["mastery"] or 0)
                st = kpm.status_of(m, int(p["attempts"] or 0),
                                   str(p["kp_type"] or "concept"))
                L.append(f"- 知识点 **{name}**：{m * 100:.0f}% "
                         f"（{kpm.STATUS_CN.get(st, st)}，证据 {int(p['attempts'] or 0)} 次）")
            if pts:
                L.append("")

        notes = self.db.query(
            "SELECT t, kind, text FROM screen_notes WHERE session_id=? ORDER BY t",
            (session_id,))
        if notes:
            L += ["## 屏幕笔记", ""]
            for n in notes:
                kind = str(n["kind"] or "slide")
                L.append(f"- `{_hms(float(n['t'] or 0))}` [{kind}] "
                         f"{str(n['text'] or '').replace(chr(10), ' ')}")
            L.append("")

        if questions:
            L += ["## 累积的问题", ""]
            for q in questions:
                tag = {"pending": "待答", "answered": "已回答", "dropped": "已丢弃"}.get(
                    str(q["status"]), str(q["status"]))
                L.append(f"- `{_hms(float(q['t_sec'] or 0))}` [{tag}] {q['text']}")
                if q["answer"]:
                    L.append("")
                    L.append(f"  > {str(q['answer']).strip().replace(chr(10), ' ')[:500]}")
            L.append("")

        if diary.get("root_causes"):
            L += ["## 结构性根因", ""]
            for rc in diary["root_causes"][:8]:
                L.append(f"- {rc['name']}（{rc.get('phrase', '')}）")
            L.append("")

        md = "\n".join(L).rstrip() + "\n"
        if path is None:
            safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title)[:40] or "session"
            # 用配置里的导出目录，不写相对的 "data/exports"：
            # 相对路径会跟着"从哪个目录启动"跑掉，测试也会污染仓库
            from .config import EXPORT_DIR as _EXPORT_DIR

            out_dir = Path(_EXPORT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{safe}-{time.strftime('%Y%m%d-%H%M%S')}.md"
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        return {"ok": True, "path": str(p), "chars": len(md), "knots": len(knots)}

    # -- 诊断（漏洞地图，方案 3.5）---------------------------------------
    def diagnosis(self, session_id: str) -> dict:
        row = self.db.get_session(session_id)
        if row is None:
            return {"error": "会话不存在"}
        knots = self.db.get_knots(session_id)
        course_id = row["course_id"]
        duration = float(row["duration"] or 0.0)

        buckets: dict[int, int] = {}
        for k in knots:
            b = int((k.t_start or 0) // 300)
            buckets[b] = buckets.get(b, 0) + 1
        distribution = [
            {"start": b * 300, "end": b * 300 + 300, "count": c,
             "label": _hms(b * 300)}
            for b, c in sorted(buckets.items())
        ]

        # 跨节/跨课程关联（用户裁决：同一门课不同节之间找重复回看点习惯）
        related: list[dict] = []
        rows = self.db.query(
            """SELECT k.id, k.session_id, k.t_start, k.score, s.title, s.idx, kp.name
               FROM knots k
               JOIN sessions s ON s.id = k.session_id
               JOIN knot_points kx ON kx.knot_id = k.id
               JOIN knowledge_points kp ON kp.id = kx.kp_id
               WHERE s.course_id = ? AND k.session_id != ?
               ORDER BY k.score DESC LIMIT 20""",
            (course_id, session_id),
        )
        mine: set[str] = set()
        for k in knots:
            for name in self._knot_terms(k.id):
                mine.add(name)
        for r in rows:
            if r["name"] in mine:
                related.append({
                    "name": r["name"], "session": r["title"],
                    "session_idx": r["idx"], "t": _hms(r["t_start"]),
                    "score": round(float(r["score"]), 3),
                })

        # 复习顺序：按知识点首次出现时间 + 共现强度排序（近似前置依赖）
        plan: list[dict] = []
        seen: set[str] = set()
        for k in sorted(knots, key=lambda x: (-x.score, x.t_start)):
            names = self._knot_terms(k.id)
            fresh = [n for n in names if n not in seen]
            if not fresh and names:
                continue
            if not names:
                # 没抽到可信知识点也要进复习顺序，否则用户会以为这个回看点不存在
                names = ["（未命名知识点）"]
            for n in fresh:
                seen.add(n)
            plan.append({
                "knot_id": k.id, "t": _hms(k.t_start), "level": k.level,
                "score": round(k.score, 3), "evidence": k.evidence,
                "terms": fresh or names, "status": k.status,
                "excerpt": k.excerpt,
            })

        root_causes = self._root_cause_terms(course_id)
        return {
            "session": {"id": session_id, "title": row["title"], "idx": row["idx"],
                        "duration": duration, "duration_text": _hms(duration)},
            "knots": [
                {"id": k.id, "t": _hms(k.t_start), "t_start": k.t_start,
                 "score": round(k.score, 3), "level": k.level, "grade": k.grade,
                 "quadrant": k.quadrant, "evidence": k.evidence,
                 "excerpt": k.excerpt, "status": k.status,
                 "terms": self._knot_terms(k.id)}
                for k in knots
            ],
            "distribution": distribution,
            "related": related,
            "plan": plan,
            "root_causes": root_causes,
            "n_knots": len(knots),
            "n_strong": sum(1 for k in knots if k.grade == "strong"),
        }

    def _knot_terms(self, knot_id: int) -> list[str]:
        rows = self.db.query(
            """SELECT kp.name FROM knot_points kx JOIN knowledge_points kp
               ON kp.id = kx.kp_id WHERE kx.knot_id=? ORDER BY kx.weight DESC""",
            (knot_id,),
        )
        return [r["name"] for r in rows]

    def _root_cause_terms(self, course_id: str, min_sessions: int = 2) -> list[dict]:
        """跨节反复卡住的知识点。**已经补上的不再算漏洞**——否则清单只增不减。"""
        from .cognitive import mastery as kpm

        rows = self.db.query(
            """SELECT kp.name AS name, COUNT(DISTINCT k.session_id) AS n_sess,
                      COUNT(*) AS n_knot, AVG(k.score) AS avg_score,
                      kp.mastery AS mastery, kp.attempts AS attempts,
                      kp.kp_type AS kp_type
               FROM knowledge_points kp
               JOIN knot_points kx ON kx.kp_id = kp.id
               JOIN knots k ON k.id = kx.knot_id
               WHERE kp.course_id = ?
               GROUP BY kp.name HAVING n_sess >= ? ORDER BY n_sess DESC, avg_score DESC""",
            (course_id, min_sessions),
        )
        out = []
        for r in rows:
            if kpm.is_mastered(float(r["mastery"] or 0), str(r["kp_type"] or "concept")):
                continue        # 已过门槛：从漏洞清单里退场
            out.append({"name": r["name"], "sessions": r["n_sess"], "knots": r["n_knot"],
                        "avg_score": round(float(r["avg_score"] or 0), 3),
                        "mastery": round(float(r["mastery"] or 0), 3)})
        return out

    # -- 校准 -----------------------------------------------------------
    def recalibrate(self) -> dict:
        """用反馈训练权重（方案 5.2.6）。无反馈时不改权重。"""
        rows = self.db.query(
            """SELECT f.action, k.score, k.evidence_json FROM feedback f
               JOIN knots k ON k.id = f.knot_id"""
        )
        c = self.settings.cognitive
        n = 0
        loss = 0.0
        for r in rows:
            y = {"confirm_useful": 1, "replay_opened": 1, "mastered": 0,
                 "not_helpful": 0, "dismissed": 0}.get(r["action"])
            if y is None:
                continue
            z = _safe_json(r["evidence_json"]).get("z", {})
            if not z:
                continue
            loss += calibrate_step(self.weights, z, y, 1.0, c)
            n += 1
        # 硬负样本：高分但从未被处置
        hard = self.db.query(
            """SELECT k.id, k.evidence_json FROM knots k LEFT JOIN feedback f
               ON f.knot_id = k.id WHERE f.id IS NULL AND k.score >= ?""",
            (c.grade_strong,),
        )
        for r in hard:
            z = _safe_json(r["evidence_json"]).get("z", {})
            if not z:
                continue
            loss += calibrate_step(self.weights, z, 0, c.hard_negative_weight, c)
            n += 1
        if n:
            self.db.set_kv("weights", json.loads(self.weights.to_json()))
            self.db.execute(
                """INSERT INTO calibrations(ts,n_samples,weights_json,loss_before,loss_after,note)
                   VALUES(?,?,?,?,?,?)""",
                (time.time(), n, self.weights.to_json(), None, loss, "auto"),
            )
        return {"samples": n, "weights": dict(self.weights.weights),
                "bias": self.weights.bias, "loss": round(loss, 4)}

    # -- 演示数据 -------------------------------------------------------
    def seed_demo(self, title: str = "演示课程：概率论与数理统计") -> str:
        """生成一份可离线跑通的演示会话（文本为合成内容，明确标注）。"""
        cid = self.db.ensure_course(title, subject="数学")
        sid = self.db.create_session(cid, "第 1 讲　随机变量与分布", source="demo")
        gen = DemoASR()
        res = gen.transcribe(None)
        self.db.update_session(sid, duration=max(s.t_end for s in res.segments),
                               state="transcribed")
        self.db.replace_segments(sid, res.as_dicts(), pass_no=1)
        # 注入几个典型行为事件
        dur = max(s.t_end for s in res.segments)
        for t, kind, pl in (
            (round(dur * 0.18, 1), "seek_back", {"distance_s": 42}),
            (round(dur * 0.33, 1), "replay", {"matched_t": dur * 0.28}),
            (round(dur * 0.52, 1), "manual_mark", {"note": "演示手动标记"}),
            (round(dur * 0.71, 1), "seek_back", {"distance_s": 12}),
            (round(dur * 0.86, 1), "slowdown", {}),
        ):
            self.db.add_event(sid, float(t), kind, pl, 1.0, "manual")
        self.analyze_session(sid)
        return sid


def _safe_json(s: Any) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s) if isinstance(s, str) else dict(s)
    except Exception:
        return {}


def _hms(sec: float) -> str:
    sec = max(0, int(float(sec or 0)))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"