"""认知层 B：行为信号、融合评分、四象限、回看点合并、权重校准。

对应方案 5.2.2 – 5.2.7。四处对方案的裁决：

1. **B 的时间衰减形式**（原文只说"取窗口内最大强度并做时间衰减"）：
   `B = max_i wᵢ·exp(-Δt²/(2σ²))`，σ = 窗口长度 × 1/3，Δt 为事件到窗口中心的距离。
   先算单事件衰减再取最大值——如果先 max 再衰减，窗口边缘的事件会被整体压低，
   与"越靠近窗口中心权重越高"的意图相反。

2. **象限阈值**（原文 L>0.5 / B>0.5）：改用课程内分位数（默认 P70）。
   理由是方案自己在 5.2.1 强调"以本课程自身的分布为基准"；绝对 0.5 会把约一半窗口
   判成高负荷，与"抑制误报"（5.2.3）的目标冲突。

3. **回看点起止时间与合并**（原文未定义）：重叠窗口按 S 取峰值，回看点区间 = 首末标记窗口
   中心 ± 25 秒，相邻标记窗口间隙 ≤ 30 秒视为同一回看点。

4. **校准的负样本**（原文只提到用户反馈作为 y）：补一条硬负样本策略——
   S ≥ 0.65 但 7 天内既未被点开也未被处置的回看点，按 y=0、样本权重 0.3 入训。
   否则绝大多数用户从不反馈，样本永远不足 20 条，权重校准形同虚设。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .features import (
    FEATURE_CN,
    FEATURE_KEYS,
    Transcript,
    Window,
    default_weights,
    explain_window,
    extract_features,
    load_scores,
    make_windows,
    znorm_windows,
    TermLexicon,
)

# 事件强度权重（方案 5.2.2 表格）
EVENT_KIND_CN = {
    "seek_back": "回退跳转",
    "replay": "重复播放",
    "overstay": "停留超时",
    "slowdown": "降速",
    "blur_pause": "失焦暂停",
    "external_search": "外部检索",
    "manual_mark": "手动标记",
    "pause": "播放暂停",
}


# 事件优先级：强度相同时谁说话算数。方案 3.2 要求"手动标记的权重高于任何自动信号"，
# 而 5.2.2 的表格里 seek_back 也能到 1.0 —— 两者在极值处并列。
# 处理办法：强度保持表格原值（0.6–1.0），并列时按优先级取手动标记。
EVENT_PRIORITY = {
    "manual_mark": 100,
    "seek_back": 60,
    "replay": 55,
    "overstay": 40,
    "external_search": 35,
    "slowdown": 30,
    "blur_pause": 20,
    "pause": 45,
}


def event_strength(kind: str, payload: dict | None = None, confidence: float = 1.0) -> float:
    """单事件强度。方案 5.2.2 的权重表 + 手动标记（3.2：权重高于任何自动信号）。"""
    p = payload or {}
    if kind == "seek_back":
        raw = p.get("distance_s", 8.0)
        dist = 8.0 if raw is None else float(raw)
        # 小幅度回退=确认细节(0.6)，大幅度回退>30s=整段没跟上(1.0)
        w = 0.6 + 0.4 * min(1.0, max(0.0, dist) / 30.0)
    elif kind == "replay":
        w = 0.8
    elif kind == "overstay":
        w = 0.7
    elif kind == "slowdown":
        w = 0.5
    elif kind == "blur_pause":
        w = 0.3
    elif kind == "external_search":
        w = 0.6
    elif kind == "pause":
        # 暂停是"我自己按的"，比失焦可信，但没有回退那么强的意图
        w = 0.45 + 0.25 * min(1.0, float(p.get("pause_s", 3.0) or 3.0) / 60.0)
    elif kind == "manual_mark":
        w = 1.0
    else:
        w = 0.5
    return float(max(0.0, min(1.0, w * confidence)))


@dataclass
class Evt:
    t: float
    kind: str
    payload: dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "manual"

    @property
    def strength(self) -> float:
        return event_strength(self.kind, self.payload, self.confidence)


def behavior_scores(wins: Sequence[Window], events: Sequence[Evt],
                    window_s: float = 60.0, sigma_ratio: float = 1.0 / 3.0) -> None:
    """就地写入每个窗口的 B。"""
    sigma = max(1e-6, window_s * sigma_ratio)
    denom = 2.0 * sigma * sigma
    for w in wins:
        best = 0.0
        best_evt: Evt | None = None
        best_pri = -1
        for e in events:
            # 只考虑与窗口有时间交叠或近邻的事件（±2σ）
            if e.t < w.t_start - 2 * sigma or e.t > w.t_end + 2 * sigma:
                continue
            dt = e.t - w.center
            val = e.strength * math.exp(-(dt * dt) / denom)
            pri = EVENT_PRIORITY.get(e.kind, 0)
            if val > best + 1e-9 or (abs(val - best) <= 1e-9 and pri > best_pri):
                best, best_evt, best_pri = val, e, pri
        w.b_score = float(min(1.0, best))
        setattr(w, "_best_event", best_evt)


def quadrant_of(l: float, b: float, l_thr: float, b_thr: float) -> str:
    if l >= l_thr and b >= b_thr:
        return "explicit"
    if l >= l_thr and b < b_thr:
        return "implicit_candidate"
    if l < l_thr and b >= b_thr:
        return "personal"
    return "ignore"


QUADRANT_CN = {
    "explicit": "显性回看点",
    "implicit_candidate": "隐性回看点候选",
    "personal": "个人因素",
    "ignore": "忽略",
}


def grade_of(s: float, strong: float = 0.65, medium: float = 0.45) -> str:
    if s >= strong:
        return "strong"
    if s >= medium:
        return "medium"
    return "none"


@dataclass
class Knot:
    t_start: float
    t_end: float
    level: str
    quadrant: str
    grade: str
    score: float
    l_score: float
    b_score: float
    evidence: str = ""
    evidence_detail: dict = field(default_factory=dict)
    excerpt: str = ""
    win_idx: int = -1


def _pct(vals: Sequence[float], p: float) -> float:
    arr = np.asarray([v for v in vals], dtype=float)
    if arr.size == 0:
        return 0.5
    return float(np.percentile(arr, p))


def fuse_and_merge(
    wins: Sequence[Window],
    cfg,
    duration: float,
    include_personal: bool = False,
) -> list[Knot]:
    """融合评分 + 四象限 + 合并成回看点。"""
    if not wins:
        return []
    valid = [w for w in wins if w.valid] or list(wins)
    for w in wins:
        w.s_score = cfg.w_load * w.l_score + cfg.w_behavior * w.b_score
        w.grade = grade_of(w.s_score, cfg.grade_strong, cfg.grade_medium)

    l_thr = _pct([w.l_score for w in valid], cfg.quadrant_percentile)
    b_thr = _pct([w.b_score for w in valid], cfg.quadrant_percentile)
    for w in wins:
        w.quadrant = quadrant_of(w.l_score, w.b_score, l_thr, b_thr)

    flagged = [w for w in wins if w.s_score >= cfg.grade_medium]
    flagged = [
        w for w in flagged
        if w.quadrant in ("explicit", "implicit_candidate") or (include_personal and w.quadrant == "personal")
    ]
    flagged.sort(key=lambda w: w.t_start)

    groups: list[list[Window]] = []
    for w in flagged:
        if groups and w.t_start - groups[-1][-1].t_start <= cfg.merge_gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])

    knots: list[Knot] = []
    for g in groups:
        peak = max(g, key=lambda x: x.s_score)
        first, last = g[0], g[-1]
        t0 = max(0.0, min(first.center - cfg.knot_half_width_s, first.t_start))
        t1 = min(duration, max(last.center + cfg.knot_half_width_s, last.t_end))
        if t1 - t0 < 5.0:
            t1 = t0 + 5.0
        level = "explicit" if peak.quadrant == "explicit" else "implicit"
        ev_load = explain_window(peak, top_k=2)
        ev_beh = _behavior_evidence(getattr(peak, "_best_event", None))
        detail = {
            "features": {k: round(peak.feats.get(k, 0.0), 3) for k in FEATURE_KEYS},
            "z": {k: round(peak.z.get(k, 0.0), 3) for k in FEATURE_KEYS},
            "windows": [round(w.t_start, 1) for w in g],
            "l_threshold": round(l_thr, 3),
            "b_threshold": round(b_thr, 3),
        }
        evidence_parts = [*ev_load, *ev_beh]
        if not evidence_parts:
            evidence_parts = ["行为与内容负荷同时偏高"]
        knots.append(Knot(
            t_start=t0, t_end=t1, level=level, quadrant=peak.quadrant, grade=peak.grade,
            score=round(peak.s_score, 4), l_score=round(peak.l_score, 4),
            b_score=round(peak.b_score, 4), evidence="；".join(evidence_parts),
            evidence_detail=detail, excerpt=_excerpt(peak.text), win_idx=peak.idx,
        ))
    _resolve_overlaps(knots, duration)
    return knots


def _resolve_overlaps(knots: list["Knot"], duration: float, min_width: float = 5.0) -> None:
    """保证相邻回看点区间不重叠。

    为什么必须做：回看点区间是「首末标记窗口中心 ± 半宽」，半宽 25s 大于相邻峰值的最小间距，
    两个独立回看点就会在时间轴上粘成一片——用户既看不出是两处，双击时也无法判断落在哪个回看点。
    这里在相邻区间的重叠段取中点切开，并保证切开后各自不小于 min_width。
    """
    knots.sort(key=lambda k: k.t_start)
    for i in range(len(knots) - 1):
        a, b = knots[i], knots[i + 1]
        if a.t_end <= b.t_start:
            continue
        mid = (a.t_end + b.t_start) / 2.0
        # 若包含关系过强，退化为按两段宽度比例分配
        if mid - min_width < a.t_start or b.t_end - mid < min_width:
            total = max(min_width * 2, b.t_end - a.t_start)
            mid = a.t_start + total * (a.t_end - a.t_start) / max(
                1e-6, (a.t_end - a.t_start) + (b.t_end - b.t_start))
        a.t_end = max(a.t_start + min_width, min(mid, duration))
        b.t_start = min(b.t_end - min_width, max(a.t_end, mid))
        if b.t_start < a.t_end:
            b.t_start = a.t_end


def _behavior_evidence(e: Evt | None) -> list[str]:
    if e is None:
        return []
    name = EVENT_KIND_CN.get(e.kind, e.kind)
    if e.kind == "seek_back":
        d = float(e.payload.get("distance_s", 0) or 0)
        return [f"你回退了约 {d:.0f} 秒" if d else "你回退到前面重看"]
    if e.kind == "replay":
        return ["同一段内容你播放了不止一次"]
    if e.kind == "overstay":
        return ["这一段你停留时间明显超过内容长度"]
    if e.kind == "slowdown":
        return ["你把播放速度降下来了"]
    if e.kind == "blur_pause":
        return ["窗口失焦并长时间暂停（可能是被消息打断）"]
    if e.kind == "external_search":
        return ["你切去搜索了相关词"]
    if e.kind == "manual_mark":
        return ["你手动标记了这里"]
    if e.kind == "pause":
        return [f"你暂停了约 {float(e.payload.get('pause_s', 0) or 0):.0f} 秒"]
    if e.kind == "overstay" and e.payload.get("pause_s"):
        return [f"这一段你停了约 {float(e.payload['pause_s']):.0f} 秒（可能去查东西了）"]
    return [name]


def _excerpt(text: str, maxlen: int = 60) -> str:
    t = " ".join((text or "").split())
    return t[:maxlen] + ("…" if len(t) > maxlen else "")


# --------------------------------------------------------------------------
# 一次性分析入口
# --------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    windows: list[Window]
    knots: list[Knot]
    stats: dict[str, Any] = field(default_factory=dict)


def analyze(
    segments: Sequence[dict],
    events: Sequence[Evt],
    cfg,
    subject: str = "通用",
    include_personal: bool = False,
    lexicon: TermLexicon | None = None,
    content_kind: str = "speech",
) -> AnalysisResult:
    tr = Transcript(segments)
    if not tr.segments:
        return AnalysisResult(windows=[], knots=[], stats={"reason": "no_transcript"})
    lex = lexicon or TermLexicon.build(tr.text, subject=subject)
    wins = make_windows(tr, cfg.window_s, cfg.step_s)
    for w in wins:
        w.content_kind = content_kind   # 兜底；extract_features 会按窗口覆盖再细分
    extract_features(wins, tr, lex, tr.text)
    stats = znorm_windows(wins)
    load_scores(wins, default_weights(cfg), tau=cfg.tau, bias=cfg.bias)
    behavior_scores(wins, events, cfg.window_s, cfg.decay_sigma_ratio)
    knots = fuse_and_merge(wins, cfg, tr.duration, include_personal=include_personal)
    return AnalysisResult(
        windows=wins,
        knots=knots,
        stats={
            "duration": tr.duration,
            "n_windows": len(wins),
            "n_valid": sum(1 for w in wins if w.valid),
            "n_knots": len(knots),
            "n_events": len(events),
            "n_terms": len(lex),
            "znorm": {k: [round(v[0], 3), round(v[1], 3)] for k, v in stats.items()},
            "mean_cps": round(
                sum(len(s["text"]) for s in segments)
                / max(1e-6, sum(float(s["t_end"]) - float(s["t_start"]) for s in segments)), 3),
        },
    )


# --------------------------------------------------------------------------
# 权重校准（方案 5.2.6）
# --------------------------------------------------------------------------
@dataclass
class WeightState:
    weights: dict[str, float]
    bias: float = 0.0
    n_samples: int = 0

    @classmethod
    def initial(cls, cfg) -> "WeightState":
        return cls(weights=default_weights(cfg), bias=cfg.bias, n_samples=0)

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "bias": self.bias, "n": self.n_samples},
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str | None, cfg) -> "WeightState":
        if not s:
            return cls.initial(cfg)
        try:
            d = json.loads(s)
            return cls(weights={k: float(v) for k, v in d["weights"].items()},
                       bias=float(d.get("bias", 0.0)), n_samples=int(d.get("n", 0)))
        except Exception:
            return cls.initial(cfg)


def calibrate_step(state: WeightState, z: dict[str, float], y: int,
                   sample_weight: float, cfg) -> float:
    """w ← w + η·s·(y − ŷ)·x（方案 5.2.6 的 SGD，加了样本权重 s 与 η 调度）。"""
    pred = 0.0
    raw = sum(state.weights.get(k, 0.0) * z.get(k, 0.0) for k in FEATURE_KEYS) + state.bias
    pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw / max(1e-6, cfg.tau)))))
    eta = cfg.eta
    if state.n_samples < cfg.eta_half_below:
        eta *= 0.5
    elif state.n_samples > cfg.eta_decay_above:
        eta = cfg.eta * (cfg.eta_decay_above / float(state.n_samples))
    err = (y - pred) * sample_weight * eta
    for k in FEATURE_KEYS:
        state.weights[k] = state.weights.get(k, 0.0) + err * z.get(k, 0.0)
    state.bias += err
    state.n_samples += 1
    return float((y - pred) ** 2)