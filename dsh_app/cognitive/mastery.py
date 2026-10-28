"""知识点掌握度：把"回看点"变成"会没会"。

借鉴 HKUDS/DeepTutor `learning/mastery.py` 的两个设计（详见 docs/参考-DeepTutor借鉴清单.md）：

1. **近因加权**：`(0.5, 0.7, 0.85, 0.95, 1.0)`，越近的尝试权重越高。
   早期答错、后来答对的，不会被历史拖死；反之最近一次答错会明显拉低。
2. **低置信上限**：只答对 1 次最高 0.5，2 次最高 0.8。
   防"蒙对一次就算会"——这是最小代价的防作弊。

为何不做完整 DSR/FSRS：那需要逐次复习的间隔数据，而我们此前根本没有复习排程。
现在补了一个**够用的一维简化**（见文件下半部分）：只记一个"稳定度" S（天），
用 FSRS 的遗忘曲线形状算当下留存，到 90% 目标留存就算到期。
不引入 difficulty/学两套参数——单用户单机，多一个参数只是多一个没人调的旋钮。
"""
from __future__ import annotations

import math
from typing import Iterable

# 最近 5 次尝试的权重（旧 → 新）
RECENCY_WEIGHTS: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0)

# 证据不足时的掌握度上限：1 次尝试最多 0.5，2 次最多 0.8，3 次起不封顶
CONFIDENCE_CAP: dict[int, float] = {1: 0.5, 2: 0.8}

# 门槛：到多少算"补上了"。0.9 对应 DeepTutor 的 QUANTITATIVE_GATE，
# 也就是"九成把握才继续"——比我们的回看点阈值（0.65）严格，因为这是放行判断。
DEFAULT_GATE = 0.9

# 知识点类型。概念类没有唯一答案，用"用户自己说本来就会/讲明白了"作为定性通过，
# 定性通过直接记 1.0（对应 DeepTutor 的 _QUALITATIVE_PASS_DISPLAY）。
KP_TYPES = ("concept", "memory", "procedure")
QUALITATIVE_TYPES = frozenset({"concept"})


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def compute_mastery(correctness: Iterable[float]) -> float:
    """由尝试历史（按时间顺序，1=对，0=错，中间值=部分对）算 0..1 掌握度。"""
    seq = [clamp01(c) for c in correctness]
    if not seq:
        return 0.0
    recent = seq[-len(RECENCY_WEIGHTS):]
    weights = RECENCY_WEIGHTS[-len(recent):]
    score = sum(w * c for c, w in zip(recent, weights)) / sum(weights)
    return min(score, CONFIDENCE_CAP.get(len(recent), 1.0))


def gate_threshold(kp_type: str = "concept") -> float:
    """该类型的放行门槛。目前所有类型同一把尺子，留出按类型分档的位置。"""
    return DEFAULT_GATE


def is_mastered(mastery: float, kp_type: str = "concept") -> bool:
    return float(mastery or 0.0) >= gate_threshold(kp_type) - 1e-9


def status_of(mastery: float, attempts: int, kp_type: str = "concept") -> str:
    """给界面用的三档状态。没有尝试 ≠ 不会，所以单独一档。"""
    if int(attempts or 0) <= 0:
        return "unknown"          # 没验证过
    if is_mastered(mastery, kp_type):
        return "mastered"         # 补上了
    if float(mastery or 0.0) >= 0.5:
        return "learning"         # 在补
    return "weak"                 # 还不会


STATUS_CN = {
    "unknown": "未验证",
    "weak": "还不会",
    "learning": "在补",
    "mastered": "已补上",
}

# 反馈动作 → 这次尝试算不算"对"
ACTION_CORRECT: dict[str, float] = {
    "confirm_useful": 1.0,      # 这个讲明白了
    "replay_opened": 0.0,       # 只是打开了重讲，不构成证据
    "mastered": 1.0,            # 我本来就会（定性通过）
}

# 定性通过的动作：直接把掌握度记为 1.0，不参与加权
QUALITATIVE_PASS_ACTIONS = frozenset({"mastered"})


# ---------------------------------------------------------------------------
# 遗忘曲线（P2：复习排程）
# ---------------------------------------------------------------------------
# 形状借自 FSRS：R(t, S) = (1 + FACTOR·t/S)^DECAY。
# 这组常数是 FSRS 默认值，好处是"稳定度 S 天时留存正好 90%"——
# 于是 S 可以直接当"下次复习间隔（天）"用，不用再调目标留存。
FSRS_FACTOR = 19.0 / 81.0
FSRS_DECAY = -0.5
TARGET_RETENTION = 0.9

# 稳定度的上下界：最短半天（当天再来一次），最长 180 天（半年后再说）
MIN_STABILITY_DAYS = 0.5
MAX_STABILITY_DAYS = 180.0
DEFAULT_STABILITY_DAYS = 1.0


def retrievability(elapsed_days: float, stability_days: float) -> float:
    """当下还记得的概率。``elapsed_days`` 是距上次证据的天数。"""
    s = max(MIN_STABILITY_DAYS, float(stability_days or 0.0))
    t = max(0.0, float(elapsed_days or 0.0))
    r = (1.0 + FSRS_FACTOR * t / s) ** FSRS_DECAY
    return clamp01(r)


def next_stability(stability_days: float, correct: bool,
                   elapsed_days: float = 0.0) -> float:
    """答对/答错后新的稳定度。

    答对：越容易想起来（留存越高）涨得越少——这正是 FSRS 的"难度自适应"直觉，
    不额外引入 difficulty 参数。答错：打回接近最小值，但不归零。
    """
    s = max(MIN_STABILITY_DAYS, float(stability_days or DEFAULT_STABILITY_DAYS))
    if correct:
        r = retrievability(elapsed_days, s)
        # 同一节里连着答对（elapsed≈0）时 r≈1，按公式涨 0 倍会让人以为"白答了"。
        # 连续成功回忆本身仍是学习证据，所以给一个 5% 的下限增长。
        gain = max(1.05, 1.0 + 0.9 * (1.0 - r))
        return min(MAX_STABILITY_DAYS, s * gain)
    return max(MIN_STABILITY_DAYS, s * 0.35)


def due_in_days(stability_days: float) -> float:
    """多久之后到期（留存掉到 TARGET_RETENTION 的那一天）。"""
    s = max(MIN_STABILITY_DAYS, float(stability_days or 0.0))
    # 由 (1 + F·t/S)^DECAY = 0.9 反解 t，F、DECAY 取默认值时 t ≈ S
    inner = TARGET_RETENTION ** (1.0 / FSRS_DECAY) - 1.0
    return max(MIN_STABILITY_DAYS, s * inner / FSRS_FACTOR)


def review_due(retrievability_now: float, due_days: float, elapsed_days: float) -> bool:
    """该不该现在复习：留存掉到目标以下，或者已经过了到期时间。"""
    return float(retrievability_now) < TARGET_RETENTION or float(elapsed_days) >= float(due_days)


__all__ = [
    "RECENCY_WEIGHTS", "CONFIDENCE_CAP", "DEFAULT_GATE", "KP_TYPES",
    "QUALITATIVE_TYPES", "STATUS_CN", "ACTION_CORRECT",
    "QUALITATIVE_PASS_ACTIONS", "clamp01", "compute_mastery",
    "gate_threshold", "is_mastered", "status_of",
    "FSRS_FACTOR", "FSRS_DECAY", "TARGET_RETENTION", "MIN_STABILITY_DAYS",
    "MAX_STABILITY_DAYS", "DEFAULT_STABILITY_DAYS", "retrievability",
    "next_stability", "due_in_days", "review_due",
]