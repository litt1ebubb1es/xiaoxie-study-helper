"""自动抓屏：由算法决定什么时候抓，而不是用户按快捷键。

时机的判定只依赖三个可解释的量，不做玄学：
1. 画面变化率（缩小到 64x36 灰度后逐像素差的比例）——画面没变就不抓，这一步就滤掉 90%+；
2. 冷却时间——同一个画面停留期间不重复抓，避免一节课抓几百帧；
3. 配额——每小时上限，防止长时间网课把 token 烧光。

抓完之后还有一道"内容闸门"：视觉模型说画面里没有课程文字（空文本）就丢弃，不入库。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoCapConfig:
    enabled: bool = False
    interval_s: float = 8.0        # 检查间隔
    change_threshold: float = 0.12  # 画面变化率阈值
    min_gap_s: float = 30.0        # 两次抓取的最小间隔
    max_per_hour: int = 40         # 每小时上限


@dataclass
class Decision:
    take: bool
    reason: str


def change_ratio(prev: bytes, cur: bytes) -> float:
    """两个同尺寸灰度帧的差异比例（0..1）。亮度差超过 24 才算"变了"。"""
    if not prev or not cur or len(prev) != len(cur):
        return 1.0
    diff = 0
    for a, b in zip(prev, cur):
        if abs(a - b) > 24:
            diff += 1
    return diff / float(len(prev))


def decide(ratio: float, since_last_s: float, taken_last_hour: int,
           cfg: AutoCapConfig) -> Decision:
    if not cfg.enabled:
        return Decision(False, "未启用")
    if taken_last_hour >= cfg.max_per_hour:
        return Decision(False, "达到每小时上限")
    if since_last_s < cfg.min_gap_s:
        return Decision(False, "距上次太近")
    if ratio < cfg.change_threshold:
        return Decision(False, "画面没变")
    return Decision(True, f"画面变化 {ratio:.0%}")
