"""生成层：五种降维重讲策略（方案 3.3 / 创新点第三）。

关键约束（方案 3.3 末段）：重讲必须带三个上下文——
这一段之前讲了什么、这一段之后要讲什么、用户此前在哪些相关概念上卡过。
任何一条缺失，重讲就退化成通用解释。

每套策略都有 LLM 路径与纯模板降级路径：没有可用模型时仍然给出结构化的
"该怎么换一条路讲"，而不是报错——这也是方案 4.6 降级策略的要求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class Strategy:
    key: str
    name: str
    gist: str            # 一句话做法
    fit: str             # 适用场景
    ask: str             # 给模型的指令
    template: list[str]  # 无模型时的结构化提纲


STRATEGIES: list[Strategy] = [
    Strategy(
        key="analogy",
        name="换类比",
        gist="用一个生活化场景替代原讲授路径",
        fit="抽象概念、数学定义",
        ask=("不要复述老师的原话。请构造一个日常生活中的类比，让这个概念的"
             "结构在类比里一一对应地出现；说明类比中哪一部分对应原概念的定义条件，"
             "并指出类比的边界在哪里（哪里不成立）。"),
        template=[
            "先给一个日常场景（吃、行、游戏、排队、记账任选其一）",
            "把场景里的每一个环节对应到原概念的定义条件",
            "指出类比在哪一步失效，避免过度外推",
        ],
    ),
    Strategy(
        key="prerequisite",
        name="补前置",
        gist="指出缺失的前置知识并先讲它",
        fit="卡在推导中间步骤",
        ask=("先判断这一段依赖了哪些前置知识，按依赖顺序只挑出最可能缺失的那一条，"
             "用三句话讲清这个前置知识本身，再回到原段落重走一遍那一步推导。"),
        template=[
            "列出这一段依赖的前置知识（一般不超过三条）",
            "只挑最可能缺失的一条，用三句话讲明白",
            "带着这个前置回到原步骤，重走一遍",
        ],
    ),
    Strategy(
        key="numeric",
        name="数值化",
        gist="用具体数字代入跑一遍",
        fit="公式、算法、财务模型",
        ask=("把抽象关系用具体数字代进去，一步步算出中间结果和最终结果，"
             "每一步都写出代入了什么数、为什么是这个数。不要省略任何一步。"),
        template=[
            "给每个变量取一个具体且好算的数（如 0.1、2、10）",
            "逐步代入，写出每一步的中间结果",
            "最后回看：结果的量级是否合理，说明了什么",
        ],
    ),
    Strategy(
        key="reverse",
        name="反向验证",
        gist="从结论倒推，解释为什么会得到它",
        fit="定理证明、架构决策",
        ask=("从结论出发倒推：先假设结论成立，说明它能推出哪些可检验的现象，"
             "再指出原证明/原设计里哪一步是唯一不可替代的关键一步。"),
        template=[
            "先写下结论，问：如果它成立，世界会有什么不同",
            "倒推出唯一不可替代的关键一步",
            "说明为什么其他步骤都可以替换，只有它在撑住结论",
        ],
    ),
    Strategy(
        key="layered",
        name="分层复述",
        gist="先给一句话版本，再逐层加细节",
        fit="信息密度过高的段落",
        ask=("先给一句话版本（不超过 30 字），再给一层展开（一句话变三句），"
             "最后才给完整细节。每一层都要能独立成立，不要依赖下一层才看懂。"),
        template=[
            "第一层：一句话版本（≤30 字）",
            "第二层：一句话变三句，补齐条件",
            "第三层：完整细节与例外情况",
        ],
    ),
]

STRATEGY_MAP = {s.key: s for s in STRATEGIES}


def recommend(content_hint: str, evidence: dict | None = None) -> str:
    """按内容与回看点证据推荐策略。规则简单、可解释、离线可跑。"""
    h = content_hint or ""
    z = (evidence or {}).get("z", {}) if evidence else {}
    if any(k in h for k in ("定义", "概念", "术语", "是什么")):
        return "analogy"
    if any(k in h for k in ("证明", "定理", "为什么", "推导", "因此", "所以")):
        return "reverse"
    if any(k in h for k in ("公式", "算法", "计算", "代入", "数值", "收益", "成本")):
        return "numeric"
    if float(z.get("SL", 0) or 0) > 0.8 or float(z.get("TD", 0) or 0) > 0.8:
        return "layered"
    if float(z.get("SD", 0) or 0) > 0.5:
        return "numeric"
    return "prerequisite"


def build_prompt(
    strategy: Strategy,
    target_text: str,
    before_text: str,
    after_text: str,
    history_terms: Sequence[str],
    course_title: str = "",
) -> tuple[str, str]:
    system = (
        "你是一位擅长把难内容讲明白的助教。你的任务不是复述，而是换一条路径重讲。"
        "遵守：① 只讲这一段，不要扩展到整门课；② 不得替用户写作业或给出可提交的答案；"
        "③ 不确定的地方明确说不知道；④ 用中文，口语化，控制在 400 字内。"
    )
    ctx = [
        f"课程：{course_title or '未命名课程'}",
        f"重讲策略：{strategy.name} —— {strategy.gist}",
        f"策略要求：{strategy.ask}",
        "",
        "【用户没听懂的这一段】",
        target_text.strip() or "（无转写文本，按标题推断）",
        "",
        "【这一段之前讲了什么】",
        before_text.strip() or "（课程开头）",
        "",
        "【这一段之后要讲什么】",
        after_text.strip() or "（课程结尾）",
    ]
    if history_terms:
        ctx += ["", "【用户此前在这些相关概念上卡过】", "、".join(history_terms[:10])]
    return system, "\n".join(ctx)


def fallback_outline(strategy: Strategy, target_text: str,
                     history_terms: Sequence[str]) -> str:
    """无模型时的降级输出：给结构，不给假答案。"""
    lines = [f"【{strategy.name}】{strategy.gist}", "", "建议按这个顺序重讲这一段："]
    for i, step in enumerate(strategy.template, 1):
        lines.append(f"{i}. {step}")
    if history_terms:
        lines += ["", "要带上的旧回看点：" + "、".join(history_terms[:6])]
    snippet = " ".join((target_text or "").split())[:120]
    if snippet:
        lines += ["", f"原文片段：{snippet}…"]
    lines += ["", "（当前未接入可用的生成模型，以上是结构化提纲；配置云端 API 或本地主脑后"
                  "会产出完整的重讲内容。）"]
    return "\n".join(lines)