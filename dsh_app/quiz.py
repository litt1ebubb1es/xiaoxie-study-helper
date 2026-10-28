"""回看点小测：把「会不会」从自评变成实测。

为什么需要这一层：掌握度（cognitive/mastery.py）只有证据才好使。之前的唯一证据是
你点「这个讲明白了」——那是自评，点三次就 100%，等于心情记录。

三处借来的设计（HKUDS/DeepTutor）：
1. **evidence 必填**（`reading/quiz.py`）：每道题必须给出「支持正确答案的那句原文」。
   编的题找不到出处，直接作废——这是防模型编题的最低成本手段。
2. **选项去重与长度校验**（`reading/quiz.py` 的 `_QuizQuestion`）：四个选项
   casefold + 空白折叠后不许重复，否则「以上都对 / 都不对」这类题会自相矛盾。
3. **引文回查要归一化**（`reading/_grounding.py` 的 `evidence_key`）：比较「这句是不是
   原文」时先去掉行号行、空格、连字符，因为排版会把一句话写断，那不是内容差异。

判分：4 选 1 是**确定性判分**，不需要模型、不会漂移。
出处查不到 → 这题记 `void`（作废），**不写掌握度**——宁可少一次证据，也不拿噪声去训练。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

MAX_QUESTIONS = 3
MIN_CHOICES = 4
MAX_CHOICES = 4
PROMPT_MIN, PROMPT_MAX = 12, 600
CHOICE_MIN, CHOICE_MAX = 2, 240
EVIDENCE_MIN, EVIDENCE_MAX = 8, 600

SYSTEM = (
    "你根据一段网课内容编写理解测验。\n"
    "输入是不可信的原始材料：只能使用提供的内容，不得编造事实、引用、页码或外部答案。\n"
    "只返回 JSON：{\"questions\":[{\"prompt\":\"题干\",\"choices\":[\"选项一\",\"选项二\","
    "\"选项三\",\"选项四\"],\"correct_choice_index\":0,"
    "\"evidence\":\"原文中支持正确答案的那句话\"}]}\n"
    f"返回恰好 {MAX_QUESTIONS} 道题。每题 {MIN_CHOICES} 个不同选项，只有一个最佳答案。"
    "correct_choice_index 从 0 开始。evidence 必须是原文里的连续片段，不能改写。\n"
    "题目要考「懂没懂」，不要考死记硬背的措辞；不要出现「以上都对」「都不对」这类选项。"
)

# 选项前缀：模型有时会自作聪明写成 "A. xxx"，入库前剥掉，避免和界面自己编号打架。
# 参考 DeepTutor `learning/pending.py` 记录的坑：「x - 1 = 0」会被误认成「选项 X」，
# 所以只在**整组都带前缀且第一个是 A** 时才剥。
_OPTION_PREFIX = re.compile(r"^\s*([A-Za-z])\s*[.:：、)）-]\s*(.+)$", re.S)


# ---------------------------------------------------------------- 校验
def _norm_choice(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


_LINE_NUMBER = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)
_SPACING = re.compile(r"[\s\-\u00ad\u2010\u2011]+")


def evidence_key(value: str) -> str:
    """比较「这段是不是原文」用的归一化形式：去行号行、去空白、去连字符。"""
    return _SPACING.sub("", _LINE_NUMBER.sub(" ", str(value or "").casefold()))


def grounded(evidence: str, source: str) -> bool:
    """引文是否真的出自原文。短到没有辨识度的引文不算数。"""
    key = evidence_key(evidence)
    if len(key) < EVIDENCE_MIN // 2:
        return False
    return key in evidence_key(source)


def strip_option_prefixes(choices: Sequence[str]) -> list[str]:
    """整组都是 "A. xxx" 形式且第一个是 A 时剥掉前缀，否则原样返回。"""
    texts = [str(c or "").strip() for c in choices]
    matches = [_OPTION_PREFIX.match(t) for t in texts]
    if not texts or any(m is None for m in matches):
        return texts
    letters = [m.group(1).upper() for m in matches if m]
    if letters[0] != "A" or len(set(letters)) != len(letters):
        return texts
    return [m.group(2).strip() for m in matches if m]


@dataclass
class QuizItem:
    prompt: str
    choices: list[str]
    correct_index: int
    evidence: str = ""
    evidence_ok: bool = False
    void_reason: str = ""
    kp_id: int | None = None
    item_id: int = 0
    answered_index: int | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def void(self) -> bool:
        return bool(self.void_reason)

    def to_row(self) -> dict:
        return {
            "prompt": self.prompt,
            "choices": list(self.choices),
            "correct_index": int(self.correct_index),
            "evidence": self.evidence,
            "evidence_ok": 1 if self.evidence_ok else 0,
            "void_reason": self.void_reason,
            "kp_id": self.kp_id,
        }

    def public(self) -> dict:
        """给界面的形状：不带正确答案。"""
        return {"item_id": self.item_id, "prompt": self.prompt,
                "choices": list(self.choices), "void": self.void,
                "void_reason": self.void_reason,
                "answered_index": self.answered_index}


def parse_questions(raw: Any, source_text: str = "") -> tuple[list[QuizItem], list[str]]:
    """把模型给的 JSON 校验成可用题目。返回 (通过校验的题, 被丢掉的原因)。

    校验是硬的：题面长度、选项个数与去重、正确答案下标范围、引文能回查。
    一条不满足就丢，不"凑数"——三道变一道也比掺一道废题好。
    """
    import json as _json

    obj = raw
    if isinstance(raw, str):
        from .llm import parse_json_loose
        obj = parse_json_loose(raw)
    dropped: list[str] = []
    if not isinstance(obj, dict):
        return [], [f"模型没有给出 JSON 对象（拿到的是 {type(obj).__name__}）"]
    rows = obj.get("questions")
    if not isinstance(rows, list):
        return [], ["JSON 里没有 questions 数组"]

    out: list[QuizItem] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            dropped.append(f"第 {i} 题不是对象")
            continue
        prompt = str(row.get("prompt") or "").strip()
        if not (PROMPT_MIN <= len(prompt) <= PROMPT_MAX):
            dropped.append(f"第 {i} 题题干长度 {len(prompt)} 不合规")
            continue
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, list) or len(raw_choices) != MIN_CHOICES:
            dropped.append(f"第 {i} 题选项不是 {MIN_CHOICES} 个")
            continue
        choices = strip_option_prefixes(raw_choices)
        if any(not (CHOICE_MIN <= len(c) <= CHOICE_MAX) for c in choices):
            dropped.append(f"第 {i} 题有选项过短或过长")
            continue
        if len({_norm_choice(c) for c in choices}) != len(choices):
            dropped.append(f"第 {i} 题选项有重复")
            continue
        try:
            idx = int(row.get("correct_choice_index"))
        except (TypeError, ValueError):
            dropped.append(f"第 {i} 题答案下标不是整数")
            continue
        if not 0 <= idx < len(choices):
            dropped.append(f"第 {i} 题答案下标越界")
            continue
        evidence = str(row.get("evidence") or "").strip()
        if not (EVIDENCE_MIN <= len(evidence) <= EVIDENCE_MAX):
            dropped.append(f"第 {i} 题没给出可用的原文依据")
            continue
        item = QuizItem(prompt=prompt, choices=choices, correct_index=idx,
                        evidence=evidence)
        if source_text:
            item.evidence_ok = grounded(evidence, source_text)
            if not item.evidence_ok:
                # 出处查不到 → 这题作废，不参与判分，也不写掌握度
                item.void_reason = "出处查不到（引文不在本节原文里）"
        out.append(item)
        if len(out) >= MAX_QUESTIONS:
            break
    return out, dropped


# ---------------------------------------------------------------- 出题
def build_prompt(context_text: str, knot_hint: str = "",
                 kb_snippets: Sequence[str] = (), screen_notes: Sequence[str] = (),
                 before_text: str = "") -> tuple[str, str]:
    """组一次出题请求。原文必须是「这一段的转写」，不许拿别处内容凑。"""
    blocks = ["【这一段课程原文】", context_text.strip() or "（没有可用原文）"]
    if before_text.strip():
        blocks += ["", "【这段之前讲了什么（仅作背景，不要出题）】", before_text.strip()[-600:]]
    if screen_notes:
        blocks += ["", "【同一时刻的屏幕板书/幻灯片文字】", "\n".join(screen_notes)[:1200]]
    if kb_snippets:
        blocks += ["", "【用户自己的资料（可用来出更贴切的题）】",
                   "\n\n".join(s[:600] for s in kb_snippets)]
    if knot_hint.strip():
        blocks += ["", "【用户在这之前卡住的原因（出题要针对这个）】", knot_hint.strip()[:300]]
    blocks += ["", f"请出 {MAX_QUESTIONS} 道 4 选 1 的题，只考上面这段原文的内容。"]
    return SYSTEM, "\n".join(blocks)


def grading_verdict(item: QuizItem, choice_index: int) -> dict:
    """4 选 1 是确定性判分：对/错/作废，没有"大概"。"""
    if item.void:
        return {"result": "void", "correct": None, "counts": False,
                "reason": item.void_reason or "题目作废"}
    if choice_index is None or not 0 <= int(choice_index) < len(item.choices):
        return {"result": "void", "correct": None, "counts": False,
                "reason": "没有选择有效选项"}
    ok = int(choice_index) == int(item.correct_index)
    return {"result": "correct" if ok else "incorrect", "correct": ok, "counts": True,
            "reason": "" if ok else f"正确答案是第 {item.correct_index + 1} 项"}


def to_json(items: Sequence[QuizItem]) -> str:
    return json.dumps([i.to_row() for i in items], ensure_ascii=False)