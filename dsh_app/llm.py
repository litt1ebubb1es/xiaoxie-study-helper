"""接入层：统一 OpenAI 兼容客户端（方案 4.6 统一接入层）。

四条设计要点逐条落地：
- 协议统一：本地 vLLM / llama.cpp / FunASR-server 与云端 API 都走同一个 OpenAI 客户端。
- 本地与云端等价：上层只认 `chat()`，不认 base_url。
- 能力探测：`probe()` 报告可达性、模型清单、是否疑似支持视觉。
- 降级策略：调用失败时返回降级标记而不是抛到界面上炸掉；界面上明确告知。
"""

from __future__ import annotations

import json
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse


# ---------------------------------------------------------------- 结构化调用
# 给"模型该给一整块结果却给了半块"这件事定一条规则：再问一次，但把预算让给答案。
# 对应 DeepTutor `services/llm/structured_retry.py` 的思路（诊断相同、实现按我们场景写）。
HARDEN_HINT = ("重要：直接输出最终结果本身。不要任何推理过程、分析、前言、致辞或解释说明；"
               "如果要求 JSON，就只输出 JSON。")

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_loose(raw: str) -> Any:
    """从模型回复里抠出 JSON（dict 或 list）。抠不出来返回 None。

    模型常见毛病：夹 ```json 围栏、前后加一句"好的，以下是结果"、把数组包在对象里。
    这里只做尽力而为，**不猜内容**——抠不出就是 None，让上层降级或重问。
    """
    s = (raw or "").strip()
    if not s:
        return None
    fence = _JSON_FENCE.search(s)
    if fence:
        s = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = s.find(opener), s.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except Exception:  # noqa: BLE001
                continue
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def call_with_retry(run, usable, max_attempts: int = 2) -> tuple[str, str, int]:
    """同一请求最多再问一次，第二次由 ``run(True)`` 带上「别推理」的强化指令。

    ``run(hardened) -> (text, err)``，``usable(text) -> bool``。返回 (文本, 错误, 尝试次数)。
    错误信息去重后拼起来，界面直接能看懂是"空返回"还是"JSON 解析不了"。
    """
    errors: list[str] = []
    attempts = max(1, int(max_attempts))
    for i in range(attempts):
        text, err = run(i > 0)
        if not err and usable(text):
            return text, "", i + 1
        errors.append((err or "输出不可用").strip())
    seen: list[str] = []
    for e in errors:
        if e not in seen:
            seen.append(e)
    return "", "；".join(seen), attempts


@dataclass
class JsonCall:
    """一次结构化调用的结果。``hardened`` 表示是靠降档重问才拿到的。"""

    payload: Any = None
    text: str = ""
    attempts: int = 0
    error: str = ""
    hardened: bool = False

    @property
    def ok(self) -> bool:
        return self.payload is not None and not self.error


@dataclass
class ProbeResult:
    reachable: bool = False
    base_url: str = ""
    latency_ms: float = 0.0
    models: list[str] = field(default_factory=list)
    model_ok: bool = False
    chat_ok: bool = False
    vision_hint: bool = False
    error: str = ""
    checked_at: float = 0.0
    reasoning_only: bool = False   # 模型只回 reasoning_content（推理模型 + max_tokens 太小）

    def summary(self) -> str:
        if not self.reachable:
            return f"不可达：{self.error}"
        bits = [f"可达 {self.latency_ms:.0f}ms"]
        bits.append(f"模型 {len(self.models)} 个" if self.models else "无模型清单")
        if self.chat_ok and self.reasoning_only:
            bits.append("对话 OK（只返回推理内容，输出被截断）")
        else:
            bits.append("对话 OK" if self.chat_ok else "对话未验证/失败")
        if self.vision_hint:
            bits.append("疑似支持视觉")
        return "｜".join(bits)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 vision_model: str = "", temperature: float = 0.3,
                 timeout_s: float = 120.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.vision_model = vision_model or model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client = None

    # -- 内部 -----------------------------------------------------------
    def _c(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url,
                                  api_key=self.api_key or "sk-local",
                                  timeout=self.timeout_s)
        return self._client

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    @staticmethod
    def _text_of(msg) -> tuple[str, str]:
        """(正文, 推理内容)。推理模型的 reasoning_content 不能当答案用。"""
        if msg is None:
            return "", ""
        content = (getattr(msg, "content", "") or "").strip()
        reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
        return content, reasoning

    # -- 探测 -----------------------------------------------------------
    def probe(self, deep: bool = True) -> ProbeResult:
        res = ProbeResult(base_url=self.base_url, checked_at=time.time())
        if not self.base_url:
            res.error = "未配置 base_url"
            return res
        u = urlparse(self.base_url)
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "https" else 80)
        t0 = time.time()
        try:
            socket.create_connection((host, port), timeout=5).close()
            res.reachable = True
            res.latency_ms = (time.time() - t0) * 1000
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            return res
        try:
            models = self._c().models.list()
            res.models = [m.id for m in getattr(models, "data", [])][:50]
            res.model_ok = (not res.models) or (self.model in res.models)
        except Exception as e:
            res.error = f"模型列表失败：{type(e).__name__}: {str(e)[:120]}"
        if deep:
            try:
                r = self._c().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "回复一个字：好"}],
                    max_tokens=64, temperature=0.0,
                )
                msg = r.choices[0].message if r.choices else None
                content, reasoning = self._text_of(msg)
                # 推理模型会把 token 先花在 reasoning_content 上：
                # 只看 content 会把"能对话"误判成"对话失败"。
                res.chat_ok = bool(content or reasoning)
                res.reasoning_only = bool(not content and reasoning)
            except Exception as e:
                res.error = (res.error + "；" if res.error else "") + \
                    f"对话失败：{type(e).__name__}: {str(e)[:120]}"
            lowered = " ".join(res.models).lower()
            res.vision_hint = any(k in lowered for k in ("vl", "vision", "omni", "multimodal"))
        return res

    # -- 调用 -----------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 1200,
             temperature: float | None = None, json_mode: bool = False) -> tuple[str, str]:
        """返回 (文本, 错误)。错误非空时表示降级，调用方应在界面上明确提示。"""
        if not self.configured:
            return "", "未配置模型端点"
        try:
            kw: dict[str, Any] = dict(
                model=self.model, messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            r = self._c().chat.completions.create(**kw)
            if not r.choices:
                return "", "模型没有返回任何候选结果"
            content, reasoning = self._text_of(r.choices[0].message)
            if content:
                return content, ""
            fr = getattr(r.choices[0], "finish_reason", "") or "?"
            if reasoning:
                return "", (f"模型只返回了推理内容（finish_reason={fr}），正文被截断："
                            f"把 max_tokens 调大再试")
            return "", f"模型返回空内容（finish_reason={fr}）"
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:200]}"

    def complete(self, system: str, user: str, max_tokens: int = 1200,
                 json_mode: bool = False) -> tuple[str, str]:
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, json_mode=json_mode,
        )

    # -- 结构化调用：全项目唯一的重试规则 -------------------------------
    def complete_json(self, system: str, user: str, *, expect_key: str | None = None,
                      max_tokens: int = 1200) -> "JsonCall":
        """要一整块 JSON，失败时**降档重问一次**。

        为什么需要这条规则（借 DeepTutor `services/llm/structured_retry.py` 的诊断）：
        推理模型把隐藏思考的 token 从同一个 max_tokens 预算里扣，请求整块 JSON 时
        思考可能先花光预算，一个字都还没吐；**provider 不报错**，只是返回空或截断。
        以前的处理是各调用点自己兜底（识别返回空、重讲给提纲、探测报假阴性），
        症状一样但口径不一。统一成：正常问 → 解析不出来 → 带「别推理」的强化指令再问一次
        （同时把 max_tokens 放宽）→ 还不行才降级。
        """
        if not self.configured:
            # 没配端点就别问两遍：重试是给「问到了但没拿到」用的
            return JsonCall(payload=None, attempts=0, error="未配置模型端点")

        def run(hardened: bool) -> tuple[str, str]:
            sys_prompt = system if not hardened else (system + "\n" + HARDEN_HINT)
            budget = max_tokens if not hardened else min(int(max_tokens * 1.5) + 200, 4000)
            return self.chat(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user}],
                max_tokens=budget, json_mode=True,
                temperature=0.2 if hardened else None,
            )

        def usable(raw: str) -> bool:
            obj = parse_json_loose(raw)
            if obj is None:
                return False
            if expect_key is None:
                return True
            return isinstance(obj, dict) and bool(obj.get(expect_key))

        text, err, attempts = call_with_retry(run, usable)
        payload = parse_json_loose(text) if text else None
        return JsonCall(payload=payload, text=text, attempts=attempts,
                        error="" if payload is not None else (err or "模型没有给出可解析的 JSON"),
                        hardened=attempts > 1)

    def complete_text(self, system: str, user: str, max_tokens: int = 1200) -> "JsonCall":
        """纯文本调用也走同一条重试规则（重讲、讲解这类长文本）。"""
        if not self.configured:
            return JsonCall(payload=None, attempts=0, error="未配置模型端点")

        def run(hardened: bool) -> tuple[str, str]:
            sys_prompt = system if not hardened else (system + "\n" + HARDEN_HINT)
            budget = max_tokens if not hardened else min(int(max_tokens * 1.5) + 200, 4000)
            return self.complete(sys_prompt, user, max_tokens=budget)

        text, err, attempts = call_with_retry(run, lambda raw: bool((raw or "").strip()))
        return JsonCall(payload=None, text=text, attempts=attempts,
                        error="" if text.strip() else (err or "模型返回空内容"),
                        hardened=attempts > 1)

    def preload(self) -> tuple[bool, str]:
        """启动时预热（用户裁决：模型在启动时选择并加载，课上永不冷启）。"""
        if not self.configured:
            return False, "未配置模型端点，跳过预热"
        t0 = time.time()
        try:
            self._c().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=64, temperature=0.0,
            )
        except Exception as e:  # noqa: BLE001
            return False, f"预热失败（{type(e).__name__}: {str(e)[:120]}）"
        return True, f"预热完成 {time.time() - t0:.1f}s"


def build_llm(settings) -> LLMClient:
    l = settings.llm
    return LLMClient(l.base_url, l.api_key, l.model, l.vision_model,
                     l.temperature, l.timeout_s)