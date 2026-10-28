"""小测 / 待答问题 / 覆盖度 / 笔记导出 / 结构化重试 —— 用假模型跑通整条链。

为什么整套都用假 LLM：这些功能的价值不在"模型答得好不好"，而在
**模型给了不靠谱的东西时我们会怎么处理**（作废、重问、降级、不许假装成功）。
假模型可以精确制造这些坏情况，真模型不行。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsh_app import quiz as qz
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.llm import JsonCall, LLMClient
from dsh_app.pipeline import Engine

SRC = ("矩阵的秩定义为行阶梯形矩阵中非零行的个数。秩等于列空间的维数。"
       "求秩的常用办法是初等行变换化成行阶梯形。")


# ---------------------------------------------------------------- 假模型
class FakeLLM:
    """按顺序吐出预置回复的假客户端，用来制造"模型不靠谱"的各种现场。"""

    def __init__(self, payloads: list[dict] | None = None, text: str = "",
                 configured: bool = True):
        self.payloads = list(payloads or [])
        self.text = text
        self.configured = configured
        self.calls: list[str] = []

    def complete_json(self, system, user, *, expect_key=None, max_tokens=1200) -> JsonCall:
        self.calls.append("json")
        if not self.payloads:
            return JsonCall(payload=None, attempts=1, error="假模型没有更多回复了")
        body = self.payloads.pop(0)
        return JsonCall(payload=body, text=json.dumps(body, ensure_ascii=False),
                        attempts=1, error="")

    def complete_text(self, system, user, max_tokens=1200) -> JsonCall:
        self.calls.append("text")
        return JsonCall(text=self.text, attempts=1, error="" if self.text else "空")

    def chat(self, messages, max_tokens=1200, **kw):
        return self.text, ""


def _quiz_payload(evidence: str, prompt: str = "关于矩阵的秩，下列说法哪一个是正确的？"):
    return {"questions": [{
        "prompt": prompt,
        "choices": ["非零行的个数", "矩阵的行数", "矩阵的列数", "零元素的个数"],
        "correct_choice_index": 0,
        "evidence": evidence,
    }]}


@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    e = Engine(db, Settings.load())
    e.llm = FakeLLM()
    sid = e.seed_demo()
    return e, sid


# ---------------------------------------------------------------- 出题校验
def test_valid_question_passes_and_gets_evidence_checked():
    items, dropped = qz.parse_questions(_quiz_payload("秩定义为行阶梯形矩阵中非零行的个数"), SRC)
    assert not dropped
    assert len(items) == 1
    assert items[0].evidence_ok is True
    assert items[0].void is False
    assert len(items[0].choices) == 4


def test_forged_evidence_voids_the_question_instead_of_dropping_it():
    """出处查不到 → 题还在（让人看见），但作废、不计分。"""
    items, _ = qz.parse_questions(_quiz_payload("教材第 88 页指出这个结论成立"), SRC)
    assert len(items) == 1
    assert items[0].evidence_ok is False
    assert items[0].void is True
    assert "出处" in items[0].void_reason
    verdict = qz.grading_verdict(items[0], 0)
    assert verdict["result"] == "void"
    assert verdict["counts"] is False


def test_bad_questions_are_dropped_not_patched():
    bad_prompt = _quiz_payload("秩定义为行阶梯形矩阵中非零行的个数", prompt="短")
    assert qz.parse_questions(bad_prompt, SRC)[0] == []
    dup = _quiz_payload("秩定义为行阶梯形矩阵中非零行的个数")
    dup["questions"][0]["choices"] = ["一样", "一样", "不同", "都不对"]
    items, dropped = qz.parse_questions(dup, SRC)
    assert items == [] and "重复" in dropped[0]
    out_of_range = _quiz_payload("秩定义为行阶梯形矩阵中非零行的个数")
    out_of_range["questions"][0]["correct_choice_index"] = 9
    assert qz.parse_questions(out_of_range, SRC)[0] == []


def test_option_prefixes_are_stripped_but_math_is_not():
    assert qz.strip_option_prefixes(["A. 甲", "B. 乙", "C. 丙", "D. 丁"]) == ["甲", "乙", "丙", "丁"]
    # 「x - 1 = 0」不能被当成「选项 X」——DeepTutor 记录过这个错位
    keep = ["x - 1 = 0", "x = 2", "x = 3", "无解"]
    assert qz.strip_option_prefixes(keep) == keep


def test_evidence_key_ignores_layout_noise():
    assert qz.grounded("非零行的个数", "秩定义为行阶梯形矩阵中\n3\n非零行的个数")
    assert qz.grounded("DeepTu-\ntor", "DeepTu-\n4\ntor 是一个项目")
    assert not qz.grounded("完全无关的一句话", SRC)


def test_grading_is_deterministic():
    items, _ = qz.parse_questions(_quiz_payload("秩定义为行阶梯形矩阵中非零行的个数"), SRC)
    item = items[0]
    assert qz.grading_verdict(item, 0)["correct"] is True
    wrong = qz.grading_verdict(item, 2)
    assert wrong["correct"] is False and wrong["counts"] is True
    assert "第 1 项" in wrong["reason"]


# ---------------------------------------------------------------- 引擎：出题→判分→掌握度
def test_make_quiz_attaches_kp_and_grading_moves_mastery(eng):
    e, sid = eng
    knot = e.db.get_knots(sid)[0]
    payload = _quiz_payload(knot.excerpt[:20] or "必要条件")
    # 让引文一定能回查到：用这一段原文里的片段
    src = e.knot_source_text(knot.id)
    payload["questions"][0]["evidence"] = src["text"][:20]
    e.llm = FakeLLM([payload])
    res = e.make_quiz(knot.id)
    assert res["ok"], res
    item = res["items"][0]
    assert item["void"] is False

    before = {int(r["id"]): float(r["mastery"] or 0)
              for r in e.db.get_knowledge_points(e.db.get_session(sid)["course_id"])}
    row = e.db.query("SELECT * FROM quiz_items WHERE id=?", (item["item_id"],))[0]
    kp_id = int(row["kp_id"])
    assert before[kp_id] == pytest.approx(0.0)

    ans = e.answer_quiz(item["item_id"], int(row["correct_index"]))
    assert ans["counts"] and ans["correct"]
    after = {int(r["id"]): float(r["mastery"] or 0)
             for r in e.db.get_knowledge_points(e.db.get_session(sid)["course_id"])}
    assert after[kp_id] > before[kp_id]

    wrong = e.answer_quiz(item["item_id"], (int(row["correct_index"]) + 1) % 4)
    assert wrong["counts"] and not wrong["correct"]
    assert wrong["kp"]["mastery"] < after[kp_id]


def test_void_quiz_does_not_write_mastery(eng):
    e, sid = eng
    knot = e.db.get_knots(sid)[0]
    e.llm = FakeLLM([_quiz_payload("这句话原文里根本没有")])
    res = e.make_quiz(knot.id)
    assert res["ok"] and res["void"], res
    item_id = res["items"][0]["item_id"]
    row = e.db.query("SELECT * FROM quiz_items WHERE id=?", (item_id,))[0]
    n_before = len(e.db.get_kp_attempts(int(row["kp_id"]), limit=100))
    ans = e.answer_quiz(item_id, 0)
    assert ans["result"] == "void" and ans["kp"] is None
    assert len(e.db.get_kp_attempts(int(row["kp_id"]), limit=100)) == n_before


def test_make_quiz_without_model_says_so_and_stores_nothing(eng):
    e, sid = eng
    knot = e.db.get_knots(sid)[0]
    e.llm = FakeLLM(configured=False)
    e.llm.payloads = []
    res = e.make_quiz(knot.id)
    assert res["ok"] is False
    assert e.db.get_quiz_items(sid) == []


# ---------------------------------------------------------------- 待答问题
def test_question_queue_roundtrip(eng):
    e, sid = eng
    q = e.queue_question(sid, "为什么秩等于列空间的维数？", t_sec=120.0)
    assert q["ok"] and q["excerpt"]
    rows = e.db.get_questions(sid, "pending")
    assert len(rows) == 1
    e.llm = FakeLLM(text="因为行秩等于列秩。")
    res = e.answer_question(q["id"])
    assert res["ok"]
    assert e.db.get_questions(sid, "pending") == []
    answered = e.db.get_questions(sid, "answered")
    assert "行秩" in answered[0]["answer"]


def test_question_without_timestamp_uses_last_known_position(eng):
    e, sid = eng
    pos = e.current_position(sid)
    q = e.queue_question(sid, "没有时间点的问题")
    assert q["ok"]
    assert q["t_sec"] == pytest.approx(pos)


def test_answer_all_stops_when_model_cannot_answer(eng):
    e, sid = eng
    e.queue_question(sid, "问题一", t_sec=10)
    e.queue_question(sid, "问题二", t_sec=20)
    e.llm = FakeLLM(text="")          # 答不出来
    res = e.answer_all_questions(sid)
    assert res["answered"] == 0
    assert len(res["failed"]) == 2
    assert len(e.db.get_questions(sid, "pending")) == 2


# ---------------------------------------------------------------- 覆盖度
def test_coverage_counts_knots_without_frames(eng):
    e, sid = eng
    cov = e.capture_coverage(sid)
    assert cov["knots"] == len(e.db.get_knots(sid))
    assert cov["frames"] == 0
    assert cov["covered"] == 0
    assert cov["missing"] == cov["knots"]


def test_coverage_marks_a_knot_covered_by_a_nearby_frame(eng):
    e, sid = eng
    knot = e.db.get_knots(sid)[0]
    e.db.add_screen_note(sid, float(knot.t_start) + 5, "板书：秩的定义", kind="board")
    cov = e.capture_coverage(sid)
    assert cov["covered"] == 1
    assert knot.id not in cov["missing_ids"]


# ---------------------------------------------------------------- 导出
def test_export_notes_writes_markdown_with_all_sections(eng, tmp_path):
    e, sid = eng
    e.queue_question(sid, "为什么秩等于列空间的维数？", t_sec=100)
    e.db.set_question_status(e.db.get_questions(sid, "pending")[0]["id"],
                             "answered", "行秩等于列秩。")
    e.db.add_screen_note(sid, 100.0, "板书：秩的定义", kind="board")
    out = tmp_path / "notes.md"
    res = e.export_notes(sid, out)
    assert res["ok"] and out.exists()
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# ")
    assert "## 回看点与掌握度" in md
    assert "## 屏幕笔记" in md
    assert "## 累积的问题" in md
    assert "画面覆盖：" in md
    assert "行秩等于列秩" in md


def test_export_notes_default_path_under_data(eng, tmp_path, monkeypatch):
    """默认不传路径时要落到配置的导出目录（测试里换成临时目录，别污染仓库）。"""
    import dsh_app.config as cfg

    monkeypatch.setattr(cfg, "EXPORT_DIR", tmp_path / "exports")
    e, sid = eng
    res = e.export_notes(sid)
    assert res["ok"]
    p = Path(res["path"])
    assert p.exists() and p.suffix == ".md"
    assert p.parent == tmp_path / "exports"


# ---------------------------------------------------------------- 结构化重试规则
def test_complete_json_retries_once_with_hardened_prompt():
    from dsh_app.llm import HARDEN_HINT

    seen: list[bool] = []
    client = LLMClient("http://127.0.0.1:1/v1", "k", "m")
    replies = [("", "模型只返回了推理内容"), ('{"questions":[1]}', "")]

    def fake_chat(messages, max_tokens=1200, **kw):
        seen.append(HARDEN_HINT in messages[0]["content"])
        return replies.pop(0)

    client.chat = fake_chat          # type: ignore[assignment]
    call = client.complete_json("出题", "内容", expect_key="questions")
    assert call.ok and call.attempts == 2 and call.hardened
    assert seen == [False, True], "第二次必须带上「别推理」的强化指令"


def test_complete_json_fails_fast_when_unconfigured():
    client = LLMClient("", "", "m")
    call = client.complete_json("s", "u")
    assert call.attempts == 0 and call.error


def test_complete_json_rejects_payload_missing_expected_key():
    client = LLMClient("http://127.0.0.1:1/v1", "k", "m")
    client.chat = lambda *a, **k: ('{"other":[]}', "")   # type: ignore[assignment]
    call = client.complete_json("s", "u", expect_key="questions")
    assert not call.ok
    assert call.attempts == 2


def test_screen_result_parse_accepts_fenced_json():
    from dsh_app.screen import parse_vision_reply

    text, kind = parse_vision_reply('```json\n{"text":"板书内容","kind":"board"}\n```')
    assert (text, kind) == ("板书内容", "board")
    assert parse_vision_reply("看起来是一页课件") == ("看起来是一页课件", "unknown")