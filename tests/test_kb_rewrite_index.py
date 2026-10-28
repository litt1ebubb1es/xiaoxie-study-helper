"""第一批 P1 的两件事：查询改写 fail-open + 索引签名/过期检测。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsh_app import kb as kbmod
from dsh_app.chat import ask, kb_search, rewrite_query
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.llm import JsonCall
from dsh_app.pipeline import Engine

DOC = """# 第三章 矩阵的秩

矩阵的秩定义为行阶梯形矩阵中非零行的个数。

## 3.2 与特征值的关系

非零特征值的个数不超过秩。这一点在判断可否对角化时经常用到。
"""


class FakeLLM:
    def __init__(self, text: str = "", configured: bool = True, payload=None):
        self.text = text
        self.configured = configured
        self.payload = payload
        self.calls = 0

    def complete_text(self, system, user, max_tokens=1200) -> JsonCall:
        self.calls += 1
        if self.text == "RAISE":
            raise RuntimeError("模型炸了")
        return JsonCall(text=self.text, attempts=1, error="" if self.text else "空")

    def complete_json(self, system, user, *, expect_key=None, max_tokens=1200) -> JsonCall:
        return JsonCall(payload=self.payload, text=json.dumps(self.payload or {}),
                        attempts=1, error="")

    def chat(self, messages, max_tokens=1200, **kw):
        self.calls += 1
        return self.text, ""


@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    e = Engine(db, Settings.load())
    e.llm = FakeLLM(text="答案")
    d = tmp_path / "n.md"
    d.write_text(DOC, encoding="utf-8")
    assert e.kb_import(str(d))["ok"]
    return e, d


# ---------------------------------------------------------------- 查询改写
def test_rewrite_returns_keywords_and_is_used_when_first_search_is_empty(eng):
    e, _ = eng
    # 「它」这种指代 BM25 检索不到
    assert kb_search(e.db, "它是什么意思", "", 4) == []
    e.llm = FakeLLM(text="秩 特征值 对角化")
    res = ask(e.db, e.llm, "", "它是什么意思", use_kb=True, use_context=False)
    assert res["query_used"] == "秩 特征值 对角化"
    assert res["sources"], "改写后的查询应该检索得到东西"
    assert res["sources"][0]["doc"] == "n"


def test_rewrite_fails_open_when_model_raises(eng):
    e, _ = eng
    e.llm = FakeLLM(text="RAISE")
    out = rewrite_query(e.llm, "它是什么", "上下文")
    assert out == "它是什么", "改写失败必须原样返回，不能吞掉用户的问题"


def test_rewrite_fails_open_when_not_configured(eng):
    e, _ = eng
    e.llm = FakeLLM(text="秩", configured=False)
    assert rewrite_query(e.llm, "它是什么") == "它是什么"
    assert e.llm.calls == 0, "没配端点连试都不该试"


def test_rewrite_fails_open_on_garbage_output(eng):
    e, _ = eng
    e.llm = FakeLLM(text="嗯……我觉得用户想问的是关于矩阵的内容，大概是秩吧" * 6)
    assert rewrite_query(e.llm, "它是什么") == "它是什么"


def test_rewrite_is_not_used_when_first_search_already_hits(eng):
    e, _ = eng
    e.llm = FakeLLM(text="不该被用到")
    res = ask(e.db, e.llm, "", "矩阵的秩是什么", use_kb=True, use_context=False)
    assert res["sources"]
    assert res["query_used"] == "矩阵的秩是什么"
    assert e.llm.calls == 1, "只应发生一次问答调用，没有额外改写"


# ---------------------------------------------------------------- 索引签名
def test_import_records_signature_and_is_not_stale(eng):
    e, _ = eng
    st = e.kb_index_status()
    assert st["has_index"] and not st["stale"]
    assert st["chunk"] == [kbmod.MAX_CHUNK, kbmod.OVERLAP]
    assert st["sig"] == st["stored"]


def test_changing_chunk_params_marks_index_stale(eng, monkeypatch):
    e, _ = eng
    monkeypatch.setattr(kbmod, "MAX_CHUNK", 240)   # 参数变了，旧索引就该被指出来
    st = e.kb_index_status()
    assert st["stale"] is True
    assert st["sig"] != st["stored"]


def test_rebuild_refreshes_signature_and_chunks(eng, monkeypatch):
    e, _ = eng
    before = e.kb_index_status()
    monkeypatch.setattr(kbmod, "MAX_CHUNK", 240)
    assert e.kb_index_status()["stale"]
    res = e.rebuild_kb_index()
    assert res["ok"] and res["rebuilt"] == 1
    after = e.kb_index_status()
    assert not after["stale"]
    assert after["sig"] == after["stored"] != before["sig"]


def test_rebuild_reports_missing_source_file_instead_of_skipping(eng):
    e, path = eng
    path.rename(path.with_suffix(".gone"))
    res = e.rebuild_kb_index()
    assert res["rebuilt"] == 0
    assert res["missing"] == ["n"], "读不回原文件必须报出来"


def test_rebuild_keeps_page_locators(eng):
    e, _ = eng
    e.rebuild_kb_index()
    rows = e.db.kb_chunks()
    assert any(str(r["section"] or "") for r in rows), "重建不该丢掉小节信息"