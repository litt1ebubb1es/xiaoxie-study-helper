"""模型列表探测：端点"不给列表"时必须给得出候选，而且不能抛怪错误。

这是踩过的坑，两个独立问题：

1. **崩溃点**：以前用 SDK 的 `client.models.list()`。它对非标准网关太脆——
   返回 `{}` → `TypeError: 'NoneType' object is not iterable`；
   返回 HTML → `AttributeError: 'str' object has no attribute ...`；
   br 压缩撞上老 brotli → `TypeError: process() takes no keyword arguments`。
   用户看到「模型列表失败：TypeError: process() ...」，跟"模型"完全联系不起来。
   现在自己 GET /models 并按多形态解析，失败只返回空列表 + 一句人话。

2. **不能用"手填模型名"当兜底**：界面要直接用预设候选填好下拉
   （`main_window._models_done` 的分支逻辑，这里用同一份 `model_names_of` 数据验证）。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dsh_app import netfix
from dsh_app.llm import LLMClient, build_llm, model_names_of


# --------------------------------------------------------------- 解析各种形态
def test_model_names_of_openai_shape():
    assert model_names_of({"object": "list",
                           "data": [{"id": "qwen-plus"}, {"id": "qwen-max"}]}) == \
        ["qwen-plus", "qwen-max"]


def test_model_names_of_other_shapes():
    assert model_names_of({"models": [{"name": "llama3"}, {"id": "qwen2.5"}]}) == \
        ["llama3", "qwen2.5"]
    assert model_names_of({"result": [{"model": "glm-4"}]}) == ["glm-4"]
    assert model_names_of(["a", "b "]) == ["a", "b"]          # 裸数组，顺手去空白
    assert model_names_of({"data": [{"id": "m"}, {"id": "m"}]}) == ["m"]   # 去重保序


def test_model_names_of_never_raises():
    for bad in ({}, {"data": None}, [], "", "<html>gateway</html>", {"data": "x"}, None):
        assert model_names_of(bad) == []


# ----------------------------------------------------------------- 假网关
class _Gateway:
    """按预设的 (状态码, content-type, body) 应答，一次一个。"""

    def __init__(self, responses):
        self.responses = list(responses)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):                       # noqa: N802
                code, ctype, body = outer.responses.pop(0)
                raw = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):              # 静音
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def stop(self):
        self.srv.shutdown()


@pytest.fixture()
def gateway():
    made = []

    def make(responses):
        g = _Gateway(responses)
        made.append(g)
        return g

    yield make
    for g in made:
        g.stop()


def _client(base_url: str) -> LLMClient:
    return LLMClient(base_url=base_url, api_key="sk-test", model="m1", timeout_s=5)


def test_list_models_ok(gateway):
    g = gateway([(200, "application/json",
                  json.dumps({"object": "list", "data": [{"id": "m1"}, {"id": "m2"}]}))])
    names, err = _client(g.base_url).list_models()
    assert names == ["m1", "m2"] and err == ""


def test_list_models_empty_object_is_not_a_crash(gateway):
    """网关对 /models 回 `{}`：以前抛 TypeError，现在给空列表 + 原因。"""
    g = gateway([(200, "application/json", "{}")])
    names, err = _client(g.base_url).list_models()
    assert names == []
    assert "没有模型名" in err


def test_list_models_html_is_not_a_crash(gateway):
    g = gateway([(200, "text/html", "<html>gateway</html>")])
    names, err = _client(g.base_url).list_models()
    assert names == []
    assert "不是 JSON" in err


def test_list_models_http_error(gateway):
    g = gateway([(404, "application/json", '{"error":"not found"}')])
    names, err = _client(g.base_url).list_models()
    assert names == []
    assert err.startswith("HTTP 404")


def test_list_models_unreachable_gives_text_not_exception():
    names, err = _client("http://127.0.0.1:9/v1").list_models()
    assert names == []
    assert err                                   # 有原因，且没抛出来


def test_probe_does_not_raise_on_bad_gateway(gateway):
    g = gateway([(200, "application/json", "{}"), (500, "text/html", "boom")])
    res = _client(g.base_url).probe(deep=False)
    assert res.models == []
    assert res.error.startswith("模型列表失败：")
    assert "TypeError" not in res.error           # 不再是那种看不懂的报错


def test_probe_reports_model_ok(gateway):
    g = gateway([(200, "application/json",
                  json.dumps({"data": [{"id": "m1"}]})), (200, "text/plain", "ok")])
    res = _client(g.base_url).probe(deep=False)
    assert res.models == ["m1"] and res.model_ok is True


# ------------------------------------------------------- br 压缩的兼容补丁
def test_patch_keeps_br_when_brotli_is_fine(monkeypatch):
    monkeypatch.setattr(netfix, "_brotli_supports_output_limit", lambda: True)
    assert netfix.patch_http_encodings() == ""


def test_patch_drops_br_when_brotli_too_old(monkeypatch):
    """brotli 1.1.0 的 C process() 不认 output_buffer_limit：摘掉 br，改 gzip。"""
    from httpx2 import _client as hx_client
    from httpx2 import _decoders as hx_decoders

    table = dict(hx_decoders.SUPPORTED_DECODERS)
    accept = hx_client.ACCEPT_ENCODING
    try:
        hx_decoders.SUPPORTED_DECODERS["br"] = table.get(
            "br") or hx_decoders.IdentityDecoder
        monkeypatch.setattr(netfix, "_brotli_supports_output_limit", lambda: False)
        note = netfix.patch_http_encodings()
        assert "br" not in hx_decoders.SUPPORTED_DECODERS
        assert "br" not in hx_client.ACCEPT_ENCODING
        assert "gzip" in hx_client.ACCEPT_ENCODING
        assert "process()" in note                # 补丁说明里写明踩过的坑
    finally:
        hx_decoders.SUPPORTED_DECODERS.clear()
        hx_decoders.SUPPORTED_DECODERS.update(table)
        hx_client.ACCEPT_ENCODING = accept


def test_patch_is_idempotent(monkeypatch):
    monkeypatch.setattr(netfix, "_brotli_supports_output_limit", lambda: False)
    netfix.patch_http_encodings()
    assert netfix.patch_http_encodings() == ""    # 已经摘过，第二次什么都不做


# ------------------------------------------------------------------ 预设候选
def test_build_llm_uses_settings():
    class S:
        class llm:                    # noqa: N801
            base_url = "http://127.0.0.1:1234/v1"
            api_key = ""
            model = "qwen2.5:7b"
            vision_model = ""
            temperature = 0.3
            timeout_s = 30.0

    c = build_llm(S)
    assert c.base_url == "http://127.0.0.1:1234/v1" and c.model == "qwen2.5:7b"