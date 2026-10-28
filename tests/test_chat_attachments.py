"""问模型小窗：附件分类 + 带图提问的回归测试。

为什么要测：
1. 图片如果没进 `image_url` 段、而是被拼成文本，模型只会"猜图里有什么"——
   界面上完全看不出来，答案还像模像样。所以必须断言消息结构。
2. 不支持的文件（exe/zip）如果被静默丢掉，用户会以为发出去了。
3. 只发附件不打字是最常见的用法（"这张图啥意思"），不能因为"问题是空的"就拒发。
"""
from __future__ import annotations

import base64
import io

import pytest

from dsh_app import attachments as att
from dsh_app.chat import ask, build_messages
from dsh_app.db import DB

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/"
    "AGtHw9jAAAAAElFTkSuQmCC")


class FakeLLM:
    """记录发出去的消息，不联网。"""

    configured = True

    def __init__(self):
        self.sent: list[list[dict]] = []

    def chat(self, messages, max_tokens=1200, temperature=None, json_mode=False):
        self.sent.append(messages)
        return "这是回答", ""


@pytest.fixture()
def db(tmp_path):
    return DB(tmp_path / "t.db")


# ------------------------------------------------------------------ 附件分类
def test_image_becomes_data_url(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(PNG_1PX)
    a = att.load_path(p)
    assert a.ok and a.kind == "image"
    assert a.data_url.startswith("data:image/jpeg;base64,")
    assert a.raw          # 原字节留着做缩略图


def test_text_file_is_read_as_material(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# 分布函数\n右连续，左极限为 P(X<x)。", encoding="utf-8")
    a = att.load_path(p)
    assert a.ok and a.kind == "file"
    assert "右连续" in a.text


def test_unknown_suffix_is_rejected_with_reason(tmp_path):
    p = tmp_path / "setup.exe"
    p.write_bytes(b"MZ....")
    a = att.load_path(p)
    assert not a.ok
    assert "不支持" in a.error and ".exe" in a.error


def test_missing_file_and_directory_are_rejected(tmp_path):
    assert not att.load_path(tmp_path / "nope.png").ok
    assert "文件夹" in att.load_path(tmp_path).error


def test_split_separates_usable_and_broken(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_1PX)
    bad = tmp_path / "a.zip"
    bad.write_bytes(b"PK")
    got = att.load_paths([str(img), str(bad)])
    images, files, problems = att.split(got)
    assert len(images) == 1 and not files and len(problems) == 1


def test_prompt_and_history_label():
    files = [att.Attachment(kind="file", name="讲义.pdf", text="正文", note="抽了 3 页")]
    assert "讲义.pdf" in att.to_prompt([], files) and "正文" in att.to_prompt([], files)
    label = att.history_label([att.Attachment(kind="image", name="a.png")])
    assert "附件" in label and "a.png" in label


# ------------------------------------------------------------------ 消息结构
def test_images_go_as_image_url_parts():
    msgs = build_messages("这块什么意思", "", [], None, images=["data:image/jpeg;base64,AAA"])
    content = msgs[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text" and "这块什么意思" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg")


def test_without_images_content_stays_plain_text():
    msgs = build_messages("问题", "", [], None)
    assert isinstance(msgs[-1]["content"], str)


def test_images_are_capped_at_four():
    urls = [f"data:image/jpeg;base64,{i}" for i in range(9)]
    msgs = build_messages("看图", "", [], None, images=urls)
    pics = [c for c in msgs[-1]["content"] if c["type"] == "image_url"]
    assert len(pics) == 4       # 再多模型也看不过来，还容易超时


# ------------------------------------------------------------------ ask 接线
def test_ask_sends_image_parts_and_files(db):
    llm = FakeLLM()
    res = ask(db, llm, "", "这页讲的什么", images=["data:image/jpeg;base64,AAA"],
              files=[("讲义.pdf", "分布函数右连续")])
    assert res["answer"] == "这是回答"
    sent = llm.sent[0]
    assert isinstance(sent[-1]["content"], list)
    joined = "".join(c.get("text", "") for c in sent[-1]["content"] if c["type"] == "text")
    assert "讲义.pdf" in joined and "右连续" in joined


def test_ask_with_only_attachment_fills_a_default_question(db):
    llm = FakeLLM()
    res = ask(db, llm, "", "", images=["data:image/jpeg;base64,AAA"])
    assert not res["error"]
    text = llm.sent[0][-1]["content"][0]["text"]
    assert "用户的问题" in text and len(text.split("用户的问题】")[-1].strip()) > 4


def test_ask_with_nothing_is_still_rejected(db):
    res = ask(db, FakeLLM(), "", "   ")
    assert res["error"] == "问题是空的"


def test_history_records_what_was_attached(db):
    db_row = None
    ask(db, FakeLLM(), "", "看图", images=["data:image/jpeg;base64,AAA"],
        files=[("讲义.pdf", "正文")])
    hist = db.get_chat("", 5)
    assert hist, "对话历史应该有记录"
    db_row = hist[-2] if len(hist) > 1 else hist[-1]
    content = str(db_row["content"])
    assert "[附件：讲义.pdf]" in content and "[图片 ×1]" in content