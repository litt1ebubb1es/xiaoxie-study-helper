"""问模型小窗的界面行为测试（offscreen，不需要真窗口）。

测的是"看不出来坏了"的三件事：
1. 粘贴截图 → 变成附件（而不是被当成文字吞掉）；
2. Enter 发送、Shift+Enter 换行（后者不该触发发送）；
3. 发送时附件真的进了 engine.ask 的参数（不是只在界面上显示）。
"""
from __future__ import annotations

import pytest

from dsh_app.ui.chat_window import ChatWindow


class DummyEngine:
    """只记录调用参数，不联网。"""

    def __init__(self):
        self.calls: list[dict] = []

    def ask(self, question, session_id="", use_kb=True, use_context=True,
            images=(), files=()):
        self.calls.append({"q": question, "session_id": session_id,
                           "images": list(images), "files": list(files)})
        return {"answer": "好的", "error": "", "sources": []}


class DummyLLM:
    configured = True
    model = "test-model"


class DummySettings:
    def __init__(self):
        self.llm = DummyLLM()


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("dsh_app.ui.chat_window.UI_STATE", tmp_path / "ui_state.json")
    w = ChatWindow(DummyEngine(), DummySettings(), [], session_id="sid-1")
    yield w
    w.close()


def _png_image():
    from PySide6.QtGui import QColor, QImage

    img = QImage(40, 24, QImage.Format_RGB32)
    img.fill(QColor("#F2621B"))
    return img


def test_paste_image_becomes_attachment(win, qapp):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setImage(_png_image())
    win.add_image(QGuiApplication.clipboard().image())
    assert len(win._atts) == 1 and win._atts[0].kind == "image"
    assert not win.att_card.isHidden()          # 有附件时附件条必须出现
    assert "已粘贴" in win.lbl_state.text()


def test_attachment_strip_is_hidden_when_empty(win):
    assert win.att_card.isVisibleTo(win) is False


def test_ctrl_v_in_editor_captures_image_without_inserting_text(win, qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent

    QGuiApplication.clipboard().setImage(_png_image())
    ev = QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier)
    handled = win.eventFilter(win.edit, ev)
    assert handled is True
    assert len(win._atts) == 1


def test_send_passes_images_to_engine(win, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    win.add_paths([str(img)])
    win._on_send()                 # 没打字、只发附件：这是最常用的"看看这张图"
    if win._worker is not None:
        win._worker.wait(3000)
    assert win.engine.calls, "应该调用过 engine.ask"
    call = win.engine.calls[-1]
    assert call["images"], "图片应作为 data URL 传给 engine.ask"


def test_enter_sends_but_shift_enter_does_not(win):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    win.edit.setPlainText("问题")
    shift = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.ShiftModifier)
    assert win.eventFilter(win.edit, shift) is False      # 换行：交给输入框自己处理
    plain = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
    assert win.eventFilter(win.edit, plain) is True       # 发送
    win._worker and win._worker.wait(3000)


def test_dropping_unsupported_file_reports_reason(win, tmp_path):
    exe = tmp_path / "x.exe"
    exe.write_bytes(b"MZ")
    win.add_paths([str(exe)])
    assert win._atts and not win._atts[0].ok
    assert "没收下" in win.lbl_state.text()


def test_window_state_is_remembered(win, tmp_path, qapp):
    win.resize(480, 600)
    win.close()
    saved = (tmp_path / "ui_state.json").read_text(encoding="utf-8")
    assert "chat_window" in saved and "600" in saved