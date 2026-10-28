"""共享 fixture：Qt 相关测试要有 QApplication（offscreen 即可）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 测试绝不出网：把默认设置里的模型端点清掉。
# 之前 seed_demo() 会顺带走一遍术语抽取，如果本机配了 key，每个用例都会真的打云端接口
# （整套测试从 14 秒涨到 158 秒，而且结果随网络波动）。
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_settings(monkeypatch, request):
    # 「测设置本身」的用例需要真的读配置，给它们一个出口
    if request.node.get_closest_marker("real_settings"):
        yield
        return
    from dsh_app.config import Settings

    real_load = Settings.load

    def _offline(*a, **kw):
        s = real_load(*a, **kw)
        s.llm.base_url = ""
        s.llm.api_key = ""
        s.llm.vision_base_url = ""
        s.llm.vision_api_key = ""
        s.llm.preload_on_start = False
        return s

    monkeypatch.setattr(Settings, "load", staticmethod(_offline))
    yield


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
