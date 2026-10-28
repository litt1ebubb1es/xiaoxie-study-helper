"""打包相关的回归测试：数据目录选对、成品里不能夹带我的配置。

这两件事出错都会很难看：
- 数据目录选错 → 换台机器一开就崩（往只读目录写库）；
- 成品夹带 `data/config.json` → 我的 API Key 跟着 zip 发给别人。
"""
from __future__ import annotations

import zipfile

from dsh_app.config import _pick_data_dir


def test_pick_data_dir_prefers_program_folder(tmp_path):
    base = tmp_path / "app"
    base.mkdir()
    assert _pick_data_dir(base) == base / "data"
    assert (base / "data").is_dir()


def test_pick_data_dir_falls_back_when_unwritable(tmp_path, monkeypatch):
    """程序目录不可写（比如装进 Program Files）时退到 %LOCALAPPDATA%。"""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    bogus = tmp_path / "not_a_dir.txt"
    bogus.write_text("x", encoding="utf-8")      # 拿文件当目录用 → mkdir 必然失败

    got = _pick_data_dir(bogus)
    assert got == tmp_path / "local" / "小蟹学习助手" / "data"
    assert got.is_dir()


def test_zip_excludes_runtime_data(tmp_path):
    """成品 zip 里只能有空 data\\ 目录，不能有库、配置、日志。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import build_win

    dist = tmp_path / "dist"
    app = dist / build_win.APP_NAME
    (app / "_internal").mkdir(parents=True)
    (app / f"{build_win.APP_NAME}.exe").write_bytes(b"MZ fake")
    (app / "data" / "logs").mkdir(parents=True)
    (app / "data" / "config.json").write_text('{"llm": {"api_key": "sk-secret"}}',
                                              encoding="utf-8")
    (app / "data" / "app.db").write_bytes(b"db")
    (app / "data" / "logs" / "app.log").write_text("我的运行日志", encoding="utf-8")

    build_win.DIST = dist
    out = build_win._zip(app)
    try:
        names = zipfile.ZipFile(out).namelist()
        assert any(n.endswith(".exe") for n in names)
        assert any(n.endswith("/data/") or n.endswith("/data") for n in names)
        leaked = [n for n in names if "/data/" in n and not n.endswith("/")]
        assert leaked == [], f"运行时数据被打进 zip 了：{leaked}"
        assert not any("sk-secret" in zipfile.ZipFile(out).read(n).decode("utf-8", "ignore")
                       for n in names if n.endswith((".json", ".txt", ".md")))
    finally:
        out.unlink(missing_ok=True)


def test_usage_txt_tells_how_to_start():
    """使用说明必须写到"双击哪个文件"和"数据在哪"，否则用户只能来问。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import build_win

    txt = build_win.USAGE_TXT
    assert "小蟹学习助手.exe" in txt
    assert "data" in txt
    assert "怎么导入模型" in txt

def test_post_writes_gbk_selftest_bat(tmp_path):
    """一键自检.bat 必须是 GBK：cmd 按系统 ANSI 读脚本，UTF-8 的中文会变成乱码指令。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import build_win

    dist = tmp_path / "dist"
    app = dist / build_win.APP_NAME
    app.mkdir(parents=True)
    (app / f"{build_win.APP_NAME}.exe").write_bytes(b"MZ fake")
    build_win._post(app)

    bat = app / "一键自检.bat"
    assert bat.exists()
    raw = bat.read_bytes()
    assert b"\r\n" in raw                       # cmd 不吃纯 LF
    text = raw.decode("gbk")                    # 能按 GBK 解开，说明没写成 UTF-8
    assert "--selftest" in text
    assert "小蟹学习助手.exe" in text
    assert "app.log" in text and "logs" in text   # 失败时指路到日志


def test_post_clears_runtime_files(tmp_path):
    """我这台机器自检产生的库/配置/日志不能顺手发给别人。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import build_win

    dist = tmp_path / "dist"
    app = dist / build_win.APP_NAME
    (app / "data" / "logs").mkdir(parents=True)
    (app / f"{build_win.APP_NAME}.exe").write_bytes(b"MZ fake")
    for junk in ("selftest.db", "app.db", "config.json", "bili_session.json"):
        (app / "data" / junk).write_text("x", encoding="utf-8")
    (app / "data" / "logs" / "app.log").write_text("日志", encoding="utf-8")

    build_win._post(app)
    assert list((app / "data").glob("*.db")) == []
    assert not (app / "data" / "config.json").exists()
    assert not (app / "data" / "logs" / "app.log").exists()
