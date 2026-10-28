"""复习提醒：到期的知识点要能被找出来，而且一天只提醒一次。

两点最容易做错：
1. `due_at` 到期的判断写错，复习提醒就永远不响（或每次都响）；
2. 提醒没有节流——每半小时跳一次就是骚扰，用户会去关通知，功能等于没有。
"""
from __future__ import annotations

import time

import pytest

from dsh_app.db import DB


@pytest.fixture()
def db(tmp_path):
    d = DB(tmp_path / "review.db")
    cid = d.ensure_course("高等数学")
    return d, cid


def _kp(db, cid, name, *, due_in, mastery=0.1, gate=0.9, retired=0, sid=""):
    db.upsert_knowledge_point(cid, name, 0.0, sid or "sess_x", origin="rule")
    rows = db.query("SELECT id FROM knowledge_points WHERE course_id=? AND name=?", (cid, name))
    kid = int(rows[0]["id"])
    db.execute("UPDATE knowledge_points SET due_at=?, mastery=?, gate=?, retired=? WHERE id=?",
               (time.time() + due_in, mastery, gate, retired, kid))
    return kid


def test_due_points_only_returns_overdue_ones(db):
    d, cid = db
    _kp(d, cid, "概率密度", due_in=-3600)            # 一小时前就该复习
    _kp(d, cid, "分布函数", due_in=+3600)            # 还没到
    _kp(d, cid, "期望", due_in=-3600, mastery=0.95)  # 已掌握，别再提醒
    _kp(d, cid, "方差", due_in=-3600, retired=1)     # 退役了
    names = [r["name"] for r in d.due_points()]
    assert names == ["概率密度"]


def test_due_points_can_be_scoped_to_a_course(db):
    d, cid = db
    other = d.ensure_course("线性代数")
    _kp(d, cid, "概率密度", due_in=-60)
    _kp(d, other, "向量空间", due_in=-60)
    assert [r["name"] for r in d.due_points(cid)] == ["概率密度"]
    assert len(d.due_points()) == 2


def test_points_without_schedule_are_not_due(db):
    d, cid = db
    _kp(d, cid, "未排程的点", due_in=0)
    d.execute("UPDATE knowledge_points SET due_at=0 WHERE name='未排程的点'")
    assert d.due_points() == []


# -- 界面侧：一天只提醒一次 ------------------------------------------------
@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    from dsh_app.config import Settings
    from dsh_app.ui import main_window as mw

    state = tmp_path / "ui_state.json"
    monkeypatch.setattr(mw.MainWindow, "_ui_state_path", lambda self: state)
    db = DB(tmp_path / "ui.db")
    w = mw.MainWindow(Settings.load(), db)
    yield w, db, state
    w.close()


def test_notice_fires_once_a_day(win, qapp):
    w, db, state = win
    cid = db.ensure_course("高等数学")
    _kp(db, cid, "概率密度", due_in=-600)
    assert db.due_points(), "先确认确实有到期的点"

    w._check_review_notice()
    qapp.processEvents()
    assert state.exists(), "提醒过后要在状态文件里记一天"
    first = state.read_text(encoding="utf-8")
    assert "notice_last" in first

    w._check_review_notice()             # 同一天再来一次：不该重复打扰
    qapp.processEvents()
    assert state.read_text(encoding="utf-8") == first


def test_no_notice_when_nothing_is_due(win, qapp):
    w, db, state = win
    w._check_review_notice()
    qapp.processEvents()
    assert not state.exists()


def test_state_file_keeps_other_keys(win, qapp, tmp_path):
    """状态文件是和小窗共用的：主窗口写自己的键时不能把别人的清掉。"""
    w, db, state = win
    state.write_text('{"chatwin": {"x": 1, "y": 2}}', encoding="utf-8")
    w._save_ui_state({"backup_last": "2026-01-01"})
    import json

    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["chatwin"] == {"x": 1, "y": 2}
    assert data["backup_last"] == "2026-01-01"


def test_daily_backup_runs_once_per_day(win, qapp, tmp_path, monkeypatch):
    from dsh_app import maintenance

    w, db, state = win
    calls = []
    monkeypatch.setattr(maintenance, "backup_db",
                        lambda d, keep=7, tag="auto", now=None: calls.append(tag))
    monkeypatch.setattr(maintenance, "last_backup_time", lambda: 0.0)   # 当成"从没备份过"
    w._maybe_daily_backup()
    w._maybe_daily_backup()              # 同一天第二次：不该再备份
    assert calls == ["auto"]


def test_tray_is_optional_not_required(win, qapp):
    """离屏/无托盘环境必须照常启动：托盘是加分项，不是启动条件。"""
    w, _db, _state = win
    assert hasattr(w, "_tray")
    w._check_review_notice()             # 没有托盘也不该抛
    qapp.processEvents()

def test_daily_backup_skips_when_a_recent_backup_exists(win, qapp, monkeypatch):
    """状态文件被别的写入方清掉时，靠"最新备份的时间"兜住，别每开一次就备一次。"""
    from dsh_app import maintenance

    w, _db, state = win
    calls = []
    monkeypatch.setattr(maintenance, "backup_db",
                        lambda d, keep=7, tag="auto", now=None: calls.append(tag))
    monkeypatch.setattr(maintenance, "last_backup_time", lambda: __import__("time").time())
    state.write_text("{}", encoding="utf-8")     # 模拟状态键丢了
    w._maybe_daily_backup()
    assert calls == []
