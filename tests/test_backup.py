"""备份与恢复。

学习记录只有一个 sqlite 文件：几十小时的转写、回看点、掌握度、复习排程全在里面。
这类东西的测试重点不是"能拷文件"，而是三个容易出事的细节：

1. 分析线程正在写库时备份，拿到的必须是完整快照（用 sqlite `backup()`，不是拷文件）；
2. 手动备份不能被下一次自动备份轮转掉（"我刚点的那份没了"会让人不敢用）；
3. 恢复前自动留一份保险，而且 Windows 上要先断开连接才能换文件（-wal/-shm 也得清）。
"""
from __future__ import annotations

import pytest

from dsh_app import maintenance
from dsh_app.db import DB


@pytest.fixture()
def env(tmp_path, monkeypatch):
    # 备份目录也跟着 tmp 走，别污染真实 data/backups
    monkeypatch.setattr(maintenance, "BACKUP_DIR", tmp_path / "backups")
    db = DB(tmp_path / "app.db")
    cid = db.ensure_course("高等数学")
    sid = db.create_session(cid, "第 1 讲 概率密度", "")
    db.replace_segments(sid, [{"idx": 0, "t_start": 0.0, "t_end": 5.0,
                               "text": "概率密度函数", "confidence": 0.9, "source": "asr"}])
    return db, sid


def test_backup_creates_a_readable_snapshot(env):
    db, _sid = env
    dest = maintenance.backup_db(db, tag="manual")
    assert dest.exists() and dest.stat().st_size > 0
    other = DB(dest)                     # 快照本身是可用的库，不是坏文件
    try:
        assert len(other.list_sessions()) == 1
    finally:
        other.close()


def test_backup_filename_carries_time_and_tag(env):
    db, _sid = env
    dest = maintenance.backup_db(db, tag="manual", now=1700000000.0)
    assert dest.name.startswith("app-manual-")
    assert dest.name.endswith(".db")


def test_two_backups_in_the_same_second_do_not_overwrite_each_other(env):
    db, _sid = env
    a = maintenance.backup_db(db, tag="manual", now=1700000000.0)
    b = maintenance.backup_db(db, tag="manual", now=1700000000.0)
    assert a != b and a.exists() and b.exists()


def test_auto_backups_rotate_but_manual_ones_survive(env):
    db, _sid = env
    keep = maintenance.backup_db(db, tag="manual")
    for i in range(5):
        maintenance.backup_db(db, keep=3, now=1700000000.0 + i)
    autos = [x for x in maintenance.list_backups() if "-auto-" in x["name"]]
    assert len(autos) == 3, "自动备份只留 keep 份"
    assert keep.exists(), "手动备份不参与轮转"


def test_restore_brings_old_state_back_and_keeps_a_safety_copy(env):
    db, sid = env
    snap = maintenance.backup_db(db, tag="manual")
    db.update_session(sid, title="被误改的名字")
    assert db.get_session(sid)["title"] == "被误改的名字"

    res = maintenance.restore_backup(db, snap)
    assert db.get_session(sid)["title"].startswith("第 1 讲")
    # 恢复前的状态没丢：保险备份里还是"误改"后的样子
    safety = maintenance.backup_dir() / res["safety"]
    assert safety.exists()
    old = DB(safety)
    try:
        assert old.list_sessions()[0]["title"] == "被误改的名字"
    finally:
        old.close()


def test_restore_survives_wal_sidecars(tmp_path, monkeypatch, env):
    """Windows 上换库文件会被占用；同时残留的 -wal 会把新内容盖回旧的。

    库是 WAL 模式，重新打开后 sqlite 自己会再建一个 -wal —— 所以判断标准不是
    "文件不存在"，而是"里面不能再有旧内容"，且恢复后的数据确实是对的。
    """
    db, sid = env
    snap = maintenance.backup_db(db, tag="manual")
    wal = tmp_path / "app.db-wal"
    wal.write_bytes(b"stale wal")
    maintenance.restore_backup(db, snap)
    if wal.exists():
        assert b"stale wal" not in wal.read_bytes(), "旧 WAL 没清掉，等于没恢复"
    assert db.get_session(sid) is not None       # 连接已重新打开，能继续用
    assert len(db.get_segments(sid)) == 1


def test_restore_missing_file_says_so(env, tmp_path):
    db, _sid = env
    with pytest.raises(FileNotFoundError):
        maintenance.restore_backup(db, tmp_path / "nope.db")


def test_last_backup_time_is_zero_before_the_first_one(env):
    db, _sid = env
    assert maintenance.last_backup_time() == 0.0
    maintenance.backup_db(db, tag="manual")
    assert maintenance.last_backup_time() > 0


# -- 音频清理 --------------------------------------------------------------
def _touch_audio(tmp_path, name="a.wav", size=2048):
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return p


def test_sweep_deletes_old_audio_but_keeps_the_lesson(env, tmp_path, monkeypatch):
    db, sid = env
    wav = _touch_audio(tmp_path)
    import time as _t

    db.update_session(sid, audio_path=str(wav), created_at=_t.time() - 40 * 86400)
    settings = __import__("dsh_app.config", fromlist=["Settings"]).Settings.load()
    settings.capture.keep_audio_days = 7
    r = maintenance.sweep_audio(db, settings)
    assert r["deleted"] == 1 and r["freed"] > 0
    assert not wav.exists()
    assert db.get_session(sid)["audio_path"] == ""
    # 课还在，转写也还在：删的只是能重放的声音
    assert db.get_session(sid) is not None
    assert len(db.get_segments(sid)) == 1


def test_sweep_leaves_recent_audio_alone(env, tmp_path):
    db, sid = env
    wav = _touch_audio(tmp_path)
    db.update_session(sid, audio_path=str(wav))
    settings = __import__("dsh_app.config", fromlist=["Settings"]).Settings.load()
    settings.capture.keep_audio_days = 7
    assert maintenance.sweep_audio(db, settings)["deleted"] == 0
    assert wav.exists()


def test_keep_days_zero_means_never_delete(env, tmp_path):
    db, sid = env
    wav = _touch_audio(tmp_path)
    import time as _t

    db.update_session(sid, audio_path=str(wav), created_at=_t.time() - 400 * 86400)
    settings = __import__("dsh_app.config", fromlist=["Settings"]).Settings.load()
    settings.capture.keep_audio_days = 0
    r = maintenance.sweep_audio(db, settings)
    assert r["skipped"] is True and r["deleted"] == 0
    assert wav.exists()


def test_audio_usage_counts_files_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, "AUDIO_DIR", tmp_path)
    _touch_audio(tmp_path, "a.wav", 1000)
    _touch_audio(tmp_path, "b.flac", 3000)
    u = maintenance.audio_usage(tmp_path)
    assert u["files"] == 2 and u["bytes"] == 4000
    assert "KB" in u["text"]