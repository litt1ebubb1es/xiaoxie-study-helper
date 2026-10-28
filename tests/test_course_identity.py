"""课程身份：「是不是同一门课」怎么判。

背景：`courses` 表最初只按**标题字符串全等**去重，于是
- 多打一个空格、括号全角/半角、「【完整版】」前缀 → 同一门课分裂成多门；
- 本地导入写死「本地导入课程」→ 不相干的课被合并成一门。

现在按 `DB.title_key()`（NFKC + 去空白/标点/首尾标注）判同一门，并加唯一索引。
这里固化四类"看起来不同、其实是同一门"的写法，防止以后改回去。

注意本文件只测**判定与迁移**，不测界面弹窗（那部分是 main_window 里的
`_ask_course_name`，属于交互，靠 run_app 的手工清单验）。
"""
from __future__ import annotations

import sqlite3

import pytest

from dsh_app.db import DB, title_key


@pytest.fixture()
def db(tmp_path):
    d = DB(tmp_path / "app.db")
    yield d
    d.close()


def test_title_key_folds_spacing_and_width():
    base = title_key("概率论与数理统计")
    # 前后空格、中间空格、全角空格
    assert title_key("  概率论与数理统计  ") == base
    assert title_key("概率论 与 数理统计") == base
    assert title_key("概率论\u3000与数理统计") == base


def test_title_key_folds_brackets_and_marks():
    base = title_key("概率论与数理统计")
    # 上/下册这种是真有区别的，不该折成同一门（见下一个用例）；
    # 这里折的是"同一门课的标注性写法"
    assert title_key("【完整版】概率论与数理统计") == base
    assert title_key("【合集】【完整版】概率论与数理统计") == base
    assert title_key("《概率论与数理统计》") == base
    assert title_key("概率论与数理统计（全36讲）") == base


def test_title_key_keeps_real_difference():
    a = title_key("概率论与数理统计(上)")
    b = title_key("概率论与数理统计（下）")
    assert a != b                      # 上册下册是两门，不能折
    assert title_key("概率统计") not in (a, b)   # 简称不做猜测，交给人工合并


def test_title_key_never_collapses_blank_titles():
    # 全是标点/空白的名字不该塌成同一门课
    assert title_key("（）") != title_key("【】")
    assert title_key("") != title_key("   ")


def test_ensure_course_reuses_one_course(db):
    ids = {
        db.ensure_course("概率论与数理统计"),
        db.ensure_course("概率论与数理统计 "),
        db.ensure_course("【完整版】概率论与数理统计"),
        db.ensure_course("《概率论与数理统计》"),
    }
    assert len(ids) == 1
    assert db.query("SELECT COUNT(*) c FROM courses")[0]["c"] == 1


def test_ensure_course_keeps_upstream_and_downstream(db):
    up = db.ensure_course("概率论与数理统计（上）")
    down = db.ensure_course("概率论与数理统计（下）")
    assert up != down


def test_ensure_course_keeps_first_title(db):
    """复用已存在的课时不要改它的显示名，只补空学科。"""
    cid = db.ensure_course("线性代数", subject="")
    db.ensure_course("线性代数 ", subject="数学")
    row = db.get_course(cid)
    assert row["title"] == "线性代数"
    assert row["subject"] == "数学"


def test_find_course_does_not_create(db):
    assert db.find_course("还没有的课") is None
    assert db.query("SELECT COUNT(*) c FROM courses")[0]["c"] == 0
    cid = db.ensure_course("还没有的课")
    assert db.find_course(" 【还没有的课】 ") == cid


def test_session_numbering_is_per_course(db):
    """节号按课程各自算——录制弹窗的默认标题就靠它。"""
    a = db.ensure_course("线性代数")
    b = db.ensure_course("毛概")
    db.create_session(a, "第 1 节")
    db.create_session(a, "第 2 节")
    db.create_session(b, "第 1 讲")
    assert db.count_sessions(a) == 2
    assert db.count_sessions(b) == 1
    assert db.count_sessions(db.ensure_course("还没录过")) == 0


def test_list_courses_prefers_recent(db):
    old = db.ensure_course("老课")
    new = db.ensure_course("新课")
    sid_old = db.create_session(old, "第 1 节")
    db.execute("UPDATE sessions SET started_at=100 WHERE id=?", (sid_old,))
    sid_new = db.create_session(new, "第 1 节")
    db.execute("UPDATE sessions SET started_at=900 WHERE id=?", (sid_new,))
    rows = db.list_courses()
    assert [r["title"] for r in rows][:2] == ["新课", "老课"]
    assert rows[0]["n_sessions"] == 1


def test_migration_backfills_legacy_db(tmp_path):
    """老库没有 title_key 列：打开时要回填，且撞键不丢数据。"""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE courses (id TEXT PRIMARY KEY, title TEXT NOT NULL,
                              subject TEXT DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE sessions (id TEXT PRIMARY KEY, course_id TEXT NOT NULL,
                               idx INTEGER DEFAULT 0, title TEXT NOT NULL,
                               audio_path TEXT DEFAULT '', started_at REAL NOT NULL,
                               duration REAL DEFAULT 0, source TEXT DEFAULT 'loopback',
                               state TEXT DEFAULT 'recorded', error TEXT DEFAULT '',
                               created_at REAL NOT NULL);
        INSERT INTO courses VALUES ('course_a','概率论与数理统计','',1);
        INSERT INTO courses VALUES ('course_b','概率论与数理统计 ','',2);
        """)
    conn.commit()
    conn.close()

    db = DB(path)                      # 触发回填
    try:
        rows = db.query("SELECT id,title,title_key FROM courses ORDER BY created_at")
        assert len(rows) == 2          # 历史两门都留着，不删数据
        keys = [r["title_key"] for r in rows]
        assert len(set(keys)) == 2     # 撞键的后来者被加了后缀，唯一索引才建得起来
        assert keys[0] == title_key("概率论与数理统计")
        assert keys[1].startswith(keys[0]) and keys[1] != keys[0]
        # 唯一索引真的在
        names = {r["name"] for r in db.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='courses'")}
        assert "ux_courses_title_key" in names
        # 迁移之后新写入仍然按归一化键合并到"老的那门"
        assert db.ensure_course("概率论与数理统计（完整版）") == "course_a"
    finally:
        db.close()