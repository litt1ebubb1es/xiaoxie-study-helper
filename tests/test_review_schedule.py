"""遗忘曲线与复习队列：掌握度会过期，这件事必须有排程才有意义。"""
from __future__ import annotations

import time

import pytest

from dsh_app.cognitive import mastery as kpm
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine


@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    e = Engine(db, Settings.load())
    sid = e.seed_demo()
    return e, sid, str(db.get_session(sid)["course_id"])


# ---------------------------------------------------------------- 曲线本身
def test_retention_is_one_at_zero_and_target_at_one_stability():
    assert kpm.retrievability(0, 1.0) == pytest.approx(1.0)
    # 这组常数（FSRS 默认）的好处：S 天时留存正好 90%，S 可直接当间隔用
    assert kpm.retrievability(1.0, 1.0) == pytest.approx(0.9, abs=1e-6)
    assert kpm.retrievability(10, 10.0) == pytest.approx(0.9, abs=1e-6)
    assert kpm.retrievability(3, 1.0) < kpm.retrievability(1, 1.0)


def test_due_interval_equals_stability_and_is_bounded():
    assert kpm.due_in_days(1.0) == pytest.approx(1.0, abs=1e-6)
    assert kpm.due_in_days(30.0) == pytest.approx(30.0, abs=1e-6)
    assert kpm.due_in_days(0) == pytest.approx(kpm.MIN_STABILITY_DAYS)


def test_stability_grows_on_success_and_shrinks_on_failure():
    grew = kpm.next_stability(5.0, True, elapsed_days=5.0)
    assert grew > 5.0
    # 同一次会话里连着答对也要涨一点，否则会被误读成"白答了"
    assert kpm.next_stability(5.0, True, elapsed_days=0.0) > 5.0
    dropped = kpm.next_stability(5.0, False)
    assert dropped < 5.0
    assert dropped >= kpm.MIN_STABILITY_DAYS


def test_stability_is_capped():
    s = 1.0
    for _ in range(50):
        s = kpm.next_stability(s, True, elapsed_days=0.0)
    assert s <= kpm.MAX_STABILITY_DAYS


def test_review_due_fires_on_either_condition():
    assert kpm.review_due(0.5, 10.0, 1.0)          # 留存不够
    assert kpm.review_due(0.95, 1.0, 2.0)          # 过了到期时间
    assert not kpm.review_due(0.95, 10.0, 1.0)


# ---------------------------------------------------------------- 落到数据上
def test_answering_writes_stability_and_due_date(eng):
    e, sid, cid = eng
    kp = e.db.get_knowledge_points(cid)[0]
    res = e.record_kp_attempt(int(kp["id"]), 1.0, source="quiz")
    assert res["stability_days"] >= kpm.MIN_STABILITY_DAYS
    row = e.db.get_kp(int(kp["id"]))
    assert float(row["stability"]) > 0
    assert float(row["due_at"]) > time.time()


def test_review_queue_ignores_untested_points(eng):
    e, sid, cid = eng
    assert e.review_queue(cid) == []


def test_review_queue_marks_overdue_after_time_passes(eng):
    e, sid, cid = eng
    kp_id = int(e.db.get_knowledge_points(cid)[0]["id"])
    e.record_kp_attempt(kp_id, 1.0, source="quiz")
    fresh = {q["id"]: q for q in e.review_queue(cid)}[kp_id]
    assert not fresh["overdue"]

    past = time.time() - 40 * 86400      # 假装 40 天没碰
    e.db.execute("UPDATE knowledge_points SET last_evidence_at=?, due_at=? WHERE id=?",
                 (past, past, kp_id))
    stale = {q["id"]: q for q in e.review_queue(cid)}[kp_id]
    assert stale["overdue"]
    assert stale["retention"] < kpm.TARGET_RETENTION
    assert stale["elapsed_h"] > 24 * 39


def test_review_queue_sorts_overdue_first(eng):
    e, sid, cid = eng
    kps = e.db.get_knowledge_points(cid)
    for kp in kps[:2]:
        e.record_kp_attempt(int(kp["id"]), 1.0, source="quiz")
    past = time.time() - 40 * 86400
    e.db.execute("UPDATE knowledge_points SET last_evidence_at=?, due_at=? WHERE id=?",
                 (past, past, int(kps[1]["id"])))
    q = e.review_queue(cid)
    assert q[0]["id"] == int(kps[1]["id"])
    assert q[0]["overdue"]


def test_next_action_prefers_overdue_then_weak_then_new(eng):
    """闸门加进来之后，顺序是：闸门锁死 → 重讲；过了闸门 → 到期复习 → 开新节。"""
    e, sid, cid = eng
    kps = e.db.get_knowledge_points(cid)
    # 全都没测过 → 闸门锁死：先重讲，不许开新节
    first = e.next_action(cid)
    assert first["action"] == "replay"
    assert first["locked"] is True

    # 把它补到门槛之上，其余知识点仍然锁着 → 还是重讲（下一次轮到的那个）
    e.record_kp_attempt(int(kps[0]["id"]), 1.0, source="quiz")
    e.record_kp_attempt(int(kps[0]["id"]), 1.0, source="quiz")
    e.record_kp_attempt(int(kps[0]["id"]), 1.0, source="quiz")
    assert float(e.db.get_knowledge_points(cid)[0]["mastery"]) >= kpm.DEFAULT_GATE
    act = e.next_action(cid)
    assert act["locked"] is True, "还有别的知识点没过门槛，闸门仍然锁着"
    assert act["action"] in ("replay", "quiz")

    # 全部补到门槛之上 + 其中一个很久没碰 → 复习优先
    for kp in e.db.get_knowledge_points(cid):
        e.record_kp_attempt(int(kp["id"]), 1.0, source="quiz")
        e.record_kp_attempt(int(kp["id"]), 1.0, source="quiz")
        e.record_kp_attempt(int(kp["id"]), 1.0, source="quiz")
    assert e.gate_lock(cid) == []
    past = time.time() - 60 * 86400
    e.db.execute("UPDATE knowledge_points SET last_evidence_at=?, due_at=? WHERE id=?",
                 (past, past, int(kps[0]["id"])))
    act = e.next_action(cid)
    assert act["action"] == "quiz"
    assert "留存" in act["why"]
    assert act["knot_id"] is not None      # 能直接跳回那个回看点

    # 没有到期、也没有锁着的 → 建议开新节
    future = time.time()
    e.db.execute("UPDATE knowledge_points SET last_evidence_at=?, due_at=?",
                 (future, future + 30 * 86400))
    assert e.next_action(cid)["action"] == "new"


def test_kp_overview_carries_retention_fields(eng):
    e, sid, cid = eng
    kp_id = int(e.db.get_knowledge_points(cid)[0]["id"])
    e.record_kp_attempt(kp_id, 1.0, source="quiz")
    row = {r["id"]: r for r in e.kp_overview(cid)}[kp_id]
    assert row["retention"] is not None
    assert row["overdue"] is False
    assert row["stability_days"] >= kpm.MIN_STABILITY_DAYS
    # 没测过的知识点不该有 retention（界面显示"—"）
    others = [r for r in e.kp_overview(cid) if r["id"] != kp_id]
    assert all(r.get("retention") is None for r in others)