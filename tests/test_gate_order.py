"""硬门槛闸门 + 用知识图谱的边排学习顺序。

两件事必须成立：
1. 掌握度没过门槛时，「下一步」不能说"去做题/开新节"——只能重讲；重讲过了才轮到验证。
2. 重讲用哪个策略不能由模型自选：没证据时补前置，差一半时用类比，差一点时分层复述。
"""
from __future__ import annotations

import pytest

from dsh_app.cognitive import mastery as kpm
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine


@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    st = Settings.load()
    st.llm.base_url = ""
    st.llm.api_key = ""
    e = Engine(db, st)
    sid = e.seed_demo()
    return e, db, sid, db.get_session(sid)["course_id"]


# ------------------------------------------------------------------ 学习顺序
def test_learning_order_puts_root_concepts_first(eng):
    e, db, _, cid = eng
    order = e.learning_order(cid)
    assert order, "有知识点就该排得出顺序"
    first = order[0]
    assert first["prereqs"] == [] or first["blocked_by"] == []


def test_prerequisites_are_strictly_earlier(eng):
    """互为前置会让顺序失去意义：必须按"首次出现时间"严格先后。"""
    e, db, _, cid = eng
    rows = {r["name"]: r for r in db.get_knowledge_points(cid)}
    for x in e.learning_order(cid):
        for p in x["prereqs"]:
            assert float(rows[p]["first_seen_t"]) <= float(rows[x["name"]]["first_seen_t"])
            assert rows[p]["name"] != rows[x["name"]]["name"]


def test_learning_order_marks_blocked_by_unmastered_prereq(eng):
    e, db, _, cid = eng
    order = e.learning_order(cid)
    blocked = [x for x in order if x["blocked_by"]]
    assert blocked, "全都没掌握时，后面依赖别人的知识点该被标成被前置挡住"
    for x in blocked:
        assert set(x["blocked_by"]) <= set(x["prereqs"])


def test_mastered_prereq_stops_blocking(eng):
    e, db, sid, cid = eng
    order = e.learning_order(cid)
    target = next((x for x in order if x["prereqs"]), None)
    assert target is not None
    # 把全部前置都补到门槛之上
    for name in target["prereqs"]:
        kp = next(r for r in db.get_knowledge_points(cid) if r["name"] == name)
        db.set_kp_mastery(int(kp["id"]), kpm.DEFAULT_GATE, 1)
    after = {x["name"]: x for x in e.learning_order(cid)}[target["name"]]
    assert after["blocked_by"] == []


# ------------------------------------------------------------------ 闸门
def test_next_action_is_replay_not_quiz_while_gate_locked(eng):
    e, db, _, cid = eng
    act = e.next_action(cid)
    assert act["action"] == "replay"
    assert act["locked"] is True
    assert "不要跳到做题" in act["why"]
    assert act["knot_id"] is not None


def test_after_replay_next_action_asks_for_quiz(eng):
    e, db, sid, cid = eng
    first = e.next_action(cid)
    e.reword(first["knot_id"], strategy_key=first["strategy"]["key"])
    second = e.next_action(cid)
    assert second["action"] == "quiz"
    assert "小测" in second["why"]


def test_gate_cleared_leaves_no_locked_points(eng):
    e, db, sid, cid = eng
    for r in db.get_knowledge_points(cid):
        db.set_kp_mastery(int(r["id"]), 1.0, 3)
        db.set_kp_mastery(int(r["id"]), kpm.DEFAULT_GATE, 3)
    assert e.gate_lock(cid) == []
    act = e.next_action(cid)
    assert act["action"] in ("new", "quiz")
    assert act["locked"] is False


@pytest.mark.parametrize("attempts,mastery,expected", [
    (0, 0.0, "prerequisite"),
    (1, 0.2, "analogy"),
    (2, 0.75, "layered"),
    (3, 0.95, "numeric"),
])
def test_forced_strategy_is_decided_by_evidence(eng, attempts, mastery, expected):
    e, *_ = eng
    assert e.forced_strategy("随便一个术语", attempts, mastery)["key"] == expected


def test_reword_uses_gate_strategy_over_model_recommendation(eng):
    """文档段落在讲"证明"时会推荐 reverse，但门槛锁死时该用闸门的策略。"""
    e, db, sid, cid = eng
    kid = e.next_action(cid)["knot_id"]
    res = e.reword(kid)                       # 不传 strategy_key
    assert res.strategy == "prerequisite"
    assert res.strategy_name


def test_gate_strategy_disappears_after_mastery(eng):
    e, db, sid, cid = eng
    kid = e.next_action(cid)["knot_id"]
    rows = db.query("SELECT kp.id, kp.name FROM knot_points kx "
                    "JOIN knowledge_points kp ON kp.id=kx.kp_id WHERE kx.knot_id=?",
                    (int(kid),))
    assert rows, "回看点应该挂着知识点"
    for kp in rows:
        db.set_kp_mastery(int(kp["id"]), 0.97, 3)
    assert e._gate_strategy_for_knot(kid) is None


def test_review_queue_marks_locked(eng):
    e, db, _, cid = eng
    kp = db.get_knowledge_points(cid)[0]
    e.record_kp_attempt(int(kp["id"]), 0.0, source="self_test")
    q = {x["name"]: x for x in e.review_queue(cid, limit=50)}
    assert q[kp["name"]]["locked"] is True


def test_next_action_never_proposes_new_session_while_gate_locked(eng):
    e, db, _, cid = eng
    for _ in range(6):
        act = e.next_action(cid)
        assert act["action"] != "new", "还有没过门槛的知识点时不该建议开新节"

def test_retired_point_never_blocks_the_gate(eng):
    """有历史证据但已经没有回看点的旧知识点 → 退役，不能挡在"下一步"前面。

    「必要条件」就是这么来的：旧词表抽的、测过一次，新规则下再没有任何回看点引用它。
    删了不诚实（证据是真的），留着更糟——没有回看点就没法重讲、没法出题，
    门槛会永远锁在一个不存在的东西上。
    """
    e, db, sid, cid = eng
    kp_id = db.upsert_knowledge_point(cid, "必要条件", 0.0, sid)
    e.record_kp_attempt(int(kp_id), 1.0, source="confirm_useful")
    assert "必要条件" in {r["name"] for r in db.get_knowledge_points(cid,
                                                             include_retired=True)}

    retired = db.retire_orphan_points(cid)
    assert "必要条件" in retired
    assert db.count_retired_points(cid) == 1
    # 退役的不再出现在"该学什么"里
    assert "必要条件" not in {r["name"] for r in db.get_knowledge_points(cid)}
    act = e.next_action(cid)
    assert act["action"] == "replay"
    assert (act.get("kp") or {}).get("name") != "必要条件"
    # 证据没被删：记录还在，只是标了 retired
    assert db.query("SELECT * FROM knowledge_points WHERE name='必要条件' AND retired=1")


def test_orphan_without_attempts_is_deleted_not_retired(eng):
    e, db, sid, cid = eng
    n0 = len(db.get_knowledge_points(cid))
    db.upsert_knowledge_point(cid, "边界条件", 0.0, sid)
    pruned = db.prune_orphan_points(cid)
    assert "边界条件" in pruned
    assert db.count_retired_points(cid) == 0
    assert len(db.get_knowledge_points(cid)) == n0
