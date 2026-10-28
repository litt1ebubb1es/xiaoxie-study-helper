"""回看点纠错：确实卡住 / 不算回看点 / 撤销。

设计要求里最重要的一条：**误判不能删行**。删了以后就没有"算法说卡住、用户说不是"
这条样本，阈值永远调不准。所以这里的断言都在验"状态变了、数据还在"，
以及"用户判过的位置，重新分析不该复活成待解决"。
"""
from __future__ import annotations

import pytest

from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine


@pytest.fixture()
def env(tmp_path):
    db = DB(tmp_path / "knots.db")
    settings = Settings.load()
    cid = db.ensure_course("测试课")
    sid = db.create_session(cid, "第 1 讲", "")
    db.replace_segments(sid, [
        {"idx": 0, "t_start": 0.0, "t_end": 20.0, "text": "我们今天讲概率密度函数。",
         "confidence": 0.9, "source": "asr"},
        {"idx": 1, "t_start": 20.0, "t_end": 40.0, "text": "它的定义是单位区间上的概率。",
         "confidence": 0.9, "source": "asr"},
        {"idx": 2, "t_start": 40.0, "t_end": 60.0, "text": "也就是说∫f=1。",
         "confidence": 0.9, "source": "asr"},
    ])
    eng = Engine(db, settings)
    return db, eng, sid


def _seed_knots(db, sid):
    """直接写两条回看点：不跑分析，测的是纠错逻辑本身。"""

    class K:
        def __init__(self, t0, t1, score):
            self.t_start, self.t_end = t0, t1
            self.level, self.quadrant, self.grade = "explicit", "high_low", "strong"
            self.score, self.l_score, self.b_score = score, 0.8, 0.7
            self.evidence, self.excerpt = "在同一句上反复回溯", "概率密度函数的定义"
            self.evidence_detail = {"why": "回溯"}

    return db.replace_knots(sid, [K(15.0, 25.0, 0.8), K(45.0, 55.0, 0.6)])


def test_dismiss_keeps_the_row_and_flips_status(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.dismiss_knot(ids[0])
    row = db.get_knot(ids[0])
    assert row is not None, "删行是最糟的做法：校准样本就没了"
    assert row["status"] == "dismissed"
    # 另一条不受影响
    assert db.get_knot(ids[1])["status"] == "pending"


def test_dismiss_writes_a_feedback_record(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.dismiss_knot(ids[0], note="老师只是在念例题")
    fb = db.get_feedback(sid)
    assert any(r["action"] == "dismissed" for r in fb)
    assert any("例题" in (r["note"] or "") for r in fb)


def test_restore_brings_it_back(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.dismiss_knot(ids[0])
    eng.restore_knot(ids[0])
    assert db.get_knot(ids[0])["status"] == "pending"
    assert ids[0] not in db.dismissed_knot_ids(sid)


def test_confirm_records_positive_feedback_but_leaves_status(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.confirm_knot(ids[0])
    assert db.get_knot(ids[0])["status"] == "pending"   # 还是待解决，不是已掌握
    assert any(r["action"] == "confirm_useful" for r in db.get_feedback(sid))


def test_dismissed_ids_are_queryable_per_session(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    cid2 = db.ensure_course("另一门课")
    sid2 = db.create_session(cid2, "第 1 讲", "")
    eng.dismiss_knot(ids[0])
    assert db.dismissed_knot_ids(sid) == {ids[0]}
    assert db.dismissed_knot_ids(sid2) == set()


def test_reanalysis_does_not_resurrect_a_dismissed_knot(env, monkeypatch):
    """用户判过"这里不算"，重新分析又算出来同一段——不能当成新回看点推给他。

    这里把分析结果钉死：合成转写跑真算法不保证出回看点，而这条测试要验的是
    "重新分析后如何保住用户的判断"，不是算法本身。
    """
    import dsh_app.pipeline as pipe
    from dsh_app.cognitive.features import Window
    from dsh_app.cognitive.scoring import AnalysisResult, Knot

    db, eng, sid = env

    def fake_analyze(segments, events, cfg, **kw):
        win = Window(idx=0, t_start=20.0, t_end=30.0, text="概率密度函数的定义是单位区间上的概率",
                     l_score=0.8, b_score=0.7, grade="strong", quadrant="high_low")
        knot = Knot(t_start=20.0, t_end=30.0, level="explicit", quadrant="high_low",
                    grade="strong", score=0.8, l_score=0.8, b_score=0.7,
                    evidence="在同一句上反复回溯", excerpt="概率密度函数的定义")
        return AnalysisResult(windows=[win], knots=[knot])

    monkeypatch.setattr(pipe, "analyze", fake_analyze)

    eng.analyze_session(sid)
    first = db.get_knots(sid)
    assert len(first) == 1
    eng.dismiss_knot(int(first[0].id))

    eng.analyze_session(sid)              # 会先 DELETE 再 INSERT
    again = db.get_knots(sid)
    assert len(again) == 1
    assert again[0].status == "dismissed", "用户判过的位置不该被重新分析复活"
    assert abs(float(again[0].t_start) - 20.0) < 1e-6


def test_reanalysis_keeps_untouched_knots_pending(env, monkeypatch):
    """别矫枉过正：没被判过的位置，重新分析后仍然是待解决。"""
    import dsh_app.pipeline as pipe
    from dsh_app.cognitive.features import Window
    from dsh_app.cognitive.scoring import AnalysisResult, Knot

    db, eng, sid = env

    def fake_analyze(segments, events, cfg, **kw):
        wins, knots = [], []
        for i, (a, b) in enumerate(((10.0, 20.0), (40.0, 50.0))):
            wins.append(Window(idx=i, t_start=a, t_end=b, text=f"第{i}段讲解内容在这里",
                               l_score=0.7, b_score=0.6, grade="strong", quadrant="high_low"))
            knots.append(Knot(t_start=a, t_end=b, level="explicit", quadrant="high_low",
                              grade="strong", score=0.7, l_score=0.7, b_score=0.6,
                              evidence="回溯", excerpt=f"第{i}段"))
        return AnalysisResult(windows=wins, knots=knots)

    monkeypatch.setattr(pipe, "analyze", fake_analyze)
    eng.analyze_session(sid)
    first = sorted(db.get_knots(sid), key=lambda k: k.t_start)
    eng.dismiss_knot(int(first[0].id))

    eng.analyze_session(sid)
    again = sorted(db.get_knots(sid), key=lambda k: k.t_start)
    assert [k.status for k in again] == ["dismissed", "pending"]


def test_active_knots_filter_drops_only_dismissed(env):
    """界面上的"参与统计的回看点"必须是过滤后的：重讲下拉、覆盖率、复习建议都靠它。"""
    from dsh_app.ui.main_window import MainWindow        # noqa: F401 —— 只验证 DB 侧契约
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.dismiss_knot(ids[1])
    bad = db.dismissed_knot_ids(sid)
    active = [k for k in db.get_knots(sid) if int(k.id) not in bad]
    assert [int(k.id) for k in active] == [ids[0]]


def test_get_knots_can_exclude_dismissed(env):
    db, eng, sid = env
    ids = _seed_knots(db, sid)
    eng.dismiss_knot(ids[0])
    only_active = db.get_knots(sid, include_dismissed=False)
    assert ids[0] not in [int(k.id) for k in only_active]
    assert len(db.get_knots(sid)) == 2
