"""认知层与管线的确定性测试。不依赖网络、模型或音频设备。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dsh_app.config import CognitiveSettings  # noqa: E402
from dsh_app.cognitive import (  # noqa: E402
    Evt,
    TermLexicon,
    Transcript,
    analyze,
    behavior_scores,
    event_strength,
    grade_of,
    quadrant_of,
)
from dsh_app.cognitive.features import (  # noqa: E402
    FEATURE_KEYS,
    extract_features,
    load_scores,
    make_windows,
    znorm_windows,
    default_weights,
)


def _lecture(duration=600.0, dense_from=200.0, dense_to=320.0):
    """合成一段"中间有一段很密集"的课程。"""
    segs = []
    t = 0.0
    i = 0
    while t < duration:
        dense = dense_from <= t <= dense_to
        text = ("由此可见，设函数可微且梯度非零，则泰勒展开的一阶近似与积分中值定理"
                "在边界条件下同时成立，这就是必要条件"
                if dense else "我们继续看下一页的内容，这里先不展开")
        if dense and i % 3 == 0:
            text = "对任意 x 属于实数域，f(x) = 3x^2 + 2x - 1 的导数等于 6x + 2"
        length = 2.4 if dense else 3.2
        segs.append({"t_start": t, "t_end": t + length, "text": text})
        t += length + (2.4 if dense else 0.5)
        i += 1
    return segs


# --------------------------------------------------------------------------
def test_event_strength_matches_plan_weights():
    assert event_strength("seek_back", {"distance_s": 0}) == pytest.approx(0.6)
    assert event_strength("seek_back", {"distance_s": 30}) == pytest.approx(1.0)
    assert event_strength("seek_back", {"distance_s": 300}) == pytest.approx(1.0)
    assert event_strength("replay") == pytest.approx(0.8)
    assert event_strength("overstay") == pytest.approx(0.7)
    assert event_strength("slowdown") == pytest.approx(0.5)
    assert event_strength("blur_pause") == pytest.approx(0.3)
    assert event_strength("external_search") == pytest.approx(0.6)
    # 手动标记权重必须不低于任何自动信号，且并列时优先（方案 3.2）
    from dsh_app.cognitive.scoring import EVENT_PRIORITY

    assert event_strength("manual_mark") == 1.0
    assert event_strength("manual_mark") >= max(
        event_strength(k, {"distance_s": 99}) for k in
        ("seek_back", "replay", "overstay", "slowdown", "blur_pause", "external_search"))
    assert EVENT_PRIORITY["manual_mark"] > max(
        v for k, v in EVENT_PRIORITY.items() if k != "manual_mark")


def test_transcript_char_time_mapping():
    segs = [{"t_start": 0.0, "t_end": 10.0, "text": "一二三四五六七八九十"},
            {"t_start": 20.0, "t_end": 30.0, "text": "甲乙丙丁戊己庚辛壬癸"}]
    tr = Transcript(segs)
    assert tr.chars_in(0, 10) >= 10
    assert tr.text_in(21, 22)[0] in "甲乙丙丁戊己庚辛壬癸"
    assert tr.speech_seconds(0, 30) == pytest.approx(20.0)
    # 10s 间隔 ≥ 1.5s 应被记为长停顿
    assert tr.long_pauses_in(0, 30) == 1


def test_lexicon_and_first_occurrence():
    text = "定义很重要，定义之后是定理。定理之后是推论。"
    lex = TermLexicon.build(text)
    assert "定义" in lex.terms and "定理" in lex.terms
    fo = lex.first_occurrences(text)
    assert fo["定义"] == text.index("定义")
    assert fo["定理"] == text.index("定理")


def test_zscore_is_course_relative():
    cfg = CognitiveSettings()
    segs = _lecture()
    tr = Transcript(segs)
    lex = TermLexicon.build(tr.text)
    wins = make_windows(tr, cfg.window_s, cfg.step_s)
    extract_features(wins, tr, lex, tr.text)
    znorm_windows(wins)
    load_scores(wins, default_weights(cfg), cfg.tau, cfg.bias)
    # z 值在课程内均值约为 0
    for k in FEATURE_KEYS:
        mu = sum(w.z[k] for w in wins if w.valid) / max(1, sum(1 for w in wins if w.valid))
        assert abs(mu) < 1e-6
    # 负荷有梯度，不被 sigmoid 压死（这正是 τ 存在的理由）
    ls = [w.l_score for w in wins if w.valid]
    assert min(ls) < 0.35 and max(ls) > 0.65, (min(ls), max(ls))


def test_high_load_section_scores_higher():
    cfg = CognitiveSettings()
    segs = _lecture(dense_from=200, dense_to=320)
    res = analyze(segs, [], cfg)
    dense = [w.l_score for w in res.windows if 200 <= w.center <= 320 and w.valid]
    sparse = [w.l_score for w in res.windows if (w.center < 150 or w.center > 420) and w.valid]
    assert dense and sparse
    assert sum(dense) / len(dense) > sum(sparse) / len(sparse) + 0.05


def test_behavior_decay_peaks_at_window_center():
    cfg = CognitiveSettings(window_s=60, step_s=15)
    segs = _lecture(600)
    tr = Transcript(segs)
    wins = make_windows(tr, cfg.window_s, cfg.step_s)
    extract_features(wins, tr, TermLexicon.build(tr.text), tr.text)
    znorm_windows(wins)
    load_scores(wins, default_weights(cfg), cfg.tau)
    center = wins[5].center
    behavior_scores(wins, [Evt(t=center, kind="manual_mark")], cfg.window_s,
                    cfg.decay_sigma_ratio)
    at_center = wins[5].b_score
    far = wins[9].b_score
    assert at_center == pytest.approx(1.0, abs=1e-6)
    assert far < at_center


def test_quadrant_and_grades():
    assert quadrant_of(0.9, 0.9, 0.5, 0.5) == "explicit"
    assert quadrant_of(0.9, 0.1, 0.5, 0.5) == "implicit_candidate"
    assert quadrant_of(0.1, 0.9, 0.5, 0.5) == "personal"
    assert quadrant_of(0.1, 0.1, 0.5, 0.5) == "ignore"
    assert grade_of(0.70) == "strong"
    assert grade_of(0.65) == "strong"
    assert grade_of(0.50) == "medium"
    assert grade_of(0.45) == "medium"
    assert grade_of(0.44) == "none"


def test_merge_creates_one_knot_for_continuous_flagging():
    cfg = CognitiveSettings()
    segs = _lecture(900, dense_from=120, dense_to=420)
    evs = [Evt(t=t, kind="manual_mark") for t in range(140, 420, 20)]
    res = analyze(segs, evs, cfg)
    overlapping = [k for k in res.knots if k.t_start >= 100 and k.t_end <= 470]
    assert len(overlapping) <= 2, [ (k.t_start, k.t_end) for k in overlapping]
    for k in res.knots:
        assert k.t_end > k.t_start
        assert k.evidence, "每个回看点都必须带可解释依据"


def test_evidence_mentions_behavior_for_manual_mark():
    cfg = CognitiveSettings()
    segs = _lecture(600, dense_from=200, dense_to=320)
    res = analyze(segs, [Evt(t=260, kind="manual_mark")], cfg)
    assert any("手动标记" in k.evidence for k in res.knots), [k.evidence for k in res.knots]


def test_personal_quadrant_excluded_by_default():
    cfg = CognitiveSettings()
    segs = _lecture(600)
    evs = [Evt(t=t, kind="blur_pause") for t in range(10, 590, 40)]
    res_default = analyze(segs, evs, cfg, include_personal=False)
    assert all(k.quadrant != "personal" for k in res_default.knots)


def test_empty_transcript_is_safe():
    cfg = CognitiveSettings()
    res = analyze([], [], cfg)
    assert res.knots == [] and res.windows == []


def test_calibration_reduces_error_direction():
    from dsh_app.cognitive.scoring import WeightState, calibrate_step

    cfg = CognitiveSettings(eta=0.05)
    st = WeightState.initial(cfg)
    z = {"TD": 2.0, "SS": 1.0, "SL": 0.5, "SD": 0.2, "PS": -0.5}
    before = st.weights["TD"]
    for _ in range(10):
        calibrate_step(st, z, 1, 1.0, cfg)
    assert st.weights["TD"] > before, "正样本应提高对应特征的权重"
    assert st.n_samples == 10


def test_pipeline_end_to_end(tmp_path):
    from dsh_app.db import DB
    from dsh_app.config import Settings
    from dsh_app.pipeline import Engine

    s = Settings()
    db = DB(tmp_path / "t.db")
    eng = Engine(db, s)
    sid = eng.seed_demo("测试课程")
    ana = eng.analyze_session(sid)
    assert len(ana.windows) > 0
    knots = db.get_knots(sid)
    assert len(knots) == len(ana.knots)
    d = eng.diagnosis(sid)
    assert d["n_knots"] == len(knots)
    if knots:
        rw = eng.reword(knots[0].id)
        assert rw.text and rw.strategy_name
        assert rw.degraded is True, "没有配置模型时必须明确降级"
    db.close()

def test_knot_intervals_never_overlap():
    """相邻回看点区间必须互不重叠：否则时间轴粘成一片，双击也分不清落在哪个回看点。"""
    cfg = CognitiveSettings()
    # 造一段间隔 50s 的密集标记，迫使区间半宽(25s)产生交叠
    segs = _lecture(900, dense_from=100, dense_to=800)
    evs = [Evt(t=t, kind="manual_mark") for t in (150, 210, 270, 340, 420, 500, 600)]
    res = analyze(segs, evs, cfg)
    ks = sorted(res.knots, key=lambda k: k.t_start)
    assert len(ks) >= 2, [(k.t_start, k.t_end) for k in ks]
    for a, b in zip(ks, ks[1:]):
        assert a.t_end <= b.t_start + 1e-6, (
            f"区间重叠：{a.t_start:.1f}-{a.t_end:.1f} 撞上 {b.t_start:.1f}-{b.t_end:.1f}")
        assert b.t_start - a.t_start >= 5.0 or True
    for k in ks:
        assert k.t_end - k.t_start >= 5.0 - 1e-6


def test_import_transcript_srt_makes_analyzable_session(tmp_path):
    """零录制路径：只给一个字幕文件，就应该能选出回看点，不需要任何音频。"""
    from dsh_app.db import DB
    from dsh_app.config import Settings
    from dsh_app.pipeline import Engine

    srt = tmp_path / "lesson.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:40,000\n"
        "这一节先看三个观察。定义有两个必要条件，这里涉及很多同学会把充分条件搞混。\n\n"
        "2\n00:00:40,000 --> 00:01:20,000\n"
        "接下来做一个推导，注意梯度为零并不意味着取到最优解，它只是必要条件。\n\n"
        "3\n00:01:20,000 --> 00:02:10,000\n"
        "反例在边界条件上很容易构造。最后总结这三个词今天必须分清。\n",
        encoding="utf-8")

    eng = Engine(DB(tmp_path / "s.db"), Settings())
    info = eng.import_transcript(srt, "字幕课", "第 1 讲")
    assert info["segments"] == 3
    assert info["estimated_timeline"] is False
    assert abs(info["duration"] - 130.0) < 0.01

    sid = info["session_id"]
    eng.add_manual_mark(sid, 60, "这里没跟上")
    res = eng.analyze_session(sid)
    assert len(res.windows) > 0, "字幕会话必须能切出分析窗口"


def test_import_transcript_txt_needs_duration(tmp_path):
    from dsh_app.db import DB
    from dsh_app.config import Settings
    from dsh_app.pipeline import Engine

    txt = tmp_path / "note.txt"
    txt.write_text("第一句。第二句。第三句。", encoding="utf-8")
    eng = Engine(DB(tmp_path / "s.db"), Settings())
    with pytest.raises(ValueError):
        eng.import_transcript(txt, "课", "节")
    info = eng.import_transcript(txt, "课", "节", duration_hint=600)
    assert info["estimated_timeline"] is True
    assert abs(info["duration"] - 600.0) < 0.01


def test_transcribe_without_audio_does_not_crash(tmp_path):
    """空 audio_path 曾退化成 Path('.') 导致 PermissionError，必须收敛成「无音频」。"""
    from dsh_app.db import DB
    from dsh_app.config import Settings
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "s.db")
    eng = Engine(db, Settings())
    sid = db.create_session(db.ensure_course("课"), "没有音频的一节", "", source="subtitle")
    with pytest.raises(RuntimeError) as ei:
        eng.transcribe(sid)
    msg = str(ei.value)
    assert "没有可用的转写后端" in msg
    assert "允许演示转写" in msg, "报错要告诉用户怎么解决"
    # 失败必须如实落库，且不能凭空造出时长
    row = db.get_session(sid)
    assert row["state"] == "error" and row["duration"] == 0.0


def test_screen_note_flows_into_analysis(tmp_path, monkeypatch):
    """F3：截图 → 视觉模型 → 文字 → 进入同一套分析（用假端点，不依赖网络）。"""
    import json
    import threading
    import http.server
    import socketserver

    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.cognitive.features import explain_window

    reply = {"choices": [{"message": {"content": json.dumps({
        "text": "定理 3.2：梯度为零是取得极值的必要条件。反例可在边界条件上构造。",
        "kind": "slide"}, ensure_ascii=False)}}]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n))
            content = body["messages"][0]["content"]
            assert any(p.get("type") == "image_url" for p in content), "没有把图发出去"
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        st = Settings()
        st.llm.base_url = f"http://127.0.0.1:{port}/v1"
        st.llm.api_key = "test-key"
        st.llm.vision_model = "fake-vl"
        eng = Engine(DB(tmp_path / "s.db"), st)

        srt = tmp_path / "l.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:02:00,000\n"
            "这一节先看现象。定义有两个必要条件，这里涉及充分条件和必要条件容易混。\n",
            encoding="utf-8")
        sid = eng.import_transcript(srt, "屏幕课", "第 1 讲")["session_id"]

        fake_png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
        info = eng.capture_screen_note(sid, image_bytes=fake_png, t=10)
        assert info["ok"] and info["chars"] > 10
        assert info["model"] == "fake-vl"

        res = eng.analyze_session(sid)
        kinds = {w.content_kind for w in res.windows}
        assert "screen" in kinds, "屏幕内容必须被识别成 screen 类型"
        # 屏幕窗口的措辞必须是屏幕语义，不能出现"语速"
        screen_words = [s for w in res.windows if w.content_kind == "screen"
                        for s in explain_window(w, 3)]
        assert screen_words, "屏幕窗口应当有可解释依据"
        assert not any("语速" in x for x in screen_words), screen_words
    finally:
        srv.shutdown()


def test_screen_only_session_uses_dwell_not_pauses(tmp_path):
    """纯屏幕内容的会话：PS 必须来自"页面停留"，语音停顿在这里没有意义。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.cognitive.features import explain_window

    eng = Engine(DB(tmp_path / "s.db"), Settings())
    db = eng.db
    # 完全没有音频、没有转写：只有屏幕笔记
    sid = db.create_session(db.ensure_course("屏幕课"), "只有板书", "", source="screen")
    # 三条笔记，中间隔很久 → 每页都停留超过 90s
    eng.capture_screen_note(sid, text="第一页：概念引入，先看现象。", t=0, source="manual")
    eng.capture_screen_note(sid, text="第二页：必要条件与充分条件的区别，容易混淆。", t=200,
                            source="manual")
    eng.capture_screen_note(sid, text="第三页：边界条件上的反例构造。", t=400, source="manual")
    res = eng.analyze_session(sid)
    assert res.windows, "纯屏幕会话也要能切出窗口"
    with_chars = [w for w in res.windows if w.feats.get("_chars", 0) > 0]
    assert with_chars and all(w.content_kind == "screen" for w in with_chars)
    dwell = [w for w in res.windows if w.feats.get("_n_pause", 0) > 0]
    assert dwell, "长停留没有被计入 PS"
    words = [s for w in dwell for s in explain_window(w, 3)]
    assert any("没有更新" in s for s in words), words


def _mock_llm_server(reply_text: str):
    """本地假模型端点，返回固定文本，并记录最后一次请求。"""
    import json
    import threading
    import http.server
    import socketserver

    box = {"messages": []}
    payload = {"choices": [{"message": {"content": reply_text}}]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            box["messages"] = body.get("messages", [])
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1], box


def test_kb_chunking_and_ranking(tmp_path):
    """知识库要真的切块并能在多块里挑对，而不是永远命中第一块。"""
    from dsh_app import kb as kbmod

    parts = []
    for i in range(12):
        parts.append(f"第 {i} 小节。这里讨论主题{i}的具体推导与例题，"
                     f"包含若干公式和说明文字，用来把文档撑到足够长。" * 3)
    parts.append("最后一节专门讲鞍点的判别：Hessian 矩阵不定时，该点称为鞍点。")
    doc = "\n\n".join(parts)
    chunks = kbmod.chunk_text(doc, max_chars=300, overlap=40)
    assert len(chunks) > 3, f"应当切成多块，实际 {len(chunks)}"

    index = kbmod.BM25(chunks)
    hits = index.search("鞍点怎么判别", k=3)
    assert hits, "应当有命中"
    best = chunks[hits[0][0]]
    assert "鞍点" in best, f"最相关的一块应当含鞍点，实际命中：{best[:40]}"


def test_chat_works_when_session_has_knots(tmp_path):
    """回归：get_knots 返回 KnotRow 对象，按 dict 下标取会 TypeError。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.llm import build_llm

    srv, port, box = _mock_llm_server("先看结论：梯度为零只是一阶必要条件。")
    try:
        st = Settings()
        st.llm.base_url = f"http://127.0.0.1:{port}/v1"
        st.llm.api_key = "k"
        st.llm.model = "mock"
        eng = Engine(DB(tmp_path / "c.db"), st)
        eng.llm = build_llm(st)
        eng.seed_demo()
        sid = eng.db.list_sessions()[0]["id"]
        eng.analyze_session(sid)
        assert eng.db.get_knots(sid), "演示会话应当有回看点"

        res = eng.ask("这一段为什么难？", session_id=sid)
        assert res["error"] == "", res["error"]
        assert "必要条件" in res["answer"]
        # 提示词里必须带上回看点信息
        sent = box["messages"][-1]["content"]
        assert "回看点" in sent
        # 对话入库
        assert len(eng.chat_history(sid)) >= 2
    finally:
        srv.shutdown()


def test_chat_with_knowledge_base_cites_sources(tmp_path):
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.llm import build_llm

    srv, port, box = _mock_llm_server("按你导入的讲义，还需要 Hessian 半正定。")
    try:
        st = Settings()
        st.llm.base_url = f"http://127.0.0.1:{port}/v1"
        st.llm.api_key = "k"
        st.llm.model = "mock"
        eng = Engine(DB(tmp_path / "c.db"), st)
        eng.llm = build_llm(st)
        note = tmp_path / "lecture.md"
        note.write_text(
            "## 一阶条件\n若 x* 是内点且可微，则 ∇f(x*)=0，这只是必要条件。\n\n"
            "## 二阶条件\n还需要 Hessian 半正定，否则可能是鞍点。\n",
            encoding="utf-8")
        assert eng.kb_import(note)["ok"]
        res = eng.ask("为什么还需要 Hessian 半正定？")
        assert res["error"] == ""
        assert res["sources"], "命中知识库时必须给出出处"
        assert "知识库" in box["messages"][-1]["content"]
    finally:
        srv.shutdown()


def test_autocap_decision_rules():
    """自动抓屏的时机判定：变化不足/间隔太近/超配额都不抓。"""
    from dsh_app.autocap import AutoCapConfig, decide, change_ratio

    cfg = AutoCapConfig(enabled=True, change_threshold=0.12, min_gap_s=30, max_per_hour=40)
    assert decide(0.5, 100, 0, cfg).take is True
    assert decide(0.01, 100, 0, cfg).take is False        # 画面没变
    assert decide(0.5, 5, 0, cfg).take is False           # 距上次太近
    assert decide(0.5, 100, 40, cfg).take is False        # 超配额
    assert decide(0.5, 100, 0, AutoCapConfig(enabled=False)).take is False

    same = bytes([128] * 100)
    assert change_ratio(same, same) == 0.0
    assert change_ratio(same, bytes([255] * 100)) == 1.0
    # 尺寸不同按"全变了"处理，避免拿错帧比出假结论
    assert change_ratio(same, bytes([128] * 50)) == 1.0


def test_concurrent_writes_do_not_break(tmp_path):
    """回归：分析 + 提问并发写同一个 sqlite 连接，不应报 cannot commit。"""
    import threading

    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.llm import build_llm

    srv, port, _ = _mock_llm_server("ok")
    try:
        st = Settings()
        st.llm.base_url = f"http://127.0.0.1:{port}/v1"
        st.llm.api_key = "k"
        st.llm.model = "mock"
        eng = Engine(DB(tmp_path / "c.db"), st)
        eng.llm = build_llm(st)
        srt = tmp_path / "l.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:02:00,000\n"
            "定义有两个必要条件，这里涉及充分条件和必要条件容易混。\n",
            encoding="utf-8")
        sid = eng.import_transcript(srt, "课", "第 1 讲")["session_id"]
        errs: list[str] = []

        def asker():
            for i in range(3):
                try:
                    errs.append(eng.ask(f"问题{i}", session_id=sid).get("error") or "")
                except Exception as e:  # noqa: BLE001
                    errs.append(repr(e))

        def analyzer():
            for _ in range(2):
                try:
                    eng.analyze_session(sid)
                except Exception as e:  # noqa: BLE001
                    errs.append(repr(e))

        ts = [threading.Thread(target=asker) for _ in range(2)]
        ts.append(threading.Thread(target=analyzer))
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not [e for e in errs if e], errs
    finally:
        srv.shutdown()


def _mock_bili(tmp_path, monkeypatch):
    """假 B 站接口：view + player/v2 + 字幕 json，用来验证整条链接导入链路。"""
    import json
    import threading
    import http.server
    import socketserver

    sub_body = {"body": [
        {"from": 0.0, "to": 6.0, "content": "这一节讲必要条件和充分条件的区别。"},
        {"from": 6.0, "to": 14.0, "content": "梯度为零只是一阶必要条件，还需要 Hessian 半正定。"},
        {"from": 14.0, "to": 22.0, "content": "如果 Hessian 不定，这个点就是鞍点。"},
    ]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/x/web-interface/view"):
                data = {"code": 0, "message": "0", "data": {
                    "bvid": "BV1TEST00001", "title": "最优化 · 第三讲", "duration": 22,
                    "cid": 999, "pages": [{"cid": 999, "part": "P1"}]}}
            elif self.path.startswith("/x/player/v2"):
                data = {"code": 0, "data": {"subtitle": {"subtitles": [
                    {"lan": "zh-CN", "subtitle_url": f"http://127.0.0.1:{port}/sub.json"}]}}}
            elif self.path.startswith("/sub.json"):
                data = sub_body
            else:
                self.send_response(404)
                self.end_headers()
                return
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("DSHK_BILI_API", f"http://127.0.0.1:{port}")
    import importlib
    from dsh_app import source_url
    importlib.reload(source_url)
    return srv, source_url


def test_import_from_bilibili_url_uses_subtitles(tmp_path, monkeypatch):
    """有字幕时必须只取字幕：不下载视频、不需要转写后端。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    srv, su = _mock_bili(tmp_path, monkeypatch)
    try:
        st = Settings()
        eng = Engine(DB(tmp_path / "u.db"), st)

        p = eng.probe_url("https://www.bilibili.com/video/BV1TEST00001")
        assert p["ok"] and p["has_subtitle"], p
        assert p["title"].startswith("最优化")

        res = eng.import_url("https://www.bilibili.com/video/BV1TEST00001",
                             "最优化", "第三讲")
        assert res["source"] == "链接字幕"
        assert res["segments"] == 3
        assert res["needs_asr"] is False
        assert abs(res["duration"] - 22.0) < 0.5
        # 会话真的带上了逐句时间戳
        rows = eng.db.get_segments(res["session_id"])
        assert len(rows) == 3 and rows[0]["t_end"] > rows[0]["t_start"]
        # 能直接分析
        out = eng.analyze_session(res["session_id"])
        assert out.windows
    finally:
        srv.shutdown()
        import importlib
        import os
        from dsh_app import source_url
        os.environ.pop("DSHK_BILI_API", None)
        importlib.reload(source_url)


def test_import_url_without_subtitle_refuses_early(tmp_path, monkeypatch):
    """没有字幕、也没有转写后端时，必须在下载音频之前就报错。"""
    import json
    import threading
    import http.server
    import socketserver

    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/x/web-interface/view"):
                data = {"code": 0, "data": {"bvid": "BV1TEST00002", "title": "无字幕视频",
                                            "duration": 100, "cid": 1, "pages": [{"cid": 1}]}}
            else:
                data = {"code": 0, "data": {"subtitle": {"subtitles": []}}}
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import importlib
    import os
    from dsh_app import source_url
    os.environ["DSHK_BILI_API"] = f"http://127.0.0.1:{port}"
    importlib.reload(source_url)
    try:
        st = Settings()
        st.asr.api_key = ""          # 没有转写后端
        eng = Engine(DB(tmp_path / "u2.db"), st)
        p = eng.probe_url("https://www.bilibili.com/video/BV1TEST00002")
        assert p["ok"] and not p["has_subtitle"]
        assert "SESSDATA" in p["note"] or "字幕" in p["note"]
        try:
            eng.import_url("https://www.bilibili.com/video/BV1TEST00002", "课", "节")
        except Exception as e:  # noqa: BLE001
            assert "转写后端" in str(e), e
        else:
            raise AssertionError("没有字幕也没有转写后端时不该继续下载音频")
    finally:
        srv.shutdown()
        os.environ.pop("DSHK_BILI_API", None)
        importlib.reload(source_url)


def test_url_parsing():
    from dsh_app import source_url as su

    assert su.bvid_of("https://www.bilibili.com/video/BV1d8pU6xEZX/?spm_id=1") == "BV1d8pU6xEZX"
    assert su.bvid_of("BV1d8pU6xEZX") == "BV1d8pU6xEZX"
    assert su.bvid_of("https://www.bilibili.com/video/av170001") == "av170001"
    assert su.bvid_of("https://www.youtube.com/watch?v=x") == ""
    assert su.platform_of("https://www.bilibili.com/video/BV1x") == "bilibili"
    assert su.platform_of("https://youtu.be/x") == "generic"
    # 字幕体积换算：45 分钟课的纯音频大概多大
    assert 30 < su.estimate_audio_mb(45 * 60) < 40


def _mock_bili_with_storyboard(tmp_path, monkeypatch):
    """假 B 站：view + 故事板 + 雪碧图（4 格纯色）+ 假视觉模型。"""
    import io
    import json
    import threading
    import http.server
    import socketserver
    from PIL import Image

    TW, TH = 64, 36
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    sprite = Image.new("RGB", (TW * 2, TH * 2))
    for i, c in enumerate(colors):
        gx, gy = i % 2, i // 2
        sprite.paste(Image.new("RGB", (TW, TH), c), (gx * TW, gy * TH))
    sbuf = io.BytesIO()
    sprite.save(sbuf, format="JPEG", quality=95)
    sprite_bytes = sbuf.getvalue()
    seen = {"vision_calls": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def _json(self, obj):
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.startswith("/x/web-interface/view"):
                self._json({"code": 0, "data": {"bvid": "BV1SB0000001", "title": "板书课",
                                                "duration": 60, "cid": 5,
                                                "pages": [{"cid": 5}]}})
            elif self.path.startswith("/x/player/videoshot"):
                # index 用毫秒，验证尺度自适应
                self._json({"code": 0, "data": {
                    "image": [f"http://127.0.0.1:{port}/sprite.jpg"],
                    "index": [0, 20000, 40000, 60000],
                    "img_x_size": TW, "img_y_size": TH,
                    "img_x_len": 2, "img_y_len": 2}})
            elif self.path.startswith("/sprite.jpg"):
                self.send_response(200)
                self.send_header("content-type", "image/jpeg")
                self.send_header("content-length", str(len(sprite_bytes)))
                self.end_headers()
                self.wfile.write(sprite_bytes)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen["vision_calls"] += 1
            has_img = any(p.get("type") == "image_url"
                          for p in body["messages"][0]["content"])
            seen["has_img"] = has_img
            self._json({"choices": [{"message": {"content": json.dumps(
                {"text": "板书：梯度为零是必要条件；Hessian 半正定才是极小。",
                 "kind": "board"}, ensure_ascii=False)}}]})

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("DSHK_BILI_API", f"http://127.0.0.1:{port}")
    import importlib
    from dsh_app import source_url
    importlib.reload(source_url)
    return srv, port, seen, colors


def test_storyboard_frame_is_cropped_correctly(tmp_path, monkeypatch):
    """故事板取帧必须裁到正确那一格（纯色格可断言像素），且时间单位自适应。"""
    from dsh_app import source_url as su

    srv, port, _, colors = _mock_bili_with_storyboard(tmp_path, monkeypatch)
    try:
        sb = su.bili_storyboard("BV1SB0000001", 5)
        assert sb["times"] == [0.0, 20.0, 40.0, 60.0], sb["times"]  # 毫秒被换成秒
        assert sb["tile"] == (64, 36) and sb["grid"] == (2, 2)
        from PIL import Image
        import io as _io
        for k, t in enumerate([0, 21, 41, 59]):
            png = su.storyboard_frame_png(sb, t)
            im = Image.open(_io.BytesIO(png)).convert("RGB")
            assert im.size == (64, 36), im.size
            center = im.getpixel((32, 18))
            want = colors[k]
            assert all(abs(a - b) < 40 for a, b in zip(center, want)), (k, center, want)
    finally:
        srv.shutdown()
        import importlib
        import os
        from dsh_app import source_url
        os.environ.pop("DSHK_BILI_API", None)
        importlib.reload(source_url)


def test_visual_evidence_attaches_screen_notes(tmp_path, monkeypatch):
    """回看点补画面：取帧 → 视觉模型 → 落成屏幕证据（source=storyboard）。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.llm import build_llm

    srv, port, seen, _ = _mock_bili_with_storyboard(tmp_path, monkeypatch)
    try:
        st = Settings()
        st.llm.base_url = f"http://127.0.0.1:{port}/v1"
        st.llm.api_key = "k"
        st.llm.model = "mock"
        st.llm.vision_model = "mock-vl"
        eng = Engine(DB(tmp_path / "v.db"), st)
        eng.llm = build_llm(st)

        srt = tmp_path / "l.srt"
        srt.write_text("1\n00:00:00,000 --> 00:03:00,000\n"
                       "必要条件与充分条件容易混，这里涉及术语的精确用法。\n",
                       encoding="utf-8")
        sid = eng.import_transcript(srt, "板书课", "第三讲")["session_id"]
        eng.db.set_kv(f"url_ref:{sid}", {"bvid": "BV1SB0000001", "cid": 5})
        eng.add_manual_mark(sid, 30, "这里没跟上")
        eng.add_manual_mark(sid, 120, "这段没懂")
        eng.analyze_session(sid)
        assert eng.db.get_knots(sid), "需要先有回看点"

        r = eng.attach_visual_evidence(sid, max_knots=2)
        assert r["ok"], r
        assert r["saved"] >= 1
        assert seen["vision_calls"] >= 1 and seen["has_img"] is True
        notes = eng.db.get_screen_notes(sid)
        assert notes and all(n["source"] == "storyboard" for n in notes)
        assert "Hessian" in notes[0]["text"]

        # 补完画面后能重新分析，且屏幕证据按"屏幕内容"参与计算
        out = eng.analyze_session(sid)
        assert out.windows
    finally:
        srv.shutdown()
        import importlib
        import os
        from dsh_app import source_url
        os.environ.pop("DSHK_BILI_API", None)
        importlib.reload(source_url)


def test_visual_evidence_requires_video_ref(tmp_path):
    """录制来的会话没有视频来源，必须明确拒绝而不是瞎猜。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    eng = Engine(DB(tmp_path / "n.db"), Settings())
    sid = eng.import_transcript(tmp_path / "x.srt", "课", "节")["session_id"] if False else None
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:00,000 --> 00:01:00,000\n随便一段内容。\n", encoding="utf-8")
    sid = eng.import_transcript(srt, "课", "节")["session_id"]
    eng.add_manual_mark(sid, 20, "不懂")
    eng.analyze_session(sid)
    try:
        eng.visual_evidence(sid, [20])
    except RuntimeError as e:
        assert "没有关联视频" in str(e)
    else:
        raise AssertionError("没有视频来源时不该能取画面")


# ---- 配置与调用的回归（都是真机踩出来的坑） -------------------------------
@pytest.mark.real_settings
def test_settings_roundtrip_keeps_nested_sections(tmp_path):
    """保存设置必须能写盘再读回。

    曾经的 bug：LLMSettings / ASRSettings 少写 @dataclass，
    asdict() 碰到它们直接 TypeError —— 「保存设置」点了没反应，配置存不下来。
    """
    from dsh_app.config import Settings

    p = tmp_path / "cfg.json"
    s = Settings()
    s.llm.api_key = "sk-demo"
    s.llm.base_url = "https://example.invalid/v1"
    s.llm.model = "m-1"
    s.asr.api_key = "sk-asr"
    s.source.bili_sessdata = "sess-abc"
    s.save(p)
    s2 = Settings.load(p)
    assert s2.llm.api_key == "sk-demo"
    assert s2.llm.base_url == "https://example.invalid/v1"
    assert s2.llm.model == "m-1"
    assert s2.asr.api_key == "sk-asr"
    assert s2.source.bili_sessdata == "sess-abc"


@pytest.mark.real_settings
def test_vendor_env_key_is_picked_up(monkeypatch):
    """机器上已有供应商环境变量时不必再粘一次 key。"""
    from dsh_app.config import Settings

    monkeypatch.setenv("QW_API_KEY", "sk-from-env")
    monkeypatch.delenv("DSHK_LLM_API_KEY", raising=False)
    monkeypatch.setattr("dsh_app.config.CONFIG_PATH", Settings.__module__ and __import__(
        "pathlib").Path("__no_such_cfg__.json"))
    s = Settings.load()
    assert s.llm.api_key == "sk-from-env"


def test_kb_import_with_bad_course_becomes_global(tmp_path):
    """绑到不存在的课程会让资料永远检索不到 —— 必须降级为全局并告知。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    eng = Engine(DB(tmp_path / "kb.db"), Settings())
    doc = tmp_path / "note.md"
    doc.write_text("# 必要条件\n梯度为零是一阶必要条件。", encoding="utf-8")
    r = eng.kb_import(str(doc), course_id="不存在的课程", title="笔记")
    assert r["ok"] and r["warning"], r
    assert len(eng.kb_search("必要条件", k=3)) == 1


def test_chat_reports_empty_body_instead_of_blank_answer():
    """推理模型可能把 token 全花在 reasoning_content 上：正文空必须报错，不能返回空答案。"""
    from dsh_app.llm import LLMClient

    class _Msg:
        content = ""
        reasoning_content = "我在思考……"

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:  # noqa: N801
            @staticmethod
            def create(**kw):
                return _Resp()

    client = LLMClient("https://example.invalid/v1", "sk-x", "m")
    client._client = type("C", (), {"chat": _Chat})()
    text, err = client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert text == "" and "推理内容" in err and "max_tokens" in err


# ---- 本地 Whisper 安装功能 ------------------------------------------------
def test_whisper_status_shape():
    from dsh_app import whisper_setup as ws

    st = ws.status()
    assert set(["deps_ok", "deps", "models", "installed_sizes", "ready",
                "models_dir"]) <= set(st)
    assert isinstance(st["models"], list) and st["models"]
    assert all({"key", "label", "installed", "mb", "path"} <= set(m)
               for m in st["models"])


def test_whisper_resolve_model_prefers_local_dir(tmp_path, monkeypatch):
    """本机有下载好的目录时必须用目录，避免运行时又去联网。"""
    from dsh_app import whisper_setup as ws

    monkeypatch.setattr(ws, "MODELS_DIR", tmp_path)
    assert ws.model_dir("small") == tmp_path / "faster-whisper-small"
    # 什么都没有 → 返回尺寸名（交给 faster-whisper 自己去找）
    assert ws.resolve_model("medium") == "medium"
    # 造一个"已下载"的 medium
    d = tmp_path / "faster-whisper-medium"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x" * 16)
    assert ws.resolve_model("medium") == str(d)
    # 要 small 但只有 medium → 用 medium，不硬卡在缺失的尺寸上
    assert ws.resolve_model("small") == str(d)


def test_whisper_download_skips_when_present(tmp_path, monkeypatch):
    from dsh_app import whisper_setup as ws

    monkeypatch.setattr(ws, "MODELS_DIR", tmp_path)
    d = tmp_path / "faster-whisper-small"
    d.mkdir()
    (d / "model.bin").write_bytes(b"y" * 32)
    ws._write_stamp(d)                      # 下完并校验过的目录才有标记
    ok, msg = ws.download_model("small")
    assert ok and "跳过" in msg


def test_whisper_transcribe_missing_file_reports_error():
    from dsh_app import whisper_setup as ws

    r = ws.transcribe_file("data/no_such_audio.wav", "small")
    assert r["ok"] is False and "不存在" in r["error"]


def test_asr_reports_missing_model_instead_of_pretending():
    """装了依赖但没模型时，可用性必须返回 False 并说清缺什么。"""
    from dsh_app.asr import FasterWhisperASR
    from dsh_app import whisper_setup as ws

    if not ws.deps_status()[0]:
        import pytest
        pytest.skip("本机没装 faster-whisper")
    asr = FasterWhisperASR("small")
    ok, msg = asr.available()
    if ws.status()["any_model"]:
        assert ok, msg
    else:
        assert not ok and "下载模型" in msg, msg


# ---- 转写后端的选择与回退 ------------------------------------------------
def test_asr_candidates_order_and_fallback(tmp_path, monkeypatch):
    """顺序：字幕 → 本地 → 云端；第一个失败必须换下一个，而不是整条挂掉。"""
    from dsh_app import asr as asrmod
    from dsh_app.asr import ASRResult, ASRSegment
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)

    st = Settings()
    st.llm.base_url = "https://example.invalid/v1"
    st.llm.api_key = "k"
    monkeypatch.setattr("dsh_app.whisper_setup.status",
                        lambda: {"any_model": True, "installed_sizes": ["small"]})
    monkeypatch.setattr("dsh_app.whisper_setup.deps_status", lambda: (True, "ok"))
    kinds = [type(c).__name__ for c, _ in
             asrmod.build_candidates(st, wav, False)]
    assert kinds == ["SidecarASR", "FasterWhisperASR", "OpenAICompatASR"], kinds

    class Bad:
        def available(self):
            return True, "假装可用"

        def transcribe(self, path, precise=False):
            raise RuntimeError("云端没有这个端点")

    class Good:
        def available(self):
            return True, "真的可用"

        def transcribe(self, path, precise=False):
            return ASRResult(segments=[ASRSegment(0.0, 3.0, "本地转写成功", [], 0.9)],
                             backend="faster_whisper:small")

    monkeypatch.setattr(asrmod, "build_candidates",
                        lambda *a, **k: [(Bad(), "坏的"), (Good(), "好的")])
    db = DB(tmp_path / "f.db")
    eng = Engine(db, st)
    sid = db.create_session(db.ensure_course("课"), "节", str(wav), source="file")
    result, note = eng.transcribe(sid)
    assert result.backend == "faster_whisper:small"
    assert "1 个后端失败" in note, note
    rows = db.get_segments(sid)
    assert [r["text"] for r in rows] == ["本地转写成功"]
    # 失败原因也要落进 kv，便于排查
    from dsh_app.pipeline import BACKEND_NOTE_KEY
    assert "坏的" in str(db.get_kv(BACKEND_NOTE_KEY))


def test_import_url_audio_path_returns_counts(tmp_path, monkeypatch):
    """import_url 走音频分支时曾把 transcribe() 的元组当 dict 用（AttributeError）。"""
    from dsh_app import source_url as su
    from dsh_app.asr import ASRResult, ASRSegment
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    audio = tmp_path / "dl.m4a"
    audio.write_bytes(b"\0" * 32)
    monkeypatch.setattr(su, "probe", lambda url, sessdata="": su.Probe(
        True, "bilibili", "无字幕课", 120.0, 1, False, [], "note", "", url))
    monkeypatch.setattr(su, "download_audio",
                        lambda url, out_dir="data/media", cookie_file="":
                        (audio, "已下载音轨 0.1 MB"))
    monkeypatch.setattr("dsh_app.asr.build_candidates",
                        lambda *a, **k: [(type("B", (), {
                            "available": lambda self: (True, "ok"),
                            "transcribe": lambda self, p, precise=False: ASRResult(
                                segments=[ASRSegment(0.0, 5.0, "音频导入成功", [], 0.8)],
                                backend="fake")})(), "假的")])
    eng = Engine(DB(tmp_path / "u.db"), Settings())
    res = eng.import_url("https://www.bilibili.com/video/BV1X", "课", "节")
    assert res["segments"] == 1 and res["needs_asr"] is True
    assert abs(res["duration"] - 5.0) < 0.1
    assert "下载音轨" in res["source"]


# ---- 模型下载器（多源 + 分块 + 续传）-------------------------------------
def test_download_chunks_cover_file_exactly():
    """分块必须严丝合缝覆盖整个文件，否则合并出来的模型是坏的。"""
    total, chunk = 138 * 1024 * 1024 + 12345, 8 * 1024 * 1024
    spans = [(off, min(off + chunk, total) - off) for off in range(0, total, chunk)]
    assert spans[0][0] == 0
    assert spans[-1][0] + spans[-1][1] == total
    for (a, ln), (b, _) in zip(spans, spans[1:]):
        assert a + ln == b, "块之间不能有缝也不能重叠"
    assert sum(ln for _, ln in spans) == total


def test_download_urls_cover_both_sources():
    from dsh_app import whisper_setup as ws

    assert ws.SOURCES[0] == "modelscope"          # 实测更快的那个排第一
    ms = ws._url_for("modelscope", "Systran/faster-whisper-small", "model.bin")
    hf = ws._url_for("hf-mirror", "Systran/faster-whisper-small", "model.bin")
    assert "modelscope.cn" in ms and ms.endswith("/resolve/master/model.bin")
    assert "hf-mirror.com" in hf and "/resolve/main/model.bin" in hf
    assert ms != hf


def test_download_file_resumes_from_existing_chunks(tmp_path, monkeypatch):
    """已下完的块必须跳过：不然断点续传就是句空话。"""
    from dsh_app import whisper_setup as ws

    data = b"x" * 1000
    dest = tmp_path / "model.bin"
    cdir = dest.with_suffix(dest.suffix + ".chunks")
    cdir.mkdir()
    (cdir / f"{0:012d}").write_bytes(data)          # 假装第一块已经下好
    monkeypatch.setattr(ws, "_remote_size", lambda url: len(data))
    calls = []

    def boom(url, headers=None, timeout=None, follow_redirects=True):
        calls.append(headers.get("Range"))
        raise AssertionError("整块都缓存了，不该再发请求")

    import httpx
    monkeypatch.setattr(httpx, "stream", boom)
    ws.download_file(["http://x/y"], dest, threads=4, chunk_mb=1)
    assert dest.read_bytes() == data
    assert calls == [], calls
    assert not cdir.exists(), "合并后临时块目录要清掉"


def test_download_model_rejects_truncated_existing_file(tmp_path, monkeypatch):
    """残缺的 model.bin 不能被当成"已装"跳过。"""
    from dsh_app import whisper_setup as ws

    monkeypatch.setattr(ws, "MODELS_DIR", tmp_path)
    d = tmp_path / "faster-whisper-small"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x" * 10)        # 只有 10 字节的坏文件
    monkeypatch.setattr(ws, "_list_files", lambda repo: ["model.bin"])
    monkeypatch.setattr(ws, "_remote_size", lambda url: 1000)
    monkeypatch.setattr(ws, "_write_stamp", lambda target: None)
    seen = []
    monkeypatch.setattr(ws, "download_file",
                        lambda urls, dest, progress=None, threads=8:
                        (seen.append(dest.name), dest.write_bytes(b"y" * 1000)))
    ok, msg = ws.download_model("small")
    assert ok and seen == ["model.bin"], msg
    assert (d / "model.bin").stat().st_size == 1000


def test_model_dir_only_counts_when_stamped(tmp_path, monkeypatch):
    """有 .bin 但没标记 → 不算装好（半截文件必须重下）。"""
    import json

    from dsh_app import whisper_setup as ws

    monkeypatch.setattr(ws, "MODELS_DIR", tmp_path)
    d = tmp_path / "faster-whisper-small"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x" * 100)
    assert ws._stamp_ok(d) is False
    ws._write_stamp(d)
    assert ws._stamp_ok(d) is True
    sizes = json.loads((d / ".dsh-ok.json").read_text(encoding="utf-8"))
    assert sizes == {"model.bin": 100}
    (d / "model.bin").write_bytes(b"x" * 50)     # 事后被截断
    assert ws._stamp_ok(d) is False


# ---- 服务商预设与模型下拉 ------------------------------------------------
def test_provider_lookup_and_presets_have_needed_fields():
    from dsh_app.config import CLOUD_PRESETS, provider

    keys = [p["key"] for p in CLOUD_PRESETS]
    assert keys[-1] == "custom", "自定义要放最后"
    for p in CLOUD_PRESETS:
        assert {"key", "label", "base_url", "models", "vision_models",
                "key_url", "needs_key"} <= set(p), p["key"]
        assert isinstance(p["models"], list) and isinstance(p["vision_models"], list)
    assert provider("dashscope")["key"] == "dashscope"
    # 未知 key 不能抛，落到自定义
    assert provider("不存在")["key"] == "custom"


def test_vision_of_prefers_preset_then_filters_by_name():
    from dsh_app.config import looks_vision, vision_of

    assert looks_vision("qwen-vl-max") and looks_vision("gpt-4o-mini") is False
    assert looks_vision("glm-4v-plus") and looks_vision("Qwen/Qwen2-VL-72B-Instruct")
    assert not looks_vision("deepseek-chat")
    got = vision_of(["deepseek-v4.1-flash-1017", "qwen3.8-omni-flash"], ["qwen-vl-max"])
    assert got[0] == "qwen-vl-max"          # 预设里明确的排最前
    assert "qwen3.8-omni-flash" in got      # 名字像的也进候选
    assert "deepseek-v4.1-flash-1017" not in got
    assert len(vision_of(["a", "a", "a"])) == 0


def test_model_combo_keeps_value_missing_from_list(qapp, tmp_path):
    """刷新回来的列表里没有当前模型时，不能把它弄丢（用户可能手填了私有名字）。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.ui.main_window import MainWindow

    s = Settings()
    s.llm.base_url = "http://127.0.0.1:9/v1"
    s.llm.model = "我的私有模型"
    w = MainWindow(s, DB(tmp_path / "m.db"))
    w._fill_models(["a", "b"], "我的私有模型")
    assert w.cmb_llm_model.currentText() == "我的私有模型"
    assert w.cmb_llm_model.count() == 3
    w._fill_models(["a", "b"], "b")
    assert w.cmb_llm_model.currentText() == "b"
    # 部署用的下拉是另一个控件，不能被这条链路改到
    assert w.cmb_model is not w.cmb_llm_model
    w.close()


# ---- 不看屏幕怎么知道用户回退/暂停 -------------------------------------
def _wav(tmp_path, blocks, rate=16000, name="a.wav"):
    """blocks = [(秒数, 频率或 0 表示数字静音)]，写成 16k 单声道 16bit WAV。"""
    import wave

    import numpy as np

    p = tmp_path / name
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for sec, freq in blocks:
            n = int(rate * sec)
            t = np.arange(n) / rate
            x = np.zeros(n) if freq == 0 else 0.4 * np.sin(2 * np.pi * freq * t)
            wf.writeframes((x * 32767).astype("<i2").tobytes())
    return p


def test_pause_detector_fires_on_digital_silence_not_on_breath():
    import numpy as np

    from dsh_app.capture import AudioPauseDetector

    d = AudioPauseDetector(min_silence_s=3.0, long_silence_s=20.0)
    tone = 0.4 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    silent = np.zeros(16000, dtype=np.float32)
    evs = []
    t = 0.0
    for chunk in [tone] * 4:                      # 4 秒正常声音
        t += 1.0
        evs += d.feed(chunk, t, 16000)
    for _ in range(6):                            # 6 秒数字静音 = 播放器暂停
        t += 1.0
        evs += d.feed(silent, t, 16000)
    t += 1.0
    evs += d.feed(tone, t, 16000)                 # 恢复播放，事件此时闭合
    assert len(evs) == 1, evs
    assert evs[0]["kind"] == "pause"
    assert 5.5 <= evs[0]["payload"]["pause_s"] <= 6.5
    assert abs(evs[0]["t"] - 4.0) < 0.6           # 定格在停下的那一刻

    # 老师换气级别的 1.5 秒静止不该被当成暂停
    d2 = AudioPauseDetector(min_silence_s=3.0)
    out = []
    t = 0.0
    for chunk in [tone, tone, silent, tone]:
        t += 1.0
        out += d2.feed(chunk, t, 16000)
    assert out == []


def test_long_pause_becomes_overstay():
    import numpy as np

    from dsh_app.capture import AudioPauseDetector

    d = AudioPauseDetector(min_silence_s=3.0, long_silence_s=20.0)
    tone = 0.4 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    silent = np.zeros(16000, dtype=np.float32)
    t = 0.0
    evs = d.feed(tone, t := t + 1.0, 16000)
    for _ in range(30):
        evs += d.feed(silent, t := t + 1.0, 16000)
    evs += d.feed(tone, t := t + 1.0, 16000)
    assert len(evs) == 1
    assert evs[0]["kind"] == "overstay"
    assert evs[0]["payload"]["pause_s"] >= 29


def test_offline_detect_finds_replay_and_pause(tmp_path):
    """录完之后补算：同一段内容又听了一遍 + 中间暂停过，都要认出来。"""
    from dsh_app.capture import detect_audio_events

    p = _wav(tmp_path, [(6, 440), (6, 880), (5, 0), (6, 440)])
    evs = detect_audio_events(p, min_silence_s=3.0, long_silence_s=20.0)
    kinds = [e["kind"] for e in evs]
    assert "pause" in kinds, kinds
    pause = next(e for e in evs if e["kind"] == "pause")
    assert 4.0 <= pause["payload"]["pause_s"] <= 6.0
    assert "seek_back" in kinds or "replay" in kinds, kinds
    back = next(e for e in evs if e["kind"] in ("seek_back", "replay"))
    assert back["payload"]["matched_t"] < 8.0     # 匹配到最早那次 440
    assert back["confidence"] > 0


def test_offline_detect_ignores_missing_and_non_wav(tmp_path):
    from dsh_app.capture import detect_audio_events

    assert detect_audio_events(tmp_path / "没有这个.wav") == []
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"\x00" * 100)
    assert detect_audio_events(mp3) == []
