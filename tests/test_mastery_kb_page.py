"""掌握度闭环 + 知识库定位的回归测试。

两件事都必须有测试盯着，因为都是"看不出来坏了"的那种：
- 掌握度算错不会报错，只会让漏洞地图永远熄灭或永远不熄灭；
- 定位字段丢了也不报错，只会让引用悄悄退回「第 N 段」。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


# ------------------------------------------------------------------ 掌握度
def test_mastery_recency_weighting_and_confidence_cap():
    from dsh_app.cognitive import mastery as m

    assert m.compute_mastery([]) == 0.0
    # 一次证据最多 0.5，两次最多 0.8：防"蒙对一次就算会"
    assert m.compute_mastery([1.0]) == 0.5
    assert m.compute_mastery([1.0, 1.0]) == 0.8
    assert m.compute_mastery([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    # 近因加权：先错后对，比先对后错高
    up = m.compute_mastery([0.0, 0.0, 1.0, 1.0, 1.0])
    down = m.compute_mastery([1.0, 1.0, 1.0, 0.0, 0.0])
    assert up > down
    # 最近一次答错要把它从"已掌握"里打下来
    assert m.compute_mastery([1.0, 1.0, 1.0]) >= m.DEFAULT_GATE
    assert m.compute_mastery([1.0, 1.0, 1.0, 0.0]) < m.DEFAULT_GATE


def test_mastery_status_tiers_and_gate():
    from dsh_app.cognitive import mastery as m

    assert m.status_of(0.0, 0) == "unknown"      # 没验证过 ≠ 不会
    assert m.status_of(0.3, 2) == "weak"
    assert m.status_of(0.6, 3) == "learning"
    assert m.status_of(0.9, 3) == "mastered"
    assert m.is_mastered(0.9) and not m.is_mastered(0.89)
    assert m.gate_threshold("memory") == m.DEFAULT_GATE


def test_kp_feedback_loop_masters_knot_then_reopens(tmp_path):
    """回看点 → 知识点 → 三次证据过门槛 → 熄灭；再答错一次重新亮起。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    cid = db.ensure_course("测试课")
    sid = db.create_session(cid, "第 1 节")
    db.replace_knots(sid, [])
    kid = db.execute(
        """INSERT INTO knots(session_id, t_start, t_end, level, quadrant, grade, score,
                              l_score, b_score, evidence, evidence_json, status, excerpt,
                              created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, 10.0, 20.0, "high", "q1", "strong", 0.8, 0.7, 0.6, "你回退了 12 秒",
         "{}", "pending", "特征值", 1.0),
    ).lastrowid
    kp_id = db.upsert_knowledge_point(cid, "特征值", 10.0, sid)
    db.link_knot_point(int(kid), kp_id, 1.0)

    assert eng.knot_solved(int(kid)) is False           # 没证据 → 不算补上
    first = eng.record_feedback(int(kid), "confirm_useful")
    assert first["kp"][0]["mastery"] == 0.5
    eng.record_feedback(int(kid), "confirm_useful")
    third = eng.record_feedback(int(kid), "confirm_useful")
    assert third["kp"][0]["mastery"] == pytest.approx(1.0)
    assert eng.knot_solved(int(kid)) is True            # 熄灭

    back = eng.record_kp_attempt(kp_id, 0.0, source="self_test")
    assert back["mastery"] < 0.9                        # 近因加权把它打下来
    assert eng.knot_solved(int(kid)) is False           # 重新亮起


def test_mastered_feedback_is_qualitative_pass(tmp_path):
    """「我本来就会」是定性通过：直接 1.0，不走加权。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    cid = db.ensure_course("测试课")
    sid = db.create_session(cid, "第 1 节")
    kp_id = db.upsert_knowledge_point(cid, "必要条件", 5.0, sid)
    res = eng.record_kp_attempt(kp_id, 1.0,
                                source=__import__("dsh_app.cognitive.mastery",
                                                  fromlist=["x"]).QUALITATIVE_PASS_ACTIONS
                                and "mastered")
    assert res["mastery"] == 1.0
    assert res["status"] == "mastered"


def test_kp_overview_hides_mastered_from_root_causes(tmp_path):
    """已过门槛的知识点要从「结构性根因」里退场，否则清单只增不减。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    cid = db.ensure_course("测试课")
    s1 = db.create_session(cid, "第 1 节")
    s2 = db.create_session(cid, "第 2 节")
    kp_id = db.upsert_knowledge_point(cid, "协方差", 1.0, s1)
    db.upsert_knowledge_point(cid, "协方差", 1.0, s2)
    for sid, t in ((s1, 5.0), (s2, 7.0)):
        kid = db.execute(
            """INSERT INTO knots(session_id, t_start, t_end, level, quadrant, grade, score,
                                  l_score, b_score, evidence, evidence_json, status, excerpt,
                                  created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, t, t + 10, "high", "q1", "strong", 0.7, 0.6, 0.6, "e",
             "{}", "pending", "协方差", 1.0),
        ).lastrowid
        db.link_knot_point(int(kid), kp_id, 1.0)

    assert [r["name"] for r in eng._root_cause_terms(cid)] == ["协方差"]
    for _ in range(3):
        eng.record_kp_attempt(kp_id, 1.0, source="confirm_useful")
    assert eng._root_cause_terms(cid) == []          # 补上了就不再是根因
    assert eng.kp_overview(cid)[0]["status"] == "mastered"


# ------------------------------------------------------------------ 知识库定位
def test_read_document_blocks_keeps_pages_sections_and_timecodes(tmp_path):
    from dsh_app import kb

    # markdown：按标题分小节（标题先切、再清标记，顺序反了就切不出来）
    md = tmp_path / "notes.md"
    md.write_text("# 第一章\n\n分布函数单调不减。\n\n## 1.2 期望\n\n期望是加权平均。\n",
                  encoding="utf-8")
    blocks, _ = kb.read_document_blocks(md)
    assert [b["section"] for b in blocks] == ["第一章", "1.2 期望"]
    assert all(b["source_ref"] for b in blocks)

    # 字幕：每条带时间码
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:03,500 --> 00:00:06,000\n矩阵的秩定义。\n\n"
                   "2\n00:01:12,000 --> 00:01:15,000\n秩是列空间维数。\n", encoding="utf-8")
    blocks, _ = kb.read_document_blocks(srt)
    assert [b["source_ref"] for b in blocks] == ["00:03", "01:12"]

    # pptx：一张幻灯片一条，带页号
    pptx = tmp_path / "d.pptx"
    with zipfile.ZipFile(pptx, "w") as z:
        for i, t in enumerate(["秩的定义", "秩与无关性"], 1):
            z.writestr(f"ppt/slides/slide{i}.xml", f'<a:t>{t}</a:t>')
    blocks, _ = kb.read_document_blocks(pptx)
    assert [(b["page"], b["source_ref"]) for b in blocks] == [(1, "幻灯片 1"), (2, "幻灯片 2")]

    # docx：标题样式做小节
    docx = tmp_path / "d.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml",
                   '<w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
                   '<w:r><w:t>第二章</w:t></w:r></w:p>'
                   '<w:p><w:r><w:t>特征多项式决定特征值。</w:t></w:r></w:p></w:body>')
    blocks, _ = kb.read_document_blocks(docx)
    assert blocks[0]["section"] == "第二章"
    assert "特征多项式" in blocks[0]["text"]


def test_chunk_blocks_never_crosses_a_page_and_keeps_short_slides(tmp_path):
    from dsh_app import kb

    blocks = [{"text": "第一页内容" * 200, "page": 1, "section": "", "source_ref": "p.1"},
              {"text": "秩的定义", "page": 2, "section": "", "source_ref": "幻灯片 2"}]
    chunks = kb.chunk_blocks(blocks)
    assert len(chunks) > 2
    assert all(c["page"] == 1 for c in chunks[:-1])       # 一个块绝不跨页
    assert chunks[-1]["page"] == 2
    assert chunks[-1]["text"] == "秩的定义"                # 短标题不能被 8 字门槛吃掉
    assert all(c["source_ref"] for c in chunks)


def test_kb_import_then_search_returns_page_label(tmp_path):
    from dsh_app import chat, kb
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    md = tmp_path / "老周笔记.md"
    md.write_text("# 特征值\n\n特征多项式决定特征值，秩等于非零特征值个数。\n",
                  encoding="utf-8")
    r = eng.kb_import(str(md))
    assert r["ok"] and r["located"] == r["chunks"]

    hits = chat.kb_search(db, "特征多项式 特征值", k=3)
    assert hits, "导入后应当能检索到"
    assert hits[0]["label"].startswith("《")
    msgs = chat.build_messages("特征值怎么算", "", hits)
    body = msgs[-1]["content"]
    assert "来源《老周笔记》" in body
    assert "第 1 段（相关度" not in body or "《" in body   # 标签优先于段号


def test_cite_label_falls_back_in_order():
    from dsh_app import kb

    assert kb.cite_label({"source_ref": "p.12"}) == "p.12"
    assert kb.cite_label({"page": 7}) == "p.7"
    assert kb.cite_label({"section": "第二章"}) == "《第二章》"
    assert kb.cite_label({}) == ""

def test_main_window_lists_kb_without_any_selected_session(tmp_path, qapp):
    """没有选中会话时也不许抛异常，全局资料必须照样列出来。

    这条守的是 QListWidget.currentData() 那个坑：写错时异常被 Qt 的定时器回调吞掉，
    界面只是"知识库空了"，看起来像数据丢了，实际是代码挂了。
    """
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.ui.main_window import MainWindow

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    md = tmp_path / "n.md"
    md.write_text("# 第一章\n\n特征值就是使方程有非零解的那个数。\n", encoding="utf-8")
    assert eng.kb_import(str(md))["ok"]

    st = Settings.load()
    st.llm.preload_on_start = False
    win = MainWindow(st, db)
    assert win._current_course_id() == ""       # 没有任何课程，也不能抛
    win._refresh_kb()
    assert win.kb_list.count() == 1
    assert "全局" in win.kb_list.item(0).text()
    win.refresh_kp_table()                      # 空课程也要能刷成 0 行
    assert win.kp_table.rowCount() == 0
