"""练习导入：老师发的 xlsx / csv 要能直接变成能判分的题。

这一层最不该有的行为是"静默成功 0 条"：表格格式千奇百怪，
认不出列、答案对不上选项，都必须带着行号说出来。
"""
from __future__ import annotations

import csv
import json

import pytest

from dsh_app import exercises as ex
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine


# ---------------------------------------------------------------- 模板与往返
@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_template_roundtrip(tmp_path, suffix):
    p = ex.write_template(tmp_path / f"tpl{suffix}")
    assert p.exists() and p.stat().st_size > 0
    rows, errors = ex.load_file(p)
    assert not errors
    assert len(rows) == 2
    first = rows[0]
    assert first.choices and len(first.choices) == 4
    assert first.answer_index == 0
    assert first.kp_name


def test_english_headers_and_numbered_options(tmp_path):
    p = tmp_path / "e.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "question", "option1", "option2", "option3", "option4",
                    "answer", "explanation"])
        w.writerow(["方差", "方差衡量什么？", "离散程度", "中心位置", "样本量", "概率",
                    "1", "方差是偏离均值的平方的期望。"])
    rows, errors = ex.load_file(p)
    assert not errors
    assert rows[0].answer_index == 0
    assert rows[0].explain.startswith("方差是")


@pytest.mark.parametrize("answer,expected", [
    ("B", 1), ("b", 1), ("2", 1), ("P(X 等于 x)", 1),
])
def test_answer_column_accepts_letter_number_or_text(tmp_path, answer, expected):
    p = tmp_path / "a.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["题干", "选项A", "选项B", "答案"])
        w.writerow(["分布函数 F(x) 等于什么？", "P(X 小于等于 x)", "P(X 等于 x)", answer])
    rows, errors = ex.load_file(p)
    assert not errors, errors
    assert rows[0].answer_index == expected


def test_bad_answer_is_reported_with_row_number(tmp_path):
    p = tmp_path / "bad.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["题干", "选项A", "选项B", "答案"])
        w.writerow(["题干一", "甲", "乙", "A"])
        w.writerow(["题干二", "丙", "丁", "E"])
    rows, errors = ex.load_file(p)
    assert len(rows) == 1
    assert errors and errors[0]["row_no"] == 3
    assert "E" in errors[0]["reason"]


def test_too_few_choices_reported(tmp_path):
    p = tmp_path / "few.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["题干", "选项A", "答案"])
        w.writerow(["只有两个以下选项", "甲", "A"])
    rows, errors = ex.load_file(p)
    assert rows == []
    assert any("选项不足" in e["reason"] for e in errors)


def test_gbk_csv_is_decoded(tmp_path):
    p = tmp_path / "gbk.csv"
    p.write_bytes("题干,选项A,选项B,答案\n分布函数是什么？,累积概率,密度,A\n".encode("gb18030"))
    rows, errors = ex.load_file(p)
    assert not errors
    assert rows[0].prompt == "分布函数是什么？"


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load_file(tmp_path / "nope.xlsx")


def test_unsupported_format_is_rejected(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(RuntimeError):
        ex.load_file(p)


# ---------------------------------------------------------------- 进引擎
@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    st = Settings.load()
    st.llm.base_url = ""
    st.llm.api_key = ""
    e = Engine(db, st)
    sid = e.seed_demo()
    return e, db, sid, db.get_session(sid)["course_id"]


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_import_matches_existing_kp_and_keeps_unmatched_name(tmp_path, eng):
    e, db, sid, cid = eng
    kp_name = db.get_knowledge_points(cid)[0]["name"]
    p = _write(tmp_path / "p.csv", [
        ["知识点", "题干", "选项A", "选项B", "答案", "解析"],
        [kp_name, "这道题考这个术语？", "是", "否", "A", "举例说明"],
        ["根本没讲过的术语", "另一道题？", "甲", "乙", "B", ""],
    ])
    res = e.import_exercises(p, cid)
    assert res["ok"] and res["imported"] == 2
    assert res["matched"] == 1
    assert res["unmatched"] == ["根本没讲过的术语"]
    ov = e.exercises_overview(cid)
    assert ov["total"] == 2 and ov["answered_kp"] == 1


def test_quiz_from_exercises_grades_and_updates_mastery(tmp_path, eng):
    e, db, sid, cid = eng
    kp = db.get_knowledge_points(cid)[0]
    p = _write(tmp_path / "q.csv", [
        ["知识点", "题干", "选项A", "选项B", "选项C", "选项D", "答案"],
        [kp["name"], "术语一的说法哪个对？", "对的说法", "错的", "也错", "还错", "A"],
    ])
    e.import_exercises(p, cid)
    q = e.quiz_from_exercises(cid, sid, k=3)
    assert q["ok"] and len(q["items"]) == 1
    item = q["items"][0]
    assert item["source"] == "import"
    assert item["correct_index"] == 0
    before = float(db.get_knowledge_points(cid)[0]["mastery"])
    out = e.answer_quiz(item["item_id"], 0)
    assert out["correct"] and out["counts"]
    after = float({r["id"]: r for r in db.get_knowledge_points(cid)}[int(kp["id"])]["mastery"])
    assert after > before, "答对导入题也要写掌握度"


def test_quiz_from_exercises_without_import(tmp_path, eng):
    e, db, sid, cid = eng
    q = e.quiz_from_exercises(cid, sid)
    assert not q["ok"]
    assert "导入" in q["error"]


def test_import_with_no_parseable_rows_reports_reason(tmp_path, eng):
    e, db, sid, cid = eng
    p = _write(tmp_path / "empty.csv", [["知识点", "题干", "答案"]])
    res = e.import_exercises(p, cid)
    assert not res["ok"]
    assert res["errors"], "空表必须给出原因，不能只是 0 条"


def test_second_quiz_skips_already_asked_imported_items(tmp_path, eng):
    e, db, sid, cid = eng
    kp_name = db.get_knowledge_points(cid)[0]["name"]
    p = _write(tmp_path / "two.csv", [
        ["知识点", "题干", "选项A", "选项B", "答案"],
        [kp_name, "第一道题？", "甲", "乙", "A"],
        [kp_name, "第二道题？", "丙", "丁", "B"],
    ])
    e.import_exercises(p, cid)
    first = e.quiz_from_exercises(cid, sid, k=1)
    second = e.quiz_from_exercises(cid, sid, k=1)
    assert first["items"][0]["prompt"] != second["items"][0]["prompt"]


def test_quiz_items_carry_source_column(tmp_path, eng):
    e, db, sid, cid = eng
    kp_name = db.get_knowledge_points(cid)[0]["name"]
    p = _write(tmp_path / "s.csv", [
        ["知识点", "题干", "选项A", "选项B", "答案"],
        [kp_name, "来源标记检查？", "甲", "乙", "A"],
    ])
    e.import_exercises(p, cid)
    e.quiz_from_exercises(cid, sid, k=1)
    rows = db.get_quiz_items(sid)
    assert rows and str(rows[0]["source"]) == "import"
    assert json.loads(rows[0]["choices_json"])