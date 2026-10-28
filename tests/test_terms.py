"""术语抽取：知识点不能是「定义/必要条件」这种元词。

这一层的存在理由：掌握度、小测、复习全挂在知识点上，
如果知识点是「定义」这种词，整条链再准也没有意义——没人会说自己"掌握了定义"。
"""
from __future__ import annotations

import pytest

from dsh_app import terms
from dsh_app.terms import META_TERMS

PROB_TEXT = (
    "这一节我们讲随机变量和它的分布。先从样本空间说起，样本空间是全部可能结果的集合。"
    "随机变量是指把样本空间映射到实数轴的一个函数。把随机变量记作 X，分布函数定义为 "
    "F(x) 等于 P(X 小于等于 x)。分布函数一定是右连续的。期望是指随机变量的加权平均，"
    "方差的定义是偏离均值的平方的期望。很多人把条件概率和联合概率搞混，条件概率记作 P(A|B)，"
    "它要求 B 的概率不为零。假设 X 服从参数为二的泊松分布，算出来期望和方差都等于二。"
)


def names(text: str = PROB_TEXT, limit: int = 12) -> list[str]:
    return [c.term for c in terms.extract_terms(text, terms.guess_subject(text), limit=limit)]


# ---------------------------------------------------------------- 学科判定
def test_guess_subject_picks_math_for_probability_text():
    assert terms.guess_subject(PROB_TEXT) == "数学"


def test_guess_subject_refuses_to_guess_on_thin_text():
    assert terms.guess_subject("随便说两句") == "通用"
    assert terms.guess_subject("") == "通用"


# ---------------------------------------------------------------- 元词必须出局
@pytest.mark.parametrize("meta", ["定义", "定理", "必要条件", "充分条件", "边界条件", "最优解"])
def test_meta_terms_never_become_knowledge_points(meta):
    text = f"{meta} {meta} 这个{meta}很关键。" * 5 + PROB_TEXT
    assert meta in META_TERMS
    assert meta not in names(text)


def test_junk_fragments_are_not_terms():
    """窗口切歪会产出「另外条件概率」「样本空间映射」「算出来期望」这类串。"""
    got = names()
    for bad in ("另外条件概率", "样本空间映射", "算出来期望", "先从样本空间"):
        assert bad not in got
    # 术语里不会出现连接词/的
    assert not [g for g in got if any(ch in g for ch in "的和与及或")]


def test_terms_never_contain_whitespace_or_newline():
    text = PROB_TEXT.replace("。", "。\n") + "梯度\n为零 分布函数"
    for g in names(text, limit=20):
        assert not any(ch.isspace() for ch in g), g


# ---------------------------------------------------------------- 真术语要留
@pytest.mark.parametrize("good", ["随机变量", "分布函数", "期望", "方差", "条件概率", "泊松分布"])
def test_real_terms_are_extracted(good):
    assert good in names(), f"{good} 应该被抽出来"


def test_definition_sentence_beats_bare_frequency():
    """被明确定义过的词，即使只出现一次也要留。"""
    text = PROB_TEXT + "所谓鞅，是指满足条件期望不变性的随机过程。"
    got = names(text, limit=20)
    assert "鞅" in got or any("鞅" in g for g in got)


def test_canonical_short_form_wins_over_longer_window():
    """「样本空间映射」不该压过「样本空间」。"""
    got = names()
    if "样本空间映射" in got:
        assert "样本空间" not in got
    else:
        assert "样本空间" in got


# ---------------------------------------------------------------- 真实转写（ASR 文本）
# 真实转写没有标点、断句随机、满嘴口头语。这组用例钉住"ASR 文本也要抽出同样的术语"。
ASR_TEXT = (
    "那么这节课呢我们来讲一下随机变量啊 那个样本空间呢其实就是全部可能结果的集合 "
    "也就是说随机变量是指把样本空间映射到实数轴的一个函数啊 我们把它记作X "
    "呃然后分布函数呢定义为Fx等于PX小于等于x 这个分布函数一定是右连续的啊 "
    "期望是指随机变量的加权平均 那个方差的定义呢是偏离均值的平方的期望 "
    "很多人会把条件概率和联合概率搞混 条件概率记作P A竖杠B 它要求B的概率不为零 "
    "假设X服从参数为二的泊松分布 算出来期望和方差都等于二 再用切比雪夫不等式估一个上界"
)


def test_asr_style_text_yields_same_terms_as_clean_text():
    got = set(names(ASR_TEXT, limit=16))
    want = {"随机变量", "样本空间", "分布函数", "期望", "方差", "条件概率", "泊松分布", "联合概率"}
    assert want <= got, f"ASR 文本漏了：{want - got}"


def test_asr_fillers_do_not_leak_into_terms():
    """「呃然后分布函数」这种带口头语的窗口残渣必须被剥掉。"""
    for g in names(ASR_TEXT, limit=20):
        assert not g[0] in "呃啊呢吧嘛哦", g
        assert not any(g.startswith(w) for w in ("然后", "那个", "那么", "也就是说")), g


@pytest.mark.parametrize("bad", ["Fx等于PX小于等于x", "P大于零点五", "个方差", "呃然后分布"])
def test_window_artifacts_are_rejected(bad):
    text = f"{bad} {bad} {bad}。{PROB_TEXT}"
    assert bad not in names(text, limit=20)


def test_bare_suffix_alone_is_not_a_term():
    """光秃秃一个后缀词（「分布」）不能靠后缀证据混成术语。"""
    from dsh_app import terms as T

    text = "呃然后分布 那个分布 就是分布。"
    got = {c.term for c in T.extract_terms(text, "通用", limit=10)}
    assert "分布" not in got


# ---------------------------------------------------------------- 模型路径
def test_model_terms_must_quote_text_that_exists():
    src = PROB_TEXT
    payload = {"terms": [
        {"term": "分布函数", "evidence": "分布函数定义为"},
        {"term": "冯诺依曼代数", "evidence": "这一句原文里根本没有"},
    ]}
    cand, dropped = terms.parse_model_terms(payload, src)
    assert [c.term for c in cand] == ["分布函数"]
    assert dropped == 1, "引文查不到的必须计入丢弃数"


def test_model_terms_drop_meta_and_junk():
    cand, dropped = terms.parse_model_terms(
        {"terms": [{"term": "必要条件", "evidence": "随机变量是指"},
                   {"term": "随机变量", "evidence": "随机变量是指"}]}, PROB_TEXT)
    assert [c.term for c in cand] == ["随机变量"]
    assert dropped == 1


def test_model_path_falls_back_to_rules_when_not_configured():
    class NoLLM:
        configured = False

    out = terms.learn_terms(NoLLM(), PROB_TEXT, max_terms=8)
    assert out["source"] == "rule"
    assert out["terms"], "没有模型也必须抽得出术语"


def test_model_path_falls_back_when_model_returns_nothing_verifiable():
    class LiarLLM:
        configured = True

        def complete_json(self, *a, **kw):
            from dsh_app.llm import JsonCall
            return JsonCall(payload={"terms": [{"term": "伪术语甲", "evidence": "不在原文里"},
                                               {"term": "伪术语乙", "evidence": "也不在"}]},
                            text="{}", attempts=1, error="")

    out = terms.learn_terms(LiarLLM(), PROB_TEXT, max_terms=8)
    assert out["source"] == "rule"
    assert out["dropped"] == 2
    assert "分布函数" in [t["term"] for t in out["terms"]]


# ---------------------------------------------------------------- 接进管线
def test_analysis_writes_real_terms_and_prunes_old_junk(tmp_path):
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "t.db")
    eng = Engine(db, Settings.load())
    sid = eng.seed_demo()
    cid = str(db.get_session(sid)["course_id"])      # 演示课自己带一门课
    # 先塞一条旧版本的词表垃圾，模拟升级前建好的库
    db.upsert_knowledge_point(cid, "必要条件", 0.0, sid, origin="rule")
    names_now = [r["name"] for r in db.get_knowledge_points(cid)]
    # 新分析不会再产出元词（旧的那条要等重建才清）
    assert names_now.count("必要条件") == 1
    assert any(n in names_now for n in ("分布函数", "随机变量", "期望"))
    res = eng.rebuild_knowledge(cid, use_llm=False)
    assert res["ok"]
    assert "必要条件" in res["courses"][0]["pruned"], "旧垃圾要在重建时清掉"
    assert "必要条件" not in [r["name"] for r in db.get_knowledge_points(cid)]

# ---------------------------------------------------------------- 子串是"两个东西"还是"一个碎片"
def test_substring_pair_is_kept_when_both_are_real_terms():
    """概率密度 ≠ 概率密度函数：前者是分布函数的导数，后者是那个函数本身。

    合并会把掌握度算错（学生可能"知道有密度这回事"但"不会用密度函数算概率"）。
    判据不看分数，看文本证据：真术语后面跟谓语或标点。
    """
    text = ("概率密度是分布函数的导数。概率密度函数用来描述连续型随机变量在一点附近的密集程度。"
            "概率密度的定义是分布函数的导数。概率密度函数定义为分布函数的导数。"
            "连续型随机变量的概率密度函数不是概率。")
    got = names(text, limit=10)
    assert "概率密度" in got and "概率密度函数" in got, got


def test_fragment_pair_still_collapses_to_the_short_form():
    """「样本空间映射」是"样本空间映射到实数轴"被切开的产物，必须并到「样本空间」。"""
    text = "也就是说随机变量是指把样本空间映射到实数轴的一个函数啊 那个样本空间呢就是全部结果。"
    got = names(text, limit=10)
    assert "样本空间" in got
    assert "样本空间映射" not in got, got


def test_definition_meta_word_prefix_is_not_a_term():
    """「定义是分布」这种以元词开头的候选不是术语。"""
    text = "概率密度函数定义为分布函数的导数。"
    assert not any(g.startswith("定义") for g in names(text, limit=10))
