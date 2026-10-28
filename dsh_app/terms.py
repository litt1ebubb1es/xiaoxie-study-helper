"""知识点抽取：从这一节的文字里找出「真正在讲的术语」。

为什么单独做一层
----------------
原来知识点直接取内置学科词表 ∩ 回看点文本。问题不是"不够多"，而是**取到的不是内容**：
「定义 / 定理 / 必要条件 / 边界条件 / 最优解」这些是**元词**——它们描述"这是哪种命题"，
不描述"这一节教了什么"。把它们当知识点，掌握度算得再准也没有意义，
因为没人会说自己"掌握了必要条件"。

三层证据（确定性，不依赖模型）
----------------------------
1. **定义句式**：`X 是指 / X 称为 / 所谓 X / 记作 X / X 定义为` —— 老师讲一个新概念时
   必须用这类句式，这是最强的"此处引入术语"信号。
2. **术语性后缀**：`-分布 / -定理 / -矩阵 / -函数 / -复杂度 / -空间 …`。
3. **领域词表命中**，但**排除元词**。
三者加权，再乘上出现频次的对数，最后做去重（一个短语不能同时是另一个的片段）。

可选：模型抽取（有端点时）
------------------------
模型给 `{"term":…, "evidence": 原句}`，**原句必须能在本节原文里查到**（复用 quiz 的
证据归一化），查不到就丢掉。模型擅长找"人在教什么"，不擅长被信任——所以只让它提名，
由代码验真。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# 元词：描述"这是哪种命题/哪种逻辑关系"，不是课程内容本身。
# 这些词出现在字面里恰恰说明"附近有术语"，所以特征计分仍用它们（见 features.py），
# 但不能当知识点。
META_TERMS = frozenset("""
定义 定理 引理 推论 公理 命题 假设 反例 归纳 演绎 抽象 范式 框架
充要条件 充分条件 必要条件 边界条件 初始条件 约束 目标函数 最优解 最优 解
近似 误差 量纲 单位 概念 结论 性质 证明 例子 例题 问题 方法 步骤 过程
显著 意义 含义 关系 区别 联系 形式 内容 一般 特殊 情况 结果 答案 思路
""".split())

# 术语性后缀：命中即"看起来像个概念名词"。单字后缀要求整体长度 ≥3，避免「秩」「基」误吸。
SUFFIXES = tuple("""
定理 定律 方程 公式 函数 矩阵 分布 期望 方差 极限 导数 微分 积分 概率 集合 映射
空间 向量 序列 级数 变换 算子 多项式 曲线 系数 不等式 中值 复数 实数 有理数
复杂度 复杂度 结构 算法 协议 模型 框架 接口 事务 索引 缓存 队列 栈 线程 进程
守恒 效应 定律 假说 频数 频率 密度 强度 梯度 散度 旋度 秩 阶 核 域 环 群
""".split())

# 定义句式：抓「X 是指 / 所谓 X / 把 X 称为 / 记作 X」里的 X
_DEF_PATTERNS = (
    re.compile(r"(?:所谓|叫做|称为|称作|定义为|记为|记作|表示)\s*"
               r"(?P<t>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{0,10})"),
    re.compile(r"(?P<t>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{0,10})\s*"
               r"(?:是指|定义为|称为|称作|记作)"),
    # 「X 的定义是…」「X 的含义」——老师换个说法讲同一个概念
    re.compile(r"(?P<t>[一-鿿]{2,10})的(?:定义|含义|概念|意义)"),
    # 「令 X = …」「设 X 表示…」（数学课的固定开场）
    re.compile(r"(?:令|设|记)\s*(?P<t>[一-鿿A-Za-z]{2,8})\s*(?:=|＝|表示|代表)"),
)

_ALNUM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+\-]{2,}\b")
# 英文通用词 / 编程关键字：课上会念到，但不是「这一节在讲的概念」
# 强后缀：出现一次也足以认定是术语（「泊松分布」「切比雪夫不等式」）
SUFFIX_STRONG = frozenset("""
定理 定律 分布 函数 矩阵 方程 公式 不等式 空间 极限 导数 积分 概率 期望 方差
样本 估计 检验 复杂度 协议 算法 结构 映射 变换 算子 多项式
""".split())

_EN_STOP = frozenset("""
self str int float bool list dict tuple set none true false return import from def class
not and or for while if else elif try except print lambda pass break continue with as
model path audio port text args kwargs json http https url api app data file test type
""".split())

# 明显不是术语的片段：口语、连接、指代、虚词头
_JUNK_HEADS = (
    "注意", "这里", "那里", "这个", "那个", "我们", "你们", "大家", "老师", "同学",
    "首先", "其次", "然后", "接着", "下面", "上面", "前面", "后面", "今天", "刚才",
    "如果", "因为", "所以", "但是", "而且", "不过", "其实", "可能", "应该", "可以",
    "需要", "这样", "那样", "什么", "怎么", "一种", "一个", "一次", "一步", "一点",
    "那么", "这么", "是不", "就是", "还是", "于是", "因此", "例如", "比如",
)
_JUNK_TAILS = ("的", "了", "着", "过", "吗", "呢", "吧", "啊", "来说", "而言", "的话", "时候")

# 学科词表粗判：用来选 features 的 subject，也用来给术语加分
SUBJECT_HINTS: dict[str, tuple[str, ...]] = {
    "数学": ("概率", "分布", "期望", "方差", "矩阵", "向量", "导数", "积分", "极限",
             "秩", "行列式", "特征值", "函数", "随机变量", "样本", "估计"),
    "计算机": ("算法", "复杂度", "指针", "线程", "进程", "哈希", "二叉树", "递归",
               "事务", "索引", "缓存", "接口", "编译", "神经网络", "梯度下降"),
    "物理化学": ("力", "加速度", "熵", "焓", "电场", "磁场", "摩尔", "浓度", "折射",
                 "干涉", "波函数", "能级", "动能", "势能"),
    "经济管理": ("边际", "弹性", "供给", "需求", "均衡", "效用", "贴现", "净现值",
                 "成本", "收益", "垄断", "博弈"),
}


@dataclass
class TermCandidate:
    term: str
    score: float = 0.0
    count: int = 0
    why: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"term": self.term, "score": round(self.score, 2),
                "count": self.count, "why": list(self.why)}


def guess_subject(text: str) -> str:
    """按词表命中密度粗判学科。判不出来就说「通用」，不要硬猜。"""
    t = text or ""
    if len(t) < 40:
        return "通用"
    best, best_hits = "通用", 0
    for name, words in SUBJECT_HINTS.items():
        hits = sum(t.count(w) for w in words)
        if hits > best_hits:
            best, best_hits = name, hits
    # 至少 3 次命中才算判出来，否则一次口误就能把学科带偏
    return best if best_hits >= 3 else "通用"


def _subject_terms(subject: str) -> set[str]:
    """该学科的预置词表（通用 + 命中的学科），去掉元词。"""
    from .cognitive.features import BUILTIN_TERMS

    out: set[str] = set()
    for name, words in BUILTIN_TERMS.items():
        if name in ("通用", subject):
            out.update(w for w in words if w not in META_TERMS)
    return out


def _is_junk(term: str, min_len: int = 2) -> bool:
    t = (term or "").strip()
    if len(t) < min_len or len(t) > 12:
        return True
    if t in META_TERMS:
        return True
    if any(t.startswith(h) for h in _JUNK_HEADS):
        return True
    if any(t.endswith(x) for x in _JUNK_TAILS):
        return True
    # 全是虚词/数字/符号
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", t):
        return True
    if min_len >= 2 and re.fullmatch(r"[\u4e00-\u9fff]{0,1}", t):
        return True
    # 放行单字时只放行汉字：「记作 X」「F(x)」里的 X/F/P 是符号，不是概念
    if len(t) == 1 and not re.fullmatch(r"[\u4e00-\u9fff]", t):
        return True
    # 公式不是术语：「Fx等于PX小于等于x」「P大于零点五」是定义正文，不是名字
    if any(w in t for w in _FORMULA_WORDS):
        return True
    if len(t) > 3 and re.search(r"[A-Za-z]", t) and re.search(r"[\u4e00-\u9fff]", t):
        return True
    return False


_LEAD_WORDS = ("另外", "此外", "所以", "因此", "于是", "例如", "比如", "然后", "接着",
               "其次", "并且", "而且", "但是", "不过", "同时", "其中", "这时", "此时",
               "下面", "上面", "前面", "后面", "那么", "这么", "这样", "那样", "这里", "那里")

# 出现在术语里的关系词 → 这是公式/命题，不是术语本身
_FORMULA_WORDS = ("等于", "小于", "大于", "不等于", "不为", "竖杠", "乘以", "除以", "加上")

_TRIM_CHARS = set("，。、；：！？的了是在和就都也把被给让对从到着过们你我他她它这那哪"
                  "呃哦呀嘛诶哎喔噢呗唔嗯啊呢吧哈"
                  "先再又才从出来说讲算得做看想用取求使令设记叫称学写读画跑试举"
                  "什怎吗呢吧啊嗯很有与及或而则其之向为以于由用和与及或得地说讲算")
_BAD_INNER = set("，。、；：！？（）《》【】“”‘’—…·")
# 允许出现在术语左边界的字符：名词短语不会由这些字"接着"上面一个词继续
_BOUNDARY_WS = " " + chr(9) + chr(10) + "　"
_BOUNDARY_BEFORE = set("，。、；：！？（）()「」") | set(_BOUNDARY_WS)
_BOUNDARY_BEFORE |= set("的是把由和与及或在从对将让给为以于个种类又再也都还就而则其之这那")

def _clean_term(term: str) -> str:
    """把候选词两头的虚词/标点削掉，中间带标点或"的"的直接判死。

    「另外条件概率」「，分布函数」「随机变量的加权平均」都是这样来的：
    窗口切出来的片段包含了前一个词的尾字或连接词。术语不可能含"的"。
    """
    t = (term or "").strip()
    if not t:
        return ""
    if any(ch in _BAD_INNER for ch in t):
        return ""
    # 术语内部不会有"的/和/与/及/或/并"——出现就说明窗口跨了两个词
    if any(ch in t for ch in "的和与及或并把"):
        return ""
    # 以元词开头（「定义是分布」「定理的内容」）不是术语本身
    if t.startswith(tuple(META_TERMS)):
        return ""
    if any(ch.isspace() for ch in t):      # 术语不可能跨词跨行
        return ""
    # 先剥掉开头的连接词/副词：「另外条件概率」「所以分布函数」这类都是左窗口多吃了一个词。
# 必须和两头的虚词剥除**交替**做到不动为止：ASR 文本里常是「呃然后分布函数」，
# 先剥「呃」才露出「然后」，只做一轮就漏了。
    changed = True
    while changed and t:
        changed = False
        for w in _LEAD_WORDS:
            if t.startswith(w) and len(t) - len(w) >= 2:
                t = t[len(w):]
                changed = True
                break
        if changed:
            continue
        before = t
        i, j = 0, len(t)
        while i < j and t[i] in _TRIM_CHARS:
            i += 1
        while j > i and t[j - 1] in _TRIM_CHARS:
            j -= 1
        t = t[i:j]
        if t != before:
            changed = True
    # 量词残留：「个方差」「些分布函数」→ 削掉。只削 个/些，
    # 别的一律不动——「条」在「条件」前面、「种」在「种类」前面。
    while len(t) >= 3 and t[0] in "个些":
        t = t[1:]
    return t


# 碎片判据：术语后面紧跟这些字，说明这句话还没说完 —— 窗口是被硬切开的
_CONTINUE_AFTER = set("到为于向成至进")


def _is_fragment(text: str, term: str) -> bool:
    """这个词在原文里是不是"句子被切开"的产物。

    「样本空间映射」在"样本空间映射到实数轴"里，后面永远跟「到」——碎片；
    「概率密度函数」在"概率密度函数是分布函数的导数"里，后面跟「是」——真术语。
    """
    if not term or term not in (text or ""):
        return False
    nxt: list[str] = []
    start = 0
    while True:
        i = text.find(term, start)
        if i < 0:
            break
        j = i + len(term)
        nxt.append(text[j] if j < len(text) else "")
        start = j
    if not nxt:
        return False
    # 只要有一次出现在正常位置（后面是标点/空白/句末），它就不是碎片
    return all((c in _CONTINUE_AFTER) for c in nxt)


def _has_suffix(term: str) -> str:
    for s in SUFFIXES:
        if len(s) >= 2 and term.endswith(s) and len(term) >= len(s) + 1:
            return s
    return ""


def _definition_terms(text: str) -> set[str]:
    """定义句式里的候选。这里放行单字术语——「所谓鞅，是指…」里的「鞅」就是术语。"""
    out: set[str] = set()
    for pat in _DEF_PATTERNS:
        for m in pat.finditer(text or ""):
            t = _clean_term(m.group("t"))
            if t and not _is_junk(t, min_len=1):
                out.add(t)
    return out


def extract_terms(text: str, subject: str = "通用", limit: int = 40,
                  known: set[str] | None = None) -> list[TermCandidate]:
    """从一段课程文字里抽术语，按"像不像知识点"排序。

    ``known`` 是内置学科词表命中项（features 那边算的），命中即加高分。

    打分只做一次：先把各路线索各自收成集合，再给每个候选算一遍分。
    （早期版本按 n-gram 窗口逐个累加，同一个词被加了三五次，排序全被频次带跑。）
    """
    t = text or ""
    if len(t) < 30:
        return []
    known = {k for k in (known or set()) if k in t and k not in META_TERMS}
    defined = _definition_terms(t)

    # 3) 术语性后缀：按"后缀出现位置"取左侧 2–4 字，而不是扫描所有 n-gram
    suffix_hits: dict[str, str] = {}
    for sfx in SUFFIXES:
        if len(sfx) < 2:
            continue
        for m in re.finditer(re.escape(sfx), t):
            j = m.start()
            for k in range(2, 5):
                i0 = max(0, j - k)
                cand = t[i0:m.end()]
                # 光秃秃一个后缀不算术语（「分布」这三个字要自己找到别处证据），
                # 否则「呃然后分布」这类窗口残渣会靠后缀混进来
                if not cand.endswith(sfx) or len(cand) <= len(sfx) or len(cand) > 6:
                    continue
                # 左边界必须是标点/空白/句首/虚词，否则窗口切在了词中间
                if i0 > 0 and t[i0 - 1] not in _BOUNDARY_BEFORE:
                    continue
                if not _is_junk(cand):
                    suffix_hits.setdefault(cand, sfx)

    alnum = {m.group(0) for m in _ALNUM_RE.finditer(t)
             if m.group(0).lower() not in _EN_STOP}

    cands: dict[str, TermCandidate] = {}
    for term, sources in (
        *((x, ["定义句式"]) for x in defined),
        *((x, ["学科词表"]) for x in known),
        *((x, [f"术语后缀「{sfx}」"]) for x, sfx in suffix_hits.items()),
        *((x, ["英文术语"]) for x in alnum),
    ):
        term = _clean_term(term)
        if not term:
            continue
        from_definition = sources == ["定义句式"]
        if _is_junk(term, min_len=1 if from_definition else 2):
            continue
        c = cands.setdefault(term, TermCandidate(term))
        c.count = t.count(term)
        c.why.extend(sources)

    out: list[TermCandidate] = []
    for c in cands.values():
        weight = 0.0
        for why in c.why:
            weight += {"定义句式": 3.0, "学科词表": 2.5, "英文术语": 1.2}.get(why, 1.6)
        # 只出现一次的多半是转写噪声。三个例外：
        # ① 被明确定义过（「所谓鞅，是指…」）；② 命中学科词表；
        # ③ 带强后缀（「泊松分布」「切比雪夫不等式」这种出现一次也是术语）。
        strong_suffix = any(w.startswith("术语后缀「") and
                            w[len("术语后缀「"):-1] in SUFFIX_STRONG for w in c.why)
        if c.count < 2 and not ({"定义句式", "学科词表"} & set(c.why)) and not strong_suffix:
            continue
        weight += 0.8 * math.log2(1 + c.count)
        if len(c.term) >= 4:
            weight += 0.3
        c.score = weight
        out.append(c)
    out.sort(key=lambda x: (-x.score, -len(x.term), x.term))

    # 去重：互为子串时**优先留短的那个**（规范形式），但要区分两种"长":
    # ① 长的是**另一个东西**——「概率密度」和「概率密度函数」是两个知识点，
    #    前者是函数的性质、后者是函数本身，合并会把掌握度算错；
    # ② 长的只是**句子碎片**——「样本空间映射」来自"样本空间映射到实数轴"。
    # 判据看文本证据而不看分数：碎片几乎总是被续接词（到/为/于/向/成）跟着，
    # 真术语后面跟的是谓语（是/的/可以/表示）或标点。
    kept: list[TermCandidate] = []
    for c in out:
        host = next((k for k in kept if c.term in k.term and c.term != k.term), None)
        if host is not None:
            if _is_fragment(text, host.term):
                kept.remove(host)          # 碎片：短的才是术语
            # 否则长的留着——两个都算知识点
            continue
        if any(c.term != k.term and k.term in c.term for k in kept) and _is_fragment(text, c.term):
            # 已经留着它的规范形式，而自己只是被切开的碎片 → 让位
            continue
        kept.append(c)
    kept.sort(key=lambda x: (-x.score, -len(x.term), x.term))
    return kept[:limit]


def evidence_key(s: str) -> str:
    """证据比对用的归一化：去掉行号行、空白、连字符。与 quiz 那边同一条规则。"""
    lines = [ln for ln in (s or "").splitlines() if not re.fullmatch(r"\s*\d+\s*", ln)]
    return re.sub(r"[\s\-—_·.,，。、]+", "", "".join(lines)).lower()


def grounded(quote: str, source_text: str) -> bool:
    """这条引文是不是真的出自原文。"""
    q = evidence_key(quote)
    if len(q) < 4:
        return False
    return q in evidence_key(source_text)


SYSTEM = (
    "你从一段中文课程转写里找出老师**真正在讲的概念术语**。规则：\n"
    "① 只找这一节里被引入或反复使用的专业概念，不要元词"
    "（定义、定理、必要条件、边界条件、最优解、结论 这类不算）。\n"
    "② 每个术语必须给出它出现的**原句**（照抄转写，不改字、不加标点）。\n"
    "③ 术语 2–10 字；不要短语、不要整句、不要人称和时间。\n"
    "④ 最多 12 个。只输出 JSON："
    '{"terms":[{"term":"…","evidence":"…"}]}'
)


def parse_model_terms(payload, source_text: str) -> tuple[list[TermCandidate], int]:
    """校验模型给的术语：evidence 必须能在原文里查到，查不到就丢。

    返回 (候选, 被丢弃数)。丢弃数要报出来——它是"模型质量"的唯一可见指标。
    """
    rows = []
    if isinstance(payload, dict):
        rows = payload.get("terms") or []
    elif isinstance(payload, list):
        rows = payload
    out: list[TermCandidate] = []
    dropped = 0
    for r in rows:
        if isinstance(r, str):
            term, ev = r.strip(), ""
        elif isinstance(r, dict):
            term = str(r.get("term") or "").strip()
            ev = str(r.get("evidence") or "").strip()
        else:
            dropped += 1
            continue
        if _is_junk(term) or term in META_TERMS:
            dropped += 1
            continue
        if not grounded(ev, source_text):
            dropped += 1
            continue
        out.append(TermCandidate(term, score=3.5, count=source_text.count(term),
                                 why=["模型提名+原文可查"]))
    # 同一批里去掉互为子串的
    kept: list[TermCandidate] = []
    for c in sorted(out, key=lambda x: (-len(x.term), x.term)):
        if any(c.term in k.term for k in kept):
            continue
        kept.append(c)
    return kept, dropped


def learn_terms(llm, source_text: str, *, max_terms: int = 12,
                timeout_s: float = 40.0) -> dict:
    """模型提名术语（可选路径）。任何异常都退回确定性抽取，不阻塞分析。"""
    src = source_text or ""
    subject = guess_subject(src)
    det = extract_terms(src, subject, limit=max_terms, known=_subject_terms(subject))
    if not getattr(llm, "configured", False) or len(src) < 60:
        return {"terms": [c.as_dict() for c in det], "source": "rule", "dropped": 0}
    try:
        import threading

        box: dict = {}

        def _run() -> None:
            try:
                call = llm.complete_json(SYSTEM, f"课程转写：\n{src[:6000]}",
                                         expect_key="terms", max_tokens=900)
                box["payload"] = call.payload
                box["error"] = call.error
            except Exception as e:  # noqa: BLE001
                box["error"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive() or not box.get("payload"):
            return {"terms": [c.as_dict() for c in det], "source": "rule",
                    "dropped": 0, "model_error": str(box.get("error") or "超时")[:120]}
        cand, dropped = parse_model_terms(box["payload"], src)
        if not cand:
            return {"terms": [c.as_dict() for c in det], "source": "rule",
                    "dropped": dropped,
                    "model_error": "模型给的术语一条都没通过原文校验"}
        return {"terms": [c.as_dict() for c in cand[:max_terms]], "source": "llm",
                "dropped": dropped}
    except Exception as e:  # noqa: BLE001
        return {"terms": [c.as_dict() for c in det], "source": "rule", "dropped": 0,
                "model_error": f"{type(e).__name__}: {e}"[:120]}


__all__ = [
    "META_TERMS", "SUFFIXES", "SUBJECT_HINTS", "TermCandidate", "guess_subject",
    "extract_terms", "evidence_key", "grounded", "parse_model_terms", "learn_terms",
]