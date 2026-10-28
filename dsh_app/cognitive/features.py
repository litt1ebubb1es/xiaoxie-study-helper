"""认知层 A：认知负荷计算（方案 5.2.1）。

五个特征全部来自转写文本与时间戳，不调用任何模型（ADR-001 检测与理解分离）：

| 特征 | 符号 | 本实现 |
|------|------|--------|
| 术语密度 | TD | 窗口内「课程级首次出现」的领域术语数 ÷ 百字 |
| 语速偏移 | SS | 窗口字速 ÷ 全课平均字速 |
| 句长     | SL | 窗口内平均分句字数（标点 + ≥300ms 停顿切分） |
| 符号密度 | SD | 数学符号 / 代码标识 / 公式片段命中数 ÷ 百字 |
| 停顿结构 | PS | 窗口内 ≥1.5s 长停顿次数（每分钟） |

三处对方案的裁决（原文未定义，见 docs/算法说明.md）：

1. **以「字」为计量单位**。方案说"总词数""分句长度"，但中文分词结果不稳定且会引入
   额外依赖。z-score 是尺度无关的，改用字不影响归一化后的判别力，且完全确定。
2. **sigmoid 前加缩放温度 τ**。原文 `L = sigmoid(0.35·TD + …)` 直接吃 z-score 加权和：
   五特征 z 值加权和后标准差约 0.49，sigmoid 会把绝大多数窗口压到 0 或 1，退化成硬阈值。
   本实现除以 τ（默认 0.5）后再过 sigmoid，使 L 在课程内保持可分辨的梯度。
3. **术语表三级来源**：内置学科词表 + 课程内自动 n-gram 抽取 + 英文/缩写正则。
   原文只写"领域词表加词性规则"而没说词表从哪来；纯规则无法覆盖任意学科。
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# 内置领域术语表（三级来源之一）。按学科粗分，命中即计入术语。
# --------------------------------------------------------------------------
BUILTIN_TERMS: dict[str, list[str]] = {
    "通用": [
        "定义", "定理", "引理", "推论", "公理", "命题", "充要条件", "充分条件", "必要条件",
        "假设", "反例", "归纳", "演绎", "抽象", "范式", "模型", "框架", "收敛", "发散",
        "变量", "参数", "常量", "边界条件", "初始条件", "约束", "目标函数", "最优解",
        "近似", "误差", "量纲", "单位", "假设检验", "置信区间", "显著性", "方差", "均值",
    ],
    "数学": [
        "导数", "偏导", "梯度", "散度", "旋度", "积分", "微分", "极限", "连续", "可微",
        "矩阵", "向量", "特征值", "特征向量", "行列式", "秩", "线性无关", "正交", "转置",
        "逆矩阵", "概率", "条件概率", "期望", "分布", "正态分布", "抽样", "极大似然",
        "贝叶斯", "先验", "后验", "似然", "积分中值定理", "洛必达", "泰勒展开", "傅里叶",
    ],
    "计算机": [
        "算法", "数据结构", "时间复杂度", "空间复杂度", "递归", "迭代", "指针", "栈", "队列",
        "哈希", "红黑树", "二叉树", "图论", "动态规划", "贪心", "分治", "回溯", "剪枝",
        "进程", "线程", "协程", "死锁", "互斥", "信号量", "虚拟内存", "页表", "中断",
        "编译", "语法分析", "词法分析", "抽象语法树", "虚拟机", "字节码", "垃圾回收",
        "接口", "多态", "继承", "封装", "设计模式", "依赖注入", "事务", "索引", "范式",
        "神经网络", "反向传播", "梯度下降", "损失函数", "过拟合", "正则化", "注意力机制",
        "卷积", "池化", "嵌入", "向量化", "推理", "训练集", "验证集", "超参数",
    ],
    "经济管理": [
        "边际", "弹性", "机会成本", "沉没成本", "供给", "需求", "均衡", "效用", "贴现",
        "现金流", "折现率", "净现值", "内部收益率", "资产负债", "利润表", "毛利率",
        "规模效应", "边际成本", "边际收益", "垄断", "外部性", "博弈", "纳什均衡",
    ],
    "物理化学": [
        "力", "加速度", "动量", "角动量", "势能", "动能", "熵", "焓", "自由能", "平衡态",
        "理想气体", "热力学第一定律", "热力学第二定律", "电场", "磁场", "通量", "场强",
        "折射", "干涉", "衍射", "波函数", "能级", "跃迁", "摩尔", "浓度", "反应速率",
    ],
}

# 符号密度：数学符号 + 代码标识 + 公式片段（方案 5.2.1 的 SD）
_SYMBOL_RE = re.compile(
    r"(?:[=+\-*/^<>≤≥≠≈≡∝∞∑∏∫∮√∛∂∇∆ΔπαβγδεζηθικλμνξοπρστυφχψωΓΘΛΞΠΣΦΨΩ%‰])"
    r"|(?:[A-Za-z_][A-Za-z0-9_]{1,}\s*\()"     # 函数调用 / 代码标识
    r"|(?:\b\d+(?:\.\d+)?\s*(?:%|次方|倍|毫秒|秒|分钟|小时|元|万元|亿|万|GB|MB|KB|ms))\b"
    r"|(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+)"    # snake_case
    r"|(?:\b[A-Z][a-z]+[A-Z][A-Za-z]*\b)"        # CamelCase
)

_CLAUSE_RE = re.compile(r"[^。！？；;!?\n]+[。！？；;!?]?")
_ALNUM_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+\-]{2,}\b")

_STOPWORDS = set(
    """的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
    自己 这 那 这个 那个 我们 他们 什么 怎么 因为 所以 但是 如果 然后 现在 可以 这样
    时候 问题 东西 地方 事情 觉得 应该 需要 可能 其实 就是 还是 已经 一些 这些 那些
    大家 老师 同学 下面 上面 左边 右边 第一 第二 第三 我们 你们 咱们 之间 以后 以前
    这里 那里 这样 那样 那么 这么 注意 来看 看到 一下 一点 一直 一样 一起 一般
    首先 其次 然后 最后 接着 下面 上面 前面 后面 左边 右边 里面 外面 中间
    对我们 对你 对我 来说 而言 的话 之类 等等 是否 以及 并且 或者 而是 不过
    这种 那种 各种 有些 有的 这些 那些 什么样 怎么办 行不行 对不对 是不是""".split()
)

# 口语/连接词的判据：n-gram 的首字或整体命中即丢弃。
# 这是"知识点列出现『注意』『这里涉及』"这类垃圾的直接原因。
_JUNK_CHARS = set("的了是在和就都也把被给让对从到着过们你我他她它这那哪什怎吗呢吧啊嗯")
_JUNK_HEADS = (
    "注意", "这里", "那里", "这个", "那个", "我们", "你们", "大家", "老师", "同学",
    "首先", "其次", "然后", "接着", "下面", "上面", "前面", "后面", "今天", "刚才",
    "如果", "因为", "所以", "但是", "而且", "不过", "其实", "可能", "应该", "可以",
    "需要", "这样", "那样", "什么", "怎么", "一种", "一个", "一次", "一步", "一点",
)
_JUNK_TAILS = ("的", "了", "着", "过", "吗", "呢", "吧", "啊", "来说", "而言", "的话")

_PUNCT = set("，。！？；：、（）《》【】“”‘’—…·,.;:!?()[]{}<>\"' \t\n\r")


# --------------------------------------------------------------------------
# 术语表
# --------------------------------------------------------------------------
@dataclass
class TermLexicon:
    terms: set[str] = field(default_factory=set)
    sources: dict[str, str] = field(default_factory=dict)  # term -> builtin|auto|alnum

    @property
    def reliable(self) -> set[str]:
        """可信术语：学科词表命中或英文/缩写。自动 n-gram 不算可信。

        知识点（F4 的核心产出）只允许来自可信集合；自动 n-gram 仅用于 TD 特征计分。
        """
        return {t for t, src in self.sources.items() if src in ("builtin", "alnum")}

    @classmethod
    def build(cls, course_text: str, subject: str = "通用",
              auto_min_count: int = 3, auto_max_terms: int = 400) -> "TermLexicon":
        lex = cls()
        for name, words in BUILTIN_TERMS.items():
            if name in ("通用", subject):
                for w in words:
                    lex.terms.add(w)
                    lex.sources[w] = "builtin"
        # 英文术语 / 缩写
        for m in _ALNUM_TERM_RE.finditer(course_text or ""):
            w = m.group(0)
            if w.lower() in _STOPWORDS:
                continue
            lex.terms.add(w)
            lex.sources.setdefault(w, "alnum")
        # 课程内自动 n-gram 抽取：高频、非停用词、非纯虚词
        auto = _extract_ngrams(course_text or "", auto_min_count, auto_max_terms)
        for w, _c in auto:
            if w not in lex.terms:
                lex.terms.add(w)
                lex.sources[w] = "auto"
        return lex

    def __len__(self) -> int:
        return len(self.terms)

    def first_occurrences(self, text: str) -> dict[str, int]:
        """课程级首次出现的字符偏移。"""
        out: dict[str, int] = {}
        if not text:
            return out
        for term in self.terms:
            i = text.find(term)
            if i >= 0:
                out[term] = i
        return out


def _extract_ngrams(text: str, min_count: int, max_terms: int) -> list[tuple[str, int]]:
    """2–5 字的高频 n-gram 作为候选术语。

    不引入分词器：中文专业术语多为 2–5 字复合词，高频重复的 n-gram 召回足够好，
    而且完全确定、可离线。缺点是会把常见搭配也抽进来——z-score 归一化下影响有限。
    """
    if not text:
        return []
    counts: dict[str, int] = {}
    n_max = 5
    for n in range(2, n_max + 1):
        for i in range(0, len(text) - n + 1):
            gram = text[i:i + n]
            if any(ch in _PUNCT for ch in gram):
                continue
            if gram in _STOPWORDS:
                continue
            # 首尾是虚词/口语，几乎不可能是术语
            if gram[0] in _JUNK_CHARS or gram[-1] in _JUNK_CHARS:
                continue
            if gram.startswith(_JUNK_HEADS) or gram.endswith(_JUNK_TAILS):
                continue
            # 含"注意/这里/我们"等口语标记的 gram 一律不要
            if any(h in gram for h in ("注意", "这里", "那里", "我们", "你们", "大家")):
                continue
            counts[gram] = counts.get(gram, 0) + 1
    # 只保留「极大」n-gram：若更长的 gram 同样高频，短的就没有信息量
    cand = [(g, c) for g, c in counts.items() if c >= min_count]
    cand.sort(key=lambda x: (-x[1] * len(x[0]), -len(x[0])))
    picked: list[tuple[str, int]] = []
    for g, c in cand:
        if len(picked) >= max_terms:
            break
        if any(g in p for p, _ in picked):
            continue
        picked.append((g, c))
    return picked


# --------------------------------------------------------------------------
# 转写流：把分段文本映射到「字符 → 时间」
# --------------------------------------------------------------------------
class Transcript:
    """把带时间戳的分段文本摊平成逐字符时间轴。

    用途：任意时刻查询"这一秒在讲哪些字"，从而把标点/符号/术语的时间位置精确落位。
    中文 45 分钟课约 1.5–2 万字，逐字符数组完全可接受（<1 MB）。
    """

    def __init__(self, segments: Sequence[dict]):
        self.segments = [s for s in segments if float(s.get("t_end", 0)) > float(s.get("t_start", 0))]
        self.text = ""
        self.times: list[float] = []
        # 逐字符来源：语音转写 vs 屏幕识别。混合会话里同一时间轴上有两路内容，
        # 措辞和信号语义必须按窗口分开——不能用整节的标签糊过去。
        self.kinds: list[str] = []
        self._seg_spans: list[tuple[int, int, float, float]] = []
        self.gaps: list[tuple[float, float]] = []  # (gap_start, gap_len)

        parts: list[str] = []
        prev_end: float | None = None
        for seg in self.segments:
            t0, t1 = float(seg["t_start"]), float(seg["t_end"])
            txt = str(seg.get("text", ""))
            if not txt:
                continue
            if prev_end is not None and t0 - prev_end >= 1.5:
                self.gaps.append((prev_end, t0 - prev_end))
            prev_end = t1 if prev_end is None else max(prev_end, t1)
            start_idx = sum(len(p) for p in parts)
            parts.append(txt)
            self._seg_spans.append((start_idx, start_idx + len(txt), t0, t1))
            n = len(txt)
            dur = max(1e-6, t1 - t0)
            self.times.extend(t0 + (i / n) * dur for i in range(n))
            kind = str(seg.get("_kind") or "speech")
            self.kinds.extend([kind] * n)
            # 段间插入分隔符，避免术语跨段误匹配
            parts.append("\n")
            self.times.append(t1)
            self.kinds.append("speech")
        self.text = "".join(parts)
        self.duration = float(self.segments[-1]["t_end"]) if self.segments else 0.0

    # -- 查询 -----------------------------------------------------------
    def chars_in(self, t0: float, t1: float) -> int:
        return max(0, bisect_right(self.times, t1) - bisect_left(self.times, t0))

    def screen_share(self, t0: float, t1: float) -> float:
        """窗口内来自屏幕识别的字符占比（0..1）。"""
        a = bisect_left(self.times, t0)
        b = bisect_right(self.times, t1)
        if b <= a:
            return 0.0
        seg = self.kinds[a:b]
        if not seg:
            return 0.0
        return sum(1 for k in seg if k == "screen") / float(len(seg))

    def text_in(self, t0: float, t1: float) -> str:
        return self.text[bisect_left(self.times, t0): bisect_right(self.times, t1)]

    def speech_seconds(self, t0: float, t1: float) -> float:
        total = 0.0
        for seg in self.segments:
            a, b = float(seg["t_start"]), float(seg["t_end"])
            lo, hi = max(a, t0), min(b, t1)
            if hi > lo:
                total += hi - lo
        return total

    def long_pauses_in(self, t0: float, t1: float, threshold: float = 1.5) -> int:
        return sum(1 for st, ln in self.gaps if ln >= threshold and t0 <= st < t1)


# --------------------------------------------------------------------------
# 特征抽取
# --------------------------------------------------------------------------
@dataclass
class Window:
    idx: int
    t_start: float
    t_end: float
    text: str
    feats: dict[str, float] = field(default_factory=dict)
    z: dict[str, float] = field(default_factory=dict)
    l_score: float = 0.0
    b_score: float = 0.0
    s_score: float = 0.0
    grade: str = ""
    quadrant: str = ""
    valid: bool = True
    content_kind: str = "speech"   # speech | screen | mixed（决定 PS 与措辞的语义）

    @property
    def center(self) -> float:
        return (self.t_start + self.t_end) / 2


FEATURE_KEYS = ("TD", "SS", "SL", "SD", "PS")
FEATURE_CN_SCREEN = {
    "TD": "新概念密集",
    "SS": "屏幕文字密度高",
    "SL": "长句/长段落难解析",
    "SD": "公式与符号密集",
    "PS": "页面长时间没有更新",
}
FEATURE_CN = {
    "TD": "术语密度高",
    "SS": "语速明显加快",
    # 屏幕内容的同义替换（见 FEATURE_CN_SCREEN）
    
    "SL": "长句难解析",
    "SD": "符号/公式密集",
    "PS": "长停顿多（思考或板书）",
}


def make_windows(tr: Transcript, window_s: float = 60.0, step_s: float = 15.0) -> list[Window]:
    dur = tr.duration
    if dur <= 0:
        return []
    wins: list[Window] = []
    starts = list(np.arange(0.0, max(0.0, dur - window_s) + 1e-6, step_s))
    if not starts:
        starts = [0.0]
    tail_end = dur
    if starts[-1] + window_s < dur - 1e-6:
        starts.append(max(0.0, dur - window_s))
    for i, st in enumerate(starts):
        en = min(dur, st + window_s) if st + window_s > dur else st + window_s
        if i == len(starts) - 1:
            en = max(en, tail_end) if tail_end - st >= step_s * 0.5 else en
        wins.append(Window(idx=i, t_start=float(st), t_end=float(en), text=tr.text_in(st, en)))
    return wins


def extract_features(wins: Sequence[Window], tr: Transcript, lex: TermLexicon,
                     course_text: str | None = None) -> None:
    """就地写入每个窗口的原始特征（方案 5.2.1 的五个特征）。"""
    text = course_text if course_text is not None else tr.text
    firsts = lex.first_occurrences(text)
    # 术语首次出现 → 时间
    term_first_time: list[float] = [tr.times[i] for i in firsts.values() if i < len(tr.times)]

    # 符号命中 → 时间
    sym_times = [tr.times[m.start()] for m in _SYMBOL_RE.finditer(text) if m.start() < len(tr.times)]

    # 分句：按标点切，且长停顿也算分句边界
    clause_times: list[tuple[float, int]] = []  # (中点时间, 字数)
    for m in _CLAUSE_RE.finditer(text):
        piece = m.group(0).strip()
        if len(piece) < 2:
            continue
        a, b = m.start(), m.end() - 1
        if a >= len(tr.times) or b >= len(tr.times):
            continue
        mid = (tr.times[a] + tr.times[min(b, len(tr.times) - 1)]) / 2
        clause_times.append((mid, len(piece)))

    # 全课平均字速
    total_speech = sum(max(0.0, float(s["t_end"]) - float(s["t_start"])) for s in tr.segments)
    total_chars = max(1, len(text.replace("\n", "")))
    mean_cps = total_chars / total_speech if total_speech > 0 else 0.0

    # 屏幕内容没有"停顿"这回事：连续截图的相邻间隔就是"这页停留了多久"。
    # 用一个足够长的停留阈值（90s）近似"卡在这一页"，语义上接近语音的长停顿。
    dwell_marks: list[float] = []
    if any(getattr(w, "content_kind", "speech") in ("screen", "mixed") for w in wins):
        # 屏幕笔记的每段跨度 = 这页停留了多久；跨度过长说明一直卡在同一屏
        for seg in tr.segments:
            if float(seg["t_end"]) - float(seg["t_start"]) >= 90.0:
                dwell_marks.append(float(seg["t_start"]))

    term_times_sorted = sorted(term_first_time)
    for w in wins:
        # 这一窗口里屏幕内容占多数时才用屏幕语义（密度/停留），否则仍按语音说
        share = tr.screen_share(w.t_start, w.t_end)
        w.content_kind = "screen" if share >= 0.4 else "speech"
        chars = max(0, tr.chars_in(w.t_start, w.t_end))
        speech = tr.speech_seconds(w.t_start, w.t_end)
        cps = chars / speech if speech > 1e-6 else 0.0
        lo = bisect_left(term_times_sorted, w.t_start)
        hi = bisect_left(term_times_sorted, w.t_end)
        n_term = hi - lo
        lo = bisect_left(sym_times, w.t_start)
        hi = bisect_left(sym_times, w.t_end)
        n_sym = hi - lo
        lens = [ln for (mid, ln) in clause_times if w.t_start <= mid < w.t_end]
        mean_clause = float(np.mean(lens)) if lens else 0.0
        is_screen = getattr(w, "content_kind", "speech") in ("screen", "mixed")
        if is_screen and dwell_marks:
            # 屏幕内容：用"这页停留超过 90s"的次数替代语音停顿次数
            lo = bisect_left(dwell_marks, w.t_start)
            hi = bisect_left(dwell_marks, w.t_end)
            pauses = float(hi - lo)
        else:
            pauses = tr.long_pauses_in(w.t_start, w.t_end)
        span_min = max(1e-6, (w.t_end - w.t_start) / 60.0)

        w.feats = {
            "TD": _per100(n_term, chars),
            "SS": (cps / mean_cps) if mean_cps > 1e-9 else 0.0,
            "SL": mean_clause,
            "SD": _per100(n_sym, chars),
            "PS": pauses / span_min,
            "_chars": float(chars),
            "_speech": float(speech),
            "_n_term": float(n_term),
            "_n_sym": float(n_sym),
            "_n_pause": float(pauses),
        }
        w.valid = chars >= 8 and speech >= 2.0


def _per100(n: int, chars: int) -> float:
    return (n * 100.0 / chars) if chars > 0 else 0.0


def znorm_windows(wins: Sequence[Window], floor_std: float = 1e-6) -> dict[str, tuple[float, float]]:
    """课程内 z-score（方案 5.2.1："以本课程自身的分布为基准"）。"""
    valid = [w for w in wins if w.valid] or list(wins)
    stats: dict[str, tuple[float, float]] = {}
    for k in FEATURE_KEYS:
        vals = np.array([w.feats.get(k, 0.0) for w in valid], dtype=float)
        mu = float(vals.mean()) if vals.size else 0.0
        sd = float(vals.std()) if vals.size else 0.0
        if sd < floor_std:
            sd = 1.0
        stats[k] = (mu, sd)
    for w in wins:
        w.z = {k: (w.feats.get(k, 0.0) - stats[k][0]) / stats[k][1] for k in FEATURE_KEYS}
    return stats


def load_scores(
    wins: Sequence[Window],
    weights: dict[str, float],
    tau: float = 0.5,
    bias: float = 0.0,
) -> None:
    """L = sigmoid((Σ wᵢ·zᵢ) / τ + b)。τ 与 b 是配置项，可被校准覆盖。"""
    tau = max(1e-6, tau)
    for w in wins:
        raw = sum(weights.get(k, 0.0) * w.z.get(k, 0.0) for k in FEATURE_KEYS) + bias
        w.l_score = _sigmoid(raw / tau)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def explain_window(w: Window, top_k: int = 2) -> list[str]:
    """可解释依据：哪个特征把这段判成高负荷（方案 5.2.7 最后一行）。"""
    contrib = [(k, float(w.z.get(k, 0.0))) for k in FEATURE_KEYS]
    contrib.sort(key=lambda x: -x[1])
    screen = getattr(w, "content_kind", "speech") in ("screen", "mixed")
    names = dict(FEATURE_CN)
    if screen:
        names.update(FEATURE_CN_SCREEN)
    out = []
    for k, z in contrib[:top_k]:
        if z <= 0.3:
            continue
        detail = ""
        f = w.feats
        if k == "TD":
            detail = f"（{int(f.get('_n_term', 0))} 个新术语）"
        elif k == "SS":
            detail = (f"（屏幕文字 {f.get('SS', 0):.2f}× 全节均值）" if screen
                      else f"（语速 {f.get('SS', 0):.2f}× 全课均值）")
        elif k == "SL":
            detail = f"（平均句长 {f.get('SL', 0):.1f} 字）"
        elif k == "SD":
            detail = f"（{int(f.get('_n_sym', 0))} 处符号/公式）"
        elif k == "PS":
            detail = (f"（{int(f.get('_n_pause', 0))} 处长时间没更新）" if screen
                      else f"（{int(f.get('_n_pause', 0))} 次长停顿）")
        out.append(names[k] + detail)
    return out


def default_weights(cfg) -> dict[str, float]:
    return {
        "TD": cfg.w_term,
        "SS": cfg.w_speed,
        "SL": cfg.w_sentence,
        "SD": cfg.w_symbol,
        "PS": cfg.w_pause,
    }