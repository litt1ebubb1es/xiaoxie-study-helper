"""知识库：导入用户自己的资料，检索出相关片段当"解释依据"（方案 4.5 的延伸）。

为什么不用向量库：
- 端侧场景下装 embedding 模型 + 建索引成本高，而用户资料通常只有几 MB；
- 中文按"字二元组 + 英文词"做 BM25，对术语/公式/专有名词的召回已经够用；
- 全部离线可算，不需要任何 API——没有模型时知识库照样能检索、能把片段拼进提纲。

检索结果一定带来源（文件名 + 片段序号），便于界面上标注"这句依据来自你导入的某文件"。
"""
from __future__ import annotations

import math
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

MAX_CHUNK = 600
OVERLAP = 60

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".srt", ".vtt", ".text", ".log", ".json", ".csv"}
CODE_SUFFIXES = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sql"}


# ---------------------------------------------------------------- 读取
def _strip_subtitles(text: str) -> str:
    """去掉 srt/vtt 的时间轴与序号，只留台词。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+$", s):
            continue
        if "-->" in s or s.upper().startswith(("WEBVTT", "NOTE ")):
            continue
        out.append(s)
    return "\n".join(out)


def _strip_markup(text: str) -> str:
    """粗去 markdown 标记，保留文字。代码块保留内容。"""
    text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`\n"), text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # 图片
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # 链接留文字
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # 标题
    text = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", text)
    return text


def _docx_text(path: Path) -> str:
    """docx 是 zip：正文在 word/document.xml。不引入额外依赖。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return xml


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "读 PDF 需要 pypdf：.venv\\Scripts\\python.exe -m pip install pypdf"
        ) from e
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def read_document(path: str | Path) -> tuple[str, str]:
    """返回 (正文, 说明)。说明里写清做了什么处理，便于界面如实展示。"""
    blocks, note = read_document_blocks(path)
    return "\n\n".join(b["text"] for b in blocks), note


# ---------------------------------------------------------------- 定位块
def _mk(text: str, page: int = 0, section: str = "", ref: str = "") -> dict:
    return {"text": text.strip(), "page": int(page or 0),
            "section": section.strip(), "source_ref": ref.strip()}


def _md_heading(line: str) -> str:
    m = re.match(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$", line)
    return m.group(2).strip() if m else ""


def _text_blocks(text: str) -> list[dict]:
    """按 markdown 标题切段，每段记住自己属于哪个小节。"""
    out: list[dict] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        if body:
            out.append(_mk(body, 0, section, f"《{section}》" if section else ""))
        buf = []

    for line in text.splitlines():
        head = _md_heading(line)
        if head:
            flush()
            section = head
            continue
        buf.append(line)
    flush()
    return out


def _md_blocks(text: str) -> list[dict]:
    """markdown 先按标题切、再逐段去掉标记。

    顺序不能反：_strip_markup 会把 # 开头的标题行一并删掉，先清理就再也切不出小节了。
    """
    out: list[dict] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = _strip_markup("\n".join(buf)).strip()
        if body:
            out.append(_mk(body, 0, section, f"《{section}》" if section else ""))
        buf = []

    for line in text.splitlines():
        head = _md_heading(line)
        if head:
            flush()
            section = head
            continue
        buf.append(line)
    flush()
    return out


def _docx_blocks(path: Path) -> list[dict]:
    """docx 是 zip：正文在 word/document.xml。按段落拆，识别标题样式做小节。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    out: list[dict] = []
    section = ""
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        style = re.search(r'<w:pStyle w:val="([^"]+)"', para)
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.S))
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue
        name = style.group(1) if style else ""
        if "eading" in name or name.lower() in ("title", "subtitle", "1", "2", "3"):
            section = text
            continue
        out.append(_mk(text, 0, section, f"《{section}》" if section else ""))
    return out


def _pptx_blocks(path: Path) -> list[dict]:
    """pptx 也是 zip：每张幻灯片一个 slideN.xml，天然带页号。

    比按字数硬切有意义得多——引用能直接说「幻灯片 7」。
    """
    out: list[dict] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
        names.sort(key=lambda n: int(re.findall(r"(\d+)", n.split("/")[-1])[0]))
        for i, name in enumerate(names, 1):
            xml = z.read(name).decode("utf-8", "ignore")
            texts = [re.sub(r"<[^>]+>", "", t).strip()
                     for t in re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S)]
            body = "\n".join(t for t in texts if t)
            if body:
                out.append(_mk(body, i, "", f"幻灯片 {i}"))
    return out


_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def _hms(sec: int) -> str:
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" \
        if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


def _subtitle_blocks(text: str) -> list[dict]:
    """字幕按条切，每条带起始时间码——最精确的定位（能直接回到那一句）。"""
    out: list[dict] = []
    ref = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = " ".join(buf).strip()
        if body:
            out.append(_mk(body, 0, "", ref))
        buf = []

    for line in text.splitlines():
        s = line.strip()
        m = _SRT_TIME.search(s)
        if m:
            flush()
            ref = _hms(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
            continue
        if not s or re.match(r"^\d+$", s) or s.upper().startswith(("WEBVTT", "NOTE ")):
            continue
        buf.append(s)
    flush()
    return out


def _pdf_blocks(path: Path) -> list[dict]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "读 PDF 需要 pypdf：.venv" + chr(92) + "Scripts" + chr(92) +
            "python.exe -m pip install pypdf"
        ) from e
    out: list[dict] = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        try:
            body = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            body = ""
        if body:
            out.append(_mk(body, i, "", f"p.{i}"))
    return out


def read_document_blocks(path: str | Path) -> tuple[list[dict], str]:
    """读成「带定位的块」：每块带 page / section / source_ref。

    这是按页索引的基础——引用必须能回到原处，否则界面上的「依据」只是装饰。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        blocks = _pdf_blocks(p)
        return blocks, f"pdf 逐页文本层抽取（{len(blocks)} 页有文字；扫描件无效）"
    if suffix == ".docx":
        return _docx_blocks(p), "docx 段落抽取（按标题样式分小节）"
    if suffix == ".pptx":
        blocks = _pptx_blocks(p)
        return blocks, f"pptx 按幻灯片抽取（{len(blocks)} 张有文字）"
    raw = p.read_text(encoding="utf-8", errors="replace")
    if suffix in {".srt", ".vtt"}:
        blocks = _subtitle_blocks(raw)
        return blocks, f"字幕按条目切分（{len(blocks)} 条，带时间码）"
    if suffix in {".md", ".markdown"}:
        return _md_blocks(raw), "markdown 按标题分小节"
    if suffix in TEXT_SUFFIXES or suffix in CODE_SUFFIXES:
        return _text_blocks(raw), "纯文本按标题分小节"
    return _text_blocks(raw), "按纯文本读取（未知后缀）"


def chunk_blocks(blocks: Iterable[dict], max_chars: int = MAX_CHUNK,
                 overlap: int = OVERLAP) -> list[dict]:
    """在每个定位块内部切块，块继承它所属的 page / section / source_ref。

    关键点：**绝不让一个块跨页**。跨页的块无法回答「这句话在第几页」。
    """
    out: list[dict] = []
    for b in blocks:
        # min_chars=2：幻灯片标题这种短文本本身就是一个完整知识点，不能按长文标准丢掉
        for piece in chunk_text(str(b.get("text") or ""), max_chars, overlap, 2):
            out.append({"text": piece, "page": int(b.get("page") or 0),
                        "section": str(b.get("section") or ""),
                        "source_ref": str(b.get("source_ref") or "")})
    return out


def cite_label(row: dict) -> str:
    """把手里的定位字段拼成一句人话，给界面/引用列表用。"""
    ref = str(row.get("source_ref") or "").strip()
    if ref:
        return ref
    page = int(row.get("page") or 0)
    if page:
        return f"p.{page}"
    section = str(row.get("section") or "").strip()
    if section:
        return f"《{section}》"
    return ""


# ---------------------------------------------------------------- 切块
def chunk_text(text: str, max_chars: int = MAX_CHUNK,
               overlap: int = OVERLAP, min_chars: int = 8) -> list[str]:
    """按段落聚合成大小合适的块，块间留一点重叠避免切断上下文。"""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        # 单段就超长：按句子再切
        if len(para) > max_chars:
            sents = re.split(r"(?<=[。！？；\.\!\?;])\s*", para)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                while len(s) > max_chars:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    chunks.append(s[:max_chars])
                    s = s[max_chars - overlap:]
                if len(buf) + len(s) + 1 > max_chars and buf:
                    chunks.append(buf)
                    buf = buf[-overlap:] if overlap else ""
                buf = (buf + ("\n" if buf else "") + s).strip()
            continue
        if len(buf) + len(para) + 2 > max_chars and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf = (buf + "\n\n" if buf else "") + para
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c.strip()) >= max(1, int(min_chars))]


# ---------------------------------------------------------------- 检索
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{1,}")


def tokens(s: str) -> list[str]:
    """中文用字二元组（无需分词），英文/数字用整词。"""
    s = unicodedata.normalize("NFKC", s).lower()
    out: list[str] = []
    for run in _CJK.findall(s):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    out.extend(_WORD.findall(s))
    return out


class BM25:
    """标准 BM25（k1=1.5, b=0.75），语料就是知识库的块。"""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = list(docs)
        self.toks = [tokens(d) for d in self.docs]
        self.lens = [len(t) for t in self.toks]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        self.df: dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.n = len(self.docs)
        self.tf: list[dict[str, int]] = []
        for t in self.toks:
            m: dict[str, int] = {}
            for w in t:
                m[w] = m.get(w, 0) + 1
            self.tf.append(m)

    def idf(self, w: str) -> float:
        df = self.df.get(w, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        qs = tokens(query)
        if not qs or not self.docs:
            return []
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * (self.lens[i] / (self.avg or 1)))
            for w in qs:
                f = tf.get(w, 0)
                if not f:
                    continue
                s += self.idf(w) * (f * (self.k1 + 1)) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        # 归一化到 0..1，便于界面显示"相关度"
        top = scores[0][1] if scores else 1.0
        return [(i, s / top) for i, s in scores[:k]]


def snippet(text: str, query: str, width: int = 120) -> str:
    """抽包含查询词的片段做预览。"""
    q = tokens(query)
    best = 0
    low = text.lower()
    for w in q[:20]:
        pos = low.find(w.lower())
        if pos >= 0:
            best = max(0, pos - width // 3)
            break
    piece = text[best:best + width].replace("\n", " ")
    return ("…" if best else "") + piece + ("…" if best + width < len(text) else "")


def index_chunks(chunks: Iterable[dict]) -> tuple[BM25, list[dict]]:
    rows = list(chunks)
    return BM25([str(r["text"]) for r in rows]), rows