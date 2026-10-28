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
        return _text_blocks(_strip_markup(raw)), "markdown 按标题分小节"
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
        for piece in chunk_text(str(b.get("text") or ""), max_chars, overlap):
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