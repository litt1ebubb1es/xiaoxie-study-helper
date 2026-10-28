"""对话窗口：把"回看点 + 知识库 + 你的问题"组织成一次问答（方案 3.3 的延伸）。

设计原则：
1. **依据必须可追溯**：回答里用到的知识库片段要带文件名和序号，界面上能点开看原文；
   没有检索到就说没有，不许让模型自由发挥成"我查过了"。
2. **不知道就说不知道**：课里没讲、知识库里也没有的问题，允许模型回答，但必须标注
   "这是模型自己的知识，不是你的课程内容"。
3. 对话按会话存库，换节不串味。
"""
from __future__ import annotations

from typing import Sequence

from . import kb as kbmod

SYSTEM = (
    "你是「小蟹学习助手」的答疑助教。规则：\n"
    "① 优先依据【课程内容】和【你的知识库】回答；这两块没提到的，可以用通用知识回答，"
    "但必须在开头注明『以下不属于你的课程/知识库，是通用知识』。\n"
    "② 不要编造课程里没有的内容，不要说『老师在视频里提到过』这类无法核实的话。\n"
    "③ 不替用户完成作业或给出可直接提交的答案；可以讲思路、指出关键步骤。\n"
    "④ 中文口语化，先给结论，再给解释；数学式子用常见的线性写法。\n"
    "⑤ 如果问题本身有歧义，就直接问清楚，不要猜。"
)


def session_context(db, session_id: str, max_knots: int = 6) -> str:
    """把当前这一节的关键信息压成一段，喂给模型。"""
    if not session_id:
        return ""
    sess = db.get_session(session_id)
    if sess is None:
        return ""
    course = ""
    try:
        row = db.get_course(sess["course_id"])
        course = str(row["title"]) if row else ""
    except Exception:  # noqa: BLE001
        pass
    lines = [f"课程：{course or '未命名'}", f"节次：{sess['title'] or '未命名'}",
             f"时长：{float(sess['duration'] or 0) / 60:.0f} 分钟"]
    knots = db.get_knots(session_id) or []
    if knots:
        lines.append("已识别出的回看点（时间｜等级｜依据）：")
        for k in knots[:max_knots]:
            # get_knots 返回 KnotRow 对象（不是 dict），必须按属性取
            t = float(k.t_start)
            lines.append(f"- {int(t // 60):02d}:{int(t % 60):02d}｜{k.grade}｜"
                         f"{(k.evidence or '')[:80]}")
    else:
        lines.append("本节还没有分析出回看点。")
    return "\n".join(lines)


REWRITE_SYSTEM = (
    "把用户的口语提问改写成适合字面检索（BM25）的关键词串。规则：\n"
    "① 只输出关键词，用空格分隔，不要句子、不要标点解释、不要引号。\n"
    "② 把指代词（这个/那个/它/上面/刚才/这一步）替换成它指代的具体对象——"
    "如果上下文里看不出来，就保留原词。\n"
    "③ 保留学科术语原样，最多 12 个词。\n"
    "④ 只输出一行结果。"
)


def rewrite_query(llm, query: str, context: str = "", timeout_s: float = 5.0) -> str:
    """把指代型提问改写成可检索的词串。**fail-open：任何异常都返回原问题。**

    为什么要 fail-open：改写是"提高召回"的加分项，不是正确性前提。
    模型慢、没配、答非所问，都不该让用户连原来的检索结果都拿不到。
    """
    q = (query or "").strip()
    if not q:
        return q
    if not getattr(llm, "configured", False):
        return q
    user = f"上下文：\n{context[:800]}\n\n用户提问：{q}" if context else f"用户提问：{q}"
    try:
        # 用线程加超时：改写慢就放弃，不拖住整次回答
        import threading

        box: dict = {}

        def _run() -> None:
            try:
                call = llm.complete_text(REWRITE_SYSTEM, user, max_tokens=80)
                box["text"] = getattr(call, "text", "") or ""
            except Exception as e:  # noqa: BLE001
                box["err"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive() or "text" not in box:
            return q
        out = " ".join(str(box.get("text") or "").split())
        out = out.strip("`'\"。.：: 　")
        if not out or len(out) > 120:
            return q
        # 改写等于原文说明没起到作用，但也不算错——照样用它（能加同义词就好）
        return out
    except Exception:  # noqa: BLE001
        return q


def kb_search(db, query: str, course_id: str = "", k: int = 4) -> list[dict]:
    """在本课程 + 全局知识库里检索，返回带来源（含页码/时间码）的片段。"""
    rows = db.kb_chunks(course_id=course_id) or []
    if not rows:
        return []
    index, items = kbmod.index_chunks(rows)
    hits = index.search(query, k)
    out = []
    for i, score in hits:
        r = items[i]
        d = dict(r)     # sqlite3.Row → dict，避免下游按属性/key 混用出错
        out.append({"doc": d.get("title", ""), "ord": int(d.get("ord") or 0),
                    "score": round(float(score), 3),
                    "page": int(d.get("page") or 0),
                    "section": str(d.get("section") or ""),
                    "source_ref": str(d.get("source_ref") or ""),
                    "label": kbmod.cite_label(d),
                    "text": str(d.get("text") or ""),
                    "snippet": kbmod.snippet(str(d.get("text") or ""), query)})
    return out


def build_messages(question: str, context: str, hits: Sequence[dict],
                   history: Sequence[dict] | None = None,
                   max_turns: int = 6, images: Sequence[str] = ()) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    for h in list(history or [])[-max_turns * 2:]:
        role = h["role"] if isinstance(h, dict) else h["role"]
        content = h["content"] if isinstance(h, dict) else h["content"]
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": str(content)})
    parts = []
    if context:
        parts.append("【课程内容】\n" + context)
    if hits:
        block = ["【你的知识库（检索到的最相关片段）】"]
        nl = chr(10)
        for i, h in enumerate(hits, 1):
            # 定位串优先：能写「p.12」就不写「第 3 段」——前者能回原文，后者回不去
            where = h.get("label") or f"第 {h['ord']} 段"
            # 标签本身带书名号时补一个间隔符，避免出现《文件》《小节》连在一起的怪样子
            sep = "" if not where.startswith("《") else chr(183)
            block.append(f"[{i}] 来源《{h['doc']}》{sep}{where}（相关度 {h['score']}）"
                         + nl + f"{h['text'][:800]}")
        parts.append("\n\n".join(block))
    parts.append("【用户的问题】\n" + question.strip())
    text = "\n\n".join(parts)
    pics = [u for u in images if u]
    if not pics:
        msgs.append({"role": "user", "content": text})
        return msgs
    # 带图的消息用 OpenAI 兼容的多段格式：先文字，再逐张图。
    # 图片如果用文本占位代替，模型只能"猜图里写了什么"——必须真发 image_url。
    content: list[dict] = [{"type": "text", "text": text}]
    for u in pics[:4]:
        content.append({"type": "image_url", "image_url": {"url": u}})
    msgs.append({"role": "user", "content": content})
    return msgs


def ask(db, llm, session_id: str, question: str, use_kb: bool = True,
        use_context: bool = True, k: int = 4, images: Sequence[str] = (),
        files: Sequence[tuple[str, str]] = ()) -> dict:
    """问一次。返回 {answer, error, sources, context_len}。

    images：base64 data URL 列表（截图/照片，走视觉）；
    files：[(文件名, 正文)] 文档内容，作为额外资料塞进提示词。
    只有图片、没打字时也允许发送——"这张图看不懂"本身就是问题。
    """
    q = (question or "").strip()
    if not q and (images or files):
        q = "看这些内容：讲讲重点、有没有错、容易误解的地方在哪。"
    if not q:
        return {"answer": "", "error": "问题是空的", "sources": []}
    sess = db.get_session(session_id) if session_id else None
    course_id = str(sess["course_id"]) if sess else ""
    hits: list[dict] = []
    used_query = q
    if use_kb:
        # 只有当问题本身检索不到时才用回看点术语兜底：BM25 是字面匹配，
        # 用户换个说法就搜不到，把本节回看点的知识点并进查询能明显提高召回。
        hits = kb_search(db, q, course_id, k)
        if not hits and session_id:
            slog = session_context(db, session_id)
            hits = kb_search(db, (q + " " + slog)[:600], course_id, k)
        if not hits:
            # 仍然空 → 把指代型提问改写一次再搜（「这个」「它」这类词 BM25 检索不到）。
            # fail-open：改写失败就保持原样，用户至少还能拿到"没检索到"这个诚实的结论。
            ctx = session_context(db, session_id) if session_id else ""
            alt = rewrite_query(llm, q, ctx)
            if alt and alt != q:
                hits = kb_search(db, alt, course_id, k)
                used_query = alt
    ctx = session_context(db, session_id) if (use_context and session_id) else ""
    # 附件正文当成额外的资料块接到课程内容后面：它同样是有出处的依据，
    # 但要标明「来自用户附件」，别和知识库检索结果混在一起。
    attach_block = ""
    for name, body in files:
        if body.strip():
            attach_block += f"【用户附件《{name}》】\n{body.strip()}\n\n"
    if attach_block:
        ctx = (ctx + "\n\n" if ctx else "") + attach_block.strip()
    history = db.get_chat(session_id, limit=12) if session_id else []
    msgs = build_messages(q, ctx, hits, history, images=images)
    label = q
    if files:
        label += "　[附件：" + "、".join(n for n, _ in files) + "]"
    if images:
        label += f"　[图片 ×{len(images)}]"
    db.add_chat(session_id, "user", label)
    if not getattr(llm, "configured", False):
        return {"answer": "", "error": "未配置模型端点（设置页填 base_url 与 api_key）",
                "sources": hits, "query_used": used_query}
    # 走统一重试规则，但保留完整消息列表（历史 + system），不能只发最后一条
    from .llm import HARDEN_HINT, call_with_retry

    def _run(hardened: bool):
        m = list(msgs)
        if hardened:
            m = [{"role": "system", "content": m[0]["content"] + "\n" + HARDEN_HINT}] + m[1:]
        return llm.chat(m, max_tokens=1400 if hardened else 1200)

    text, err, attempts = call_with_retry(_run, lambda s: bool((s or "").strip()))
    if err:
        return {"answer": "", "error": err, "sources": hits, "query_used": used_query}
    db.add_chat(session_id, "assistant", text)
    return {"answer": text, "error": "",
            "query_used": used_query,
            "retried": attempts > 1,
            "sources": [{"doc": h["doc"], "ord": h["ord"], "score": h["score"],
                         "page": h.get("page", 0), "section": h.get("section", ""),
                         "source_ref": h.get("source_ref", ""),
                         "label": h.get("label", ""),
                         "snippet": h["snippet"]} for h in hits],
            "context_len": len(ctx)}