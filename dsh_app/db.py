"""存储层：SQLite 单机单用户（方案 5.4 数据模型）。

相对方案原表，这里补了四处缺口（理由见 docs/算法说明.md §数据模型补全）：
- courses：原方案 sessions.course_id 引用了它却没有定义。
- words：方案 5.3 要求词级时间戳对齐到数十毫秒，原模型只有段级边界，无处存放。
- windows：认知负荷的滑动窗口中间结果，原方案只存 segments.load_score，无法重算/调试。
- knowledge_edges：方案 3.5 要求"按前置依赖排序"，那需要一张有向图，原模型只有节点。

knots 表额外保存 l_score / b_score / grade / quadrant，并新增 feedback.action 枚举。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import DB_PATH, ensure_dirs

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS courses (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    subject      TEXT DEFAULT '',
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    course_id    TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    idx          INTEGER DEFAULT 0,
    title        TEXT NOT NULL,
    audio_path   TEXT DEFAULT '',
    started_at   REAL NOT NULL,
    duration     REAL DEFAULT 0,
    source       TEXT DEFAULT 'loopback',   -- loopback | file | demo
    state        TEXT DEFAULT 'recorded',   -- recorded | transcribed | analyzed | failed
    error        TEXT DEFAULT '',
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    text         TEXT NOT NULL,
    speaker      TEXT DEFAULT '',
    load_score   REAL DEFAULT 0,
    pass_no      INTEGER DEFAULT 1,         -- 1=实时流式, 2=课后精转（方案 5.1）
    confidence   REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_seg_session ON segments(session_id, t_start);

-- 屏幕内容（方案 3.4 F3）：截图经视觉模型识别的结果。
-- 与语音转写分开存：两者语义不同（语速 vs 页面停留），混在一张表里会互相污染。
CREATE TABLE IF NOT EXISTS screen_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t          REAL NOT NULL,
    text       TEXT NOT NULL,
    kind       TEXT DEFAULT 'slide',   -- slide|board|notes|code|formula|empty|unknown
    model      TEXT DEFAULT '',
    source     TEXT DEFAULT 'vision',  -- vision | manual
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_screen_session ON screen_notes(session_id, t);

-- 知识库：用户自己导入的资料（讲义、论文、教材、代码、字幕…）。
-- 切块后存 kb_chunks，检索用 BM25（离线可算，不需要向量模型）。
CREATE TABLE IF NOT EXISTS kb_docs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id  TEXT DEFAULT '',        -- 空串=全局，对本门课所有节可见
    title      TEXT NOT NULL,
    path       TEXT DEFAULT '',
    note       TEXT DEFAULT '',        -- 导入时做了什么处理（如"字幕去时间轴"）
    chars      INTEGER DEFAULT 0,
    n_chunks   INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    INTEGER NOT NULL REFERENCES kb_docs(id) ON DELETE CASCADE,
    course_id TEXT DEFAULT '',
    ord       INTEGER NOT NULL,
    title     TEXT DEFAULT '',
    text      TEXT NOT NULL,
    page      INTEGER DEFAULT 0,          -- PDF 页码 / PPT 页号，0=无页码信息
    section   TEXT DEFAULT '',            -- 章节标题（markdown 标题 / 文档大纲）
    source_ref TEXT DEFAULT ''            -- 精确定位串：如 "p.12"、"00:03:12"、"幻灯片 7"
);
CREATE INDEX IF NOT EXISTS idx_kb_course ON kb_chunks(course_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_doc ON kb_chunks(doc_id);

-- 自动抓取被丢掉的短抓取：浏览器提示音/广告前奏/误点播放。
-- 留着是样本：攒够一批就能反推"这台机器上多短才算误触发"，阈值不用拍脑袋。
CREATE TABLE IF NOT EXISTS capture_discards (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         REAL NOT NULL,          -- unix 时间
    proc       TEXT DEFAULT '',        -- 触发进程（msedge/chrome…）
    site       TEXT DEFAULT '',        -- 认出来的站点
    title      TEXT DEFAULT '',        -- 当时拿到的页面标题
    duration   REAL DEFAULT 0,         -- 录了多久
    threshold  REAL DEFAULT 0,         -- 当时的阈值（事后才知道它该不该丢）
    reason     TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_discard_at ON capture_discards(at);

-- 对话历史（按会话隔离，换节不串味）
CREATE TABLE IF NOT EXISTS chat_msgs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    role       TEXT NOT NULL,          -- user | assistant
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_msgs(session_id, id);

CREATE TABLE IF NOT EXISTS words (
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seg_id       INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    text         TEXT NOT NULL,
    pass_no      INTEGER DEFAULT 1,
    PRIMARY KEY (session_id, seg_id, idx)
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t            REAL NOT NULL,             -- 播放进度轴（方案 5.3：以播放进度为主轴）
    kind         TEXT NOT NULL,
    payload_json TEXT DEFAULT '{}',
    confidence   REAL DEFAULT 1.0,
    source       TEXT DEFAULT 'manual'      -- manual | player | audio_fp | ui_infer
);
CREATE INDEX IF NOT EXISTS idx_evt_session ON events(session_id, t);

CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    l_score      REAL NOT NULL,
    b_score      REAL NOT NULL,
    s_score      REAL NOT NULL,
    grade        TEXT DEFAULT '',
    quadrant     TEXT DEFAULT '',
    feats_json   TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_win_session ON windows(session_id, idx);

CREATE TABLE IF NOT EXISTS knots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    level        TEXT NOT NULL,             -- explicit | implicit | structural
    quadrant     TEXT DEFAULT '',
    grade        TEXT DEFAULT '',
    score        REAL NOT NULL,
    l_score      REAL DEFAULT 0,
    b_score      REAL DEFAULT 0,
    evidence     TEXT DEFAULT '',
    evidence_json TEXT DEFAULT '{}',
    status       TEXT DEFAULT 'pending',    -- pending | explained | mastered | dismissed
    excerpt      TEXT DEFAULT '',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knot_session ON knots(session_id, t_start);

CREATE TABLE IF NOT EXISTS knowledge_points (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id    TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    first_seen_t REAL DEFAULT 0,
    session_ids  TEXT DEFAULT '[]',
    occurrences  INTEGER DEFAULT 0,
    kp_type      TEXT DEFAULT 'concept',   -- concept | memory | procedure
    mastery      REAL DEFAULT 0.0,         -- 0..1，由 kp_attempts 按近因加权算出
    attempts     INTEGER DEFAULT 0,
    gate         REAL DEFAULT 0.9,         -- 放行门槛
    last_evidence_at REAL DEFAULT 0,
    origin       TEXT DEFAULT '',       -- 术语来源：llm / rule / manual
    retired      INTEGER DEFAULT 0,     -- 已退役：术语规则改过，现在没有任何回看点了
    stability    REAL DEFAULT 0,      -- 记忆稳定度（天），FSRS 简化的那个 S
    due_at       REAL DEFAULT 0,      -- 到期复习时间（epoch 秒）
    UNIQUE(course_id, name)
);

-- 每次"这个知识点我会不会"的证据。掌握度可以随时由它重算，不是算完就丢的中间值。
CREATE TABLE IF NOT EXISTS kp_attempts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kp_id     INTEGER NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    at        REAL NOT NULL,
    correct   REAL DEFAULT 0.0,            -- 1=对 0=错，0.5=部分对
    source    TEXT DEFAULT '',             -- confirm_useful | mastered | self_test | manual
    knot_id   INTEGER DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_kp_attempt ON kp_attempts(kp_id, at);

CREATE TABLE IF NOT EXISTS knot_points (
    knot_id      INTEGER NOT NULL REFERENCES knots(id) ON DELETE CASCADE,
    kp_id        INTEGER NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    weight       REAL DEFAULT 1.0,
    PRIMARY KEY (knot_id, kp_id)
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    src_id       INTEGER NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    dst_id       INTEGER NOT NULL REFERENCES knowledge_points(id) ON DELETE CASCADE,
    kind         TEXT DEFAULT 'co_occur',   -- prerequisite | co_occur
    weight       REAL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, kind)
);

-- 回看点小测：题从「这一段原文」出，evidence 必须能回查到原文，否则作废、不写掌握度
CREATE TABLE IF NOT EXISTS quiz_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    knot_id       INTEGER DEFAULT NULL,
    kp_id         INTEGER DEFAULT NULL,
    prompt        TEXT NOT NULL,
    choices_json  TEXT NOT NULL DEFAULT '[]',
    correct_index INTEGER NOT NULL DEFAULT 0,
    evidence      TEXT DEFAULT '',
    evidence_ok   INTEGER DEFAULT 0,
    void_reason   TEXT DEFAULT '',
    answered_index INTEGER DEFAULT NULL,
    answered_at   REAL DEFAULT 0,
    source        TEXT DEFAULT 'llm',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quiz_session ON quiz_items(session_id, created_at);

-- 老师发的练习册：不定长、和具体回看点无关，所以单独一张表
CREATE TABLE IF NOT EXISTS exercises (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id     TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    kp_id         INTEGER DEFAULT NULL,
    kp_name       TEXT DEFAULT '',
    prompt        TEXT NOT NULL,
    choices_json  TEXT NOT NULL DEFAULT '[]',
    answer_index  INTEGER NOT NULL DEFAULT 0,
    explain       TEXT DEFAULT '',
    kind          TEXT DEFAULT '',
    source        TEXT DEFAULT 'import',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ex_course ON exercises(course_id, created_at);

-- 待答问题队列：听课途中先攒着（带时间戳），课后集中答。对应 ADR-004「静默运行、课后交付」
CREATE TABLE IF NOT EXISTS pending_questions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_sec         REAL DEFAULT 0,
    text          TEXT NOT NULL,
    excerpt       TEXT DEFAULT '',
    status        TEXT DEFAULT 'pending',   -- pending | answered | dropped
    answer        TEXT DEFAULT '',
    created_at    REAL NOT NULL,
    answered_at   REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pq_session ON pending_questions(session_id, status);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    knot_id      INTEGER NOT NULL REFERENCES knots(id) ON DELETE CASCADE,
    action       TEXT NOT NULL,             -- confirm_useful | not_helpful | mastered | dismissed | replay_opened
    ts           REAL NOT NULL,
    note         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS calibrations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    n_samples    INTEGER NOT NULL,
    weights_json TEXT NOT NULL,
    loss_before  REAL,
    loss_after   REAL,
    note         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS settings_kv (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS deploy_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    action       TEXT NOT NULL,
    target       TEXT NOT NULL,
    detail       TEXT DEFAULT ''
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class KnotRow:
    id: int
    session_id: str
    t_start: float
    t_end: float
    level: str
    quadrant: str
    grade: str
    score: float
    l_score: float
    b_score: float
    evidence: str
    status: str
    excerpt: str
    evidence_json: dict = field(default_factory=dict)


class DB:
    """极薄的一层 DAO。单用户单机，不开连接池。"""

    def __init__(self, path: Path | None = None):
        ensure_dirs()
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 单连接多线程访问：分析、重讲、对话、自动抓屏都会在不同线程里写库。
        # sqlite3 的 commit 不是原子的，两个线程交错 commit 会报
        # "cannot commit - no transaction is active"，所以统一加一把可重入锁。
        self._lock = threading.RLock()
        self._open()

    def _open(self) -> None:
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_columns()
        self.conn.commit()

    def close(self) -> None:
        """关连接。备份恢复要先把库文件换掉，Windows 上文件被占用就换不了。"""
        with self._lock:
            try:
                self.conn.commit()
            except Exception:              # noqa: BLE001
                pass
            self.conn.close()

    def reopen(self) -> None:
        self._open()

    # 老库是 CREATE TABLE IF NOT EXISTS 建的，加新列不会自动生效。
    # 这里显式补列，保证用户已有的 data/app.db 也能升级而不丢数据。
    _ADD_COLUMNS: dict[str, dict[str, str]] = {
        "knowledge_points": {
            "kp_type": "TEXT DEFAULT 'concept'",
            "mastery": "REAL DEFAULT 0.0",
            "attempts": "INTEGER DEFAULT 0",
            "gate": "REAL DEFAULT 0.9",
            "last_evidence_at": "REAL DEFAULT 0",
            "origin": "TEXT DEFAULT ''",
            "stability": "REAL DEFAULT 0",
            "due_at": "REAL DEFAULT 0",
            "retired": "INTEGER DEFAULT 0",
        },
        "kb_chunks": {
            "page": "INTEGER DEFAULT 0",
            "section": "TEXT DEFAULT ''",
            "source_ref": "TEXT DEFAULT ''",
        },
        "quiz_items": {
            "source": "TEXT DEFAULT 'llm'",
        },
        "sessions": {
            "video_url": "TEXT DEFAULT ''",
        },
    }

    def _migrate_columns(self) -> None:
        for table, cols in self._ADD_COLUMNS.items():
            try:
                have = {r["name"] for r in
                        self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.Error:
                continue
            if not have:
                continue
            for col, decl in cols.items():
                if col in have:
                    continue
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                except sqlite3.Error:
                    pass

    # -- 通用 -----------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params))

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        with self._lock:
            self.conn.executemany(sql, seq)
            self.conn.commit()

    # -- 课程 / 会话 -----------------------------------------------------
    def get_course(self, course_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM courses WHERE id=?", (course_id,))
        return rows[0] if rows else None
    def ensure_course(self, title: str, subject: str = "") -> str:
        row = self.query("SELECT id FROM courses WHERE title=?", (title,))
        if row:
            return row[0]["id"]
        cid = new_id("course")
        self.execute(
            "INSERT INTO courses(id,title,subject,created_at) VALUES(?,?,?,?)",
            (cid, title, subject, time.time()),
        )
        return cid

    def create_session(
        self,
        course_id: str,
        title: str,
        audio_path: str = "",
        source: str = "loopback",
        idx: int | None = None,
    ) -> str:
        sid = new_id("sess")
        if idx is None:
            row = self.query("SELECT COUNT(*) c FROM sessions WHERE course_id=?", (course_id,))
            idx = int(row[0]["c"]) + 1
        self.execute(
            """INSERT INTO sessions(id,course_id,idx,title,audio_path,started_at,source,state,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (sid, course_id, idx, title, audio_path, time.time(), source, "recorded", time.time()),
        )
        return sid

    def update_session(self, sid: str, **kw: Any) -> None:
        if not kw:
            return
        keys = ", ".join(f"{k}=?" for k in kw)
        self.execute(f"UPDATE sessions SET {keys} WHERE id=?", (*kw.values(), sid))

    def get_session(self, sid: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM sessions WHERE id=?", (sid,))
        return rows[0] if rows else None

    def count_knots(self, course_id: str | None = None) -> int:
        """这门课一共有多少个回看点（左栏统计用）。"""
        if not course_id:
            return 0
        row = self.query(
            """SELECT COUNT(*) AS c FROM knots k
               JOIN sessions s ON s.id = k.session_id WHERE s.course_id=?""",
            (course_id,))
        return int(row[0]["c"] if row else 0)

    def list_sessions(self, course_id: str | None = None) -> list[sqlite3.Row]:
        if course_id:
            return self.query(
                "SELECT * FROM sessions WHERE course_id=? ORDER BY idx", (course_id,)
            )
        return self.query("SELECT * FROM sessions ORDER BY created_at DESC")

    def delete_session(self, sid: str) -> None:
        """删一节，连带它的段/词/事件/窗口/回看点/屏幕笔记/对话/小测/待答。

        知识点不跟着删：它按课程共享，可能被别的节用着。
        删完之后用 `prune_orphan_points(course_id)` 把没人挂的清理掉。
        """
        s = str(sid)
        for table in ("segments", "words", "events", "windows", "screen_notes",
                      "chat_msgs", "quiz_items", "pending_questions"):
            self.execute(f"DELETE FROM {table} WHERE session_id=?", (s,))
        self.execute(
            "DELETE FROM knot_points WHERE knot_id IN (SELECT id FROM knots WHERE session_id=?)",
            (s,))
        self.execute("DELETE FROM feedback WHERE knot_id IN "
                     "(SELECT id FROM knots WHERE session_id=?)", (s,))
        self.execute("DELETE FROM knots WHERE session_id=?", (s,))
        self.execute("DELETE FROM sessions WHERE id=?", (s,))

    # -- 自动抓取的丢弃记录 ---------------------------------------------
    def add_capture_discard(self, proc: str = "", site: str = "", title: str = "",
                            duration: float = 0.0, threshold: float = 0.0,
                            reason: str = "") -> int:
        cur = self.conn.execute(
            """INSERT INTO capture_discards(at,proc,site,title,duration,threshold,reason)
               VALUES(?,?,?,?,?,?,?)""",
            (time.time(), str(proc), str(site), str(title)[:200],
             float(duration), float(threshold), str(reason)))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def capture_discards(self, limit: int = 200) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM capture_discards ORDER BY at DESC LIMIT ?", (int(limit),))

    def discard_stats(self, days: int = 30) -> dict:
        """最近 N 天丢了多少、都多长。设置页拿它算建议阈值。"""
        since = time.time() - max(1, int(days)) * 86400
        rows = self.query(
            "SELECT duration, threshold FROM capture_discards WHERE at>=?", (since,)) or []
        durs = sorted(float(r["duration"] or 0) for r in rows)
        n = len(durs)
        median = durs[n // 2] if n else 0.0
        return {
            "n": n, "median_s": round(median, 1),
            "max_s": round(durs[-1], 1) if durs else 0.0,
            "days": int(days),
            "current_s": round(float(rows[0]["threshold"] or 0), 1) if rows else 0.0,
        }

    def clear_capture_discards(self) -> None:
        self.conn.execute("DELETE FROM capture_discards")
        self.conn.commit()

    def list_courses(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM courses ORDER BY created_at DESC")

    # -- 转写 -----------------------------------------------------------
    def replace_segments(self, session_id: str, segs: Iterable[dict], pass_no: int = 1) -> int:
        self.execute(
            "DELETE FROM segments WHERE session_id=? AND pass_no=?", (session_id, pass_no)
        )
        n = 0
        for i, s in enumerate(segs):
            cur = self.execute(
                """INSERT INTO segments(session_id,idx,t_start,t_end,text,speaker,load_score,pass_no,confidence)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    i,
                    float(s["t_start"]),
                    float(s["t_end"]),
                    str(s.get("text", "")),
                    str(s.get("speaker", "")),
                    float(s.get("load_score", 0.0)),
                    pass_no,
                    float(s.get("confidence", 1.0)),
                ),
            )
            seg_id = int(cur.lastrowid)
            for j, w in enumerate(s.get("words", []) or []):
                self.execute(
                    """INSERT INTO words(session_id,seg_id,idx,t_start,t_end,text,pass_no)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        session_id,
                        seg_id,
                        j,
                        float(w.get("t_start", s["t_start"])),
                        float(w.get("t_end", s["t_end"])),
                        str(w.get("text", "")),
                        pass_no,
                    ),
                )
            n += 1
        return n

    def get_segments(self, session_id: str, pass_no: int | None = None) -> list[sqlite3.Row]:
        if pass_no is None:
            rows = self.query(
                "SELECT * FROM segments WHERE session_id=? ORDER BY t_start, idx", (session_id,)
            )
            # 同一时间段两遍转写都在时，优先精转（pass_no=2）
            best: dict[int, sqlite3.Row] = {}
            for r in rows:
                key = int(round(r["t_start"] * 2))
                if key not in best or r["pass_no"] > best[key]["pass_no"]:
                    best[key] = r
            return [best[k] for k in sorted(best)]
        return self.query(
            "SELECT * FROM segments WHERE session_id=? AND pass_no=? ORDER BY t_start, idx",
            (session_id, pass_no),
        )

    # -- 事件 -----------------------------------------------------------
    def add_event(self, session_id: str, t: float, kind: str, payload: dict | None = None,
                  confidence: float = 1.0, source: str = "manual") -> int:
        cur = self.execute(
            "INSERT INTO events(session_id,t,kind,payload_json,confidence,source) VALUES(?,?,?,?,?,?)",
            (session_id, float(t), kind, json.dumps(payload or {}, ensure_ascii=False),
             float(confidence), source),
        )
        return int(cur.lastrowid)

    def get_events(self, session_id: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM events WHERE session_id=? ORDER BY t", (session_id,))

    # -- 知识库 -----------------------------------------------------------
    def add_kb_doc(self, title: str, path: str, note: str, chars: int,
                   chunks: list, course_id: str = "") -> int:
        """chunks 可以是纯字符串列表，也可以是 {text, page, section, source_ref} 字典。"""
        cur = self.execute(
            """INSERT INTO kb_docs(course_id,title,path,note,chars,n_chunks,created_at)
               VALUES(?,?,?,?,?,?,datetime('now','localtime'))""",
            (course_id, title, path, note, chars, len(chunks)),
        )
        doc_id = int(cur.lastrowid)
        for i, c in enumerate(chunks):
            if isinstance(c, dict):
                text = str(c.get("text") or "")
                page = int(c.get("page") or 0)
                section = str(c.get("section") or "")
                ref = str(c.get("source_ref") or "")
            else:
                text, page, section, ref = str(c), 0, "", ""
            self.execute(
                """INSERT INTO kb_chunks(doc_id,course_id,ord,title,text,page,section,source_ref)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (doc_id, course_id, i + 1, title, text, page, section, ref))
        return doc_id

    def replace_kb_chunks(self, doc_id: int, chunks: list) -> None:
        """重建一份资料的块（导入参数变了以后用），块格式同 add_kb_doc。"""
        rows = self.query("SELECT course_id, title FROM kb_docs WHERE id=?", (int(doc_id),))
        if not rows:
            return
        course_id = str(rows[0]["course_id"] or "")
        title = str(rows[0]["title"] or "")
        self.execute("DELETE FROM kb_chunks WHERE doc_id=?", (int(doc_id),))
        for i, c in enumerate(chunks):
            if isinstance(c, dict):
                text = str(c.get("text") or "")
                page = int(c.get("page") or 0)
                section = str(c.get("section") or "")
                ref = str(c.get("source_ref") or "")
            else:
                text, page, section, ref = str(c), 0, "", ""
            self.execute(
                """INSERT INTO kb_chunks(doc_id,course_id,ord,title,text,page,section,source_ref)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (int(doc_id), course_id, i + 1, title, text, page, section, ref))
        self.execute("UPDATE kb_docs SET n_chunks=? WHERE id=?",
                     (len(chunks), int(doc_id)))

    def kb_doc_by_path(self, path: str) -> sqlite3.Row | None:
        """按 path 找资料。课堂记录用 `session:<sid>` 当 path，重跑就替换旧的。"""
        rows = self.query("SELECT * FROM kb_docs WHERE path=? ORDER BY id DESC LIMIT 1",
                          (str(path),))
        return rows[0] if rows else None

    def list_kb_docs(self, course_id: str | None = None) -> list[sqlite3.Row]:
        if course_id is None:
            return self.query("SELECT * FROM kb_docs ORDER BY id DESC")
        return self.query(
            "SELECT * FROM kb_docs WHERE course_id='' OR course_id=? ORDER BY id DESC",
            (course_id,))

    def kb_chunks(self, course_id: str = "") -> list[sqlite3.Row]:
        if course_id:
            return self.query(
                "SELECT * FROM kb_chunks WHERE course_id='' OR course_id=? ORDER BY doc_id, ord",
                (course_id,))
        return self.query("SELECT * FROM kb_chunks ORDER BY doc_id, ord")

    def delete_kb_doc(self, doc_id: int) -> None:
        self.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
        self.execute("DELETE FROM kb_docs WHERE id=?", (doc_id,))

    # -- 对话 -------------------------------------------------------------
    def add_chat(self, session_id: str, role: str, content: str) -> int:
        cur = self.execute(
            """INSERT INTO chat_msgs(session_id,role,content,created_at)
               VALUES(?,?,?,datetime('now','localtime'))""",
            (session_id or "", role, content))
        return int(cur.lastrowid)

    def get_chat(self, session_id: str, limit: int = 50) -> list[sqlite3.Row]:
        rows = self.query(
            "SELECT * FROM chat_msgs WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id or "", int(limit)))
        return list(reversed(rows))

    def clear_chat(self, session_id: str) -> None:
        self.execute("DELETE FROM chat_msgs WHERE session_id=?", (session_id or "",))

    # -- 屏幕内容（F3） ---------------------------------------------------
    def add_screen_note(self, session_id: str, t: float, text: str,
                        kind: str = "slide", model: str = "",
                        source: str = "vision") -> int:
        cur = self.execute(
            """INSERT INTO screen_notes(session_id,t,text,kind,model,source,created_at)
               VALUES(?,?,?,?,?,?,datetime('now','localtime'))""",
            (session_id, float(t), text, kind, model, source),
        )
        return int(cur.lastrowid)

    def get_screen_notes(self, session_id: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM screen_notes WHERE session_id=? ORDER BY t, id", (session_id,))

    def delete_screen_notes(self, session_id: str) -> None:
        self.execute("DELETE FROM screen_notes WHERE session_id=?", (session_id,))

    # -- 窗口 -----------------------------------------------------------
    def replace_windows(self, session_id: str, windows: Iterable[Any]) -> None:
        self.execute("DELETE FROM windows WHERE session_id=?", (session_id,))
        for w in windows:
            self.execute(
                """INSERT INTO windows(session_id,idx,t_start,t_end,l_score,b_score,s_score,
                                       grade,quadrant,feats_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, int(w.idx), float(w.t_start), float(w.t_end),
                    float(w.l_score), float(w.b_score), float(w.s_score),
                    w.grade, w.quadrant,
                    json.dumps(w.feats, ensure_ascii=False),
                ),
            )

    def get_windows(self, session_id: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM windows WHERE session_id=? ORDER BY idx", (session_id,))

    # -- 回看点 -----------------------------------------------------------
    def replace_knots(self, session_id: str, knots: Iterable[Any]) -> list[int]:
        self.execute("DELETE FROM knots WHERE session_id=?", (session_id,))
        ids: list[int] = []
        for k in knots:
            cur = self.execute(
                """INSERT INTO knots(session_id,t_start,t_end,level,quadrant,grade,score,
                                     l_score,b_score,evidence,evidence_json,status,excerpt,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, float(k.t_start), float(k.t_end), k.level, k.quadrant, k.grade,
                    float(k.score), float(k.l_score), float(k.b_score), k.evidence,
                    json.dumps(k.evidence_detail, ensure_ascii=False),
                    "pending", getattr(k, "excerpt", ""), time.time(),
                ),
            )
            ids.append(int(cur.lastrowid))
        return ids

    def get_knots(self, session_id: str | None = None, status: str | None = None,
                  include_dismissed: bool = True) -> list[KnotRow]:
        sql = "SELECT * FROM knots"
        where, params = [], []
        if session_id:
            where.append("session_id=?")
            params.append(session_id)
        if status:
            where.append("status=?")
            params.append(status)
        elif not include_dismissed:
            where.append("status<>'dismissed'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY score DESC, t_start"
        out: list[KnotRow] = []
        for r in self.query(sql, params):
            try:
                ej = json.loads(r["evidence_json"] or "{}")
            except Exception:
                ej = {}
            out.append(KnotRow(
                id=r["id"], session_id=r["session_id"], t_start=r["t_start"], t_end=r["t_end"],
                level=r["level"], quadrant=r["quadrant"], grade=r["grade"], score=r["score"],
                l_score=r["l_score"], b_score=r["b_score"], evidence=r["evidence"],
                status=r["status"], excerpt=r["excerpt"], evidence_json=ej,
            ))
        return out

    def set_knot_status(self, knot_id: int, status: str) -> None:
        self.execute("UPDATE knots SET status=? WHERE id=?", (status, knot_id))

    def add_feedback(self, knot_id: int, action: str, note: str = "") -> None:
        self.execute(
            "INSERT INTO feedback(knot_id,action,ts,note) VALUES(?,?,?,?)",
            (knot_id, action, time.time(), note),
        )

    def get_feedback(self, session_id: str | None = None) -> list[sqlite3.Row]:
        if session_id:
            return self.query(
                """SELECT f.*, k.session_id, k.score FROM feedback f
                   JOIN knots k ON k.id=f.knot_id WHERE k.session_id=? ORDER BY f.ts""",
                (session_id,),
            )
        return self.query("SELECT * FROM feedback ORDER BY ts")

    def get_knot(self, knot_id: int) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM knots WHERE id=?", (knot_id,))
        return rows[0] if rows else None

    def dismissed_knot_ids(self, session_id: str | None = None) -> set[int]:
        """被用户判为"不算回看点"的那些。列表里要显示（灰掉），但别再参与统计和建议。"""
        sql = "SELECT id FROM knots WHERE status='dismissed'"
        params: tuple = ()
        if session_id:
            sql += " AND session_id=?"
            params = (session_id,)
        return {int(r["id"]) for r in self.query(sql, params)}

    # -- 复习排程 -------------------------------------------------------
    def due_points(self, course_id: str | None = None, now: float | None = None) -> list[sqlite3.Row]:
        """到期待复习的知识点：排程时间已到、没退役、还没过门槛。"""
        t = time.time() if now is None else float(now)
        sql = ("SELECT * FROM knowledge_points "
               "WHERE retired=0 AND due_at>0 AND due_at<=? "
               "AND mastery < COALESCE(NULLIF(gate,0),0.9)")
        params: list[Any] = [t]
        if course_id:
            sql += " AND course_id=?"
            params.append(course_id)
        sql += " ORDER BY due_at"
        return self.query(sql, params)

    # -- 知识点 ---------------------------------------------------------
    def upsert_knowledge_point(self, course_id: str, name: str, t: float,
                               session_id: str, origin: str = "") -> int:
        rows = self.query(
            "SELECT * FROM knowledge_points WHERE course_id=? AND name=?", (course_id, name)
        )
        if rows:
            r = rows[0]
            sids = json.loads(r["session_ids"] or "[]")
            if session_id not in sids:
                sids.append(session_id)
            self.execute(
                """UPDATE knowledge_points
                   SET occurrences=occurrences+1, session_ids=?, first_seen_t=?,
                       origin=CASE WHEN ?<>'' THEN ? ELSE origin END
                   WHERE id=?""",
                (json.dumps(sids, ensure_ascii=False), min(r["first_seen_t"] or t, t),
                 origin, origin, r["id"]),
            )
            return int(r["id"])
        cur = self.execute(
            """INSERT INTO knowledge_points(course_id,name,first_seen_t,session_ids,occurrences,origin)
               VALUES(?,?,?,?,1,?)""",
            (course_id, name, t, json.dumps([session_id], ensure_ascii=False), origin),
        )
        return int(cur.lastrowid)

    def delete_knowledge_point(self, kp_id: int) -> None:
        """删知识点，连带它的尝试记录、回看点链接和共现边。"""
        self.execute("DELETE FROM kp_attempts WHERE kp_id=?", (int(kp_id),))
        self.execute("DELETE FROM knot_points WHERE kp_id=?", (int(kp_id),))
        try:
            self.execute("DELETE FROM knowledge_edges WHERE src_id=? OR dst_id=?",
                         (int(kp_id), int(kp_id)))
        except Exception:  # noqa: BLE001
            pass
        self.execute("DELETE FROM knowledge_points WHERE id=?", (int(kp_id),))

    def prune_orphan_points(self, course_id: str) -> list[str]:
        """清掉既没被验证过、也没挂上任何回看点的知识点（重分析后的词表垃圾）。"""
        rows = self.query(
            """SELECT kp.id, kp.name FROM knowledge_points kp
               WHERE kp.course_id=? AND COALESCE(kp.attempts,0)=0
                 AND NOT EXISTS (SELECT 1 FROM knot_points k WHERE k.kp_id=kp.id)""",
            (course_id,),
        ) or []
        names = [str(r["name"]) for r in rows]
        for r in rows:
            self.delete_knowledge_point(int(r["id"]))
        return names

    def link_knot_point(self, knot_id: int, kp_id: int, weight: float = 1.0) -> None:
        self.execute(
            "INSERT OR REPLACE INTO knot_points(knot_id,kp_id,weight) VALUES(?,?,?)",
            (knot_id, kp_id, weight),
        )

    def add_edge(self, src: int, dst: int, kind: str = "co_occur", weight: float = 1.0) -> None:
        if src == dst:
            return
        self.execute(
            "INSERT OR REPLACE INTO knowledge_edges(src_id,dst_id,kind,weight) VALUES(?,?,?,?)",
            (src, dst, kind, weight),
        )

    def get_knowledge_points(self, course_id: str,
                             include_retired: bool = False) -> list[sqlite3.Row]:
        """这门课的知识点。默认不返回**退役**的。

        退役是什么：术语规则升级后，有些旧知识点在新规则下再也没被任何回看点引用
        （「必要条件」这种）。它们身上可能还留着历史证据，删了不诚实；但留着更糟——
        没有回看点就没法重讲、没法出题，门槛会永远锁着，把人卡在一个不存在的东西上。
        所以：保留记录，退出一切"该学什么"的计算。
        """
        sql = "SELECT * FROM knowledge_points WHERE course_id=?"
        if not include_retired:
            sql += " AND COALESCE(retired,0)=0"
        return self.query(sql + " ORDER BY occurrences DESC", (course_id,))

    def count_retired_points(self, course_id: str) -> int:
        rows = self.query(
            """SELECT COUNT(*) AS c FROM knowledge_points
               WHERE course_id=? AND COALESCE(retired,0)=1""", (course_id,))
        return int(rows[0]["c"]) if rows else 0

    def retire_orphan_points(self, course_id: str) -> list[str]:
        """把"有历史证据但已经没有回看点"的知识点标记退役，而不是删掉证据。"""
        rows = self.query(
            """SELECT kp.id, kp.name FROM knowledge_points kp
               WHERE kp.course_id=? AND COALESCE(kp.retired,0)=0
                 AND COALESCE(kp.attempts,0)>0
                 AND NOT EXISTS (SELECT 1 FROM knot_points k WHERE k.kp_id=kp.id)""",
            (course_id,)) or []
        for r in rows:
            self.execute("UPDATE knowledge_points SET retired=1 WHERE id=?", (int(r["id"]),))
        return [str(r["name"]) for r in rows]

    # -- 掌握度（知识点会不会） -----------------------------------------
    def add_kp_attempt(self, kp_id: int, correct: float, source: str = "",
                       knot_id: int | None = None) -> int:
        cur = self.execute(
            "INSERT INTO kp_attempts(kp_id,at,correct,source,knot_id) VALUES(?,?,?,?,?)",
            (int(kp_id), time.time(), float(correct), source, knot_id),
        )
        return int(cur.lastrowid)

    def get_kp_attempts(self, kp_id: int, limit: int = 20) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM kp_attempts WHERE kp_id=? ORDER BY at ASC, id ASC", (int(kp_id),)
        )[-int(limit):]

    def set_kp_mastery(self, kp_id: int, mastery: float, attempts: int,
                       gate: float | None = None, stability: float | None = None,
                       due_at: float | None = None) -> None:
        """写掌握度。（可选）同时写记忆稳定度与到期时间，供复习队列用。"""
        now = time.time()
        s_val = None if stability is None else float(stability)
        d_val = None if due_at is None else float(due_at)
        if gate is None:
            self.execute(
                """UPDATE knowledge_points
                   SET mastery=?, attempts=?, last_evidence_at=?,
                       stability=COALESCE(?, stability), due_at=COALESCE(?, due_at)
                   WHERE id=?""",
                (float(mastery), int(attempts), now, s_val, d_val, int(kp_id)),
            )
            return
        self.execute(
            """UPDATE knowledge_points
               SET mastery=?, attempts=?, gate=?, last_evidence_at=?,
                   stability=COALESCE(?, stability), due_at=COALESCE(?, due_at)
               WHERE id=?""",
            (float(mastery), int(attempts), float(gate), now, s_val, d_val, int(kp_id)),
        )

    def get_kp(self, kp_id: int) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM knowledge_points WHERE id=?", (int(kp_id),))
        return rows[0] if rows else None

    def get_knot_points(self, knot_id: int) -> list[sqlite3.Row]:
        """某个回看点挂着的知识点（带掌握度）。"""
        return self.query(
            """SELECT kp.*, kx.weight FROM knot_points kx
               JOIN knowledge_points kp ON kp.id = kx.kp_id
               WHERE kx.knot_id=? ORDER BY kx.weight DESC""",
            (int(knot_id),),
        )

    def kp_knots(self, kp_id: int) -> list[sqlite3.Row]:
        """反查：这个知识点在哪些回看点上出现过（带节次，否则同一时间点会看起来重复）。"""
        return self.query(
            """SELECT k.*, s.title AS session_title, s.idx AS session_idx
               FROM knot_points kx JOIN knots k ON k.id = kx.knot_id
               LEFT JOIN sessions s ON s.id = k.session_id
               WHERE kx.kp_id=? ORDER BY s.idx, k.t_start""",
            (int(kp_id),),
        )

    # -- 小测 -----------------------------------------------------------
    def add_quiz_items(self, session_id: str, knot_id: int | None, items: list[dict]) -> list[int]:
        """批量存题。items 是 quiz.QuizItem.to_row() 的形状。"""
        ids: list[int] = []
        now = time.time()
        for it in items:
            cur = self.execute(
                """INSERT INTO quiz_items(session_id,knot_id,kp_id,prompt,choices_json,
                                          correct_index,evidence,evidence_ok,void_reason,
                                          source,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, knot_id, it.get("kp_id"), it["prompt"],
                 json.dumps(it.get("choices") or [], ensure_ascii=False),
                 int(it.get("correct_index") or 0), it.get("evidence") or "",
                 1 if it.get("evidence_ok") else 0, it.get("void_reason") or "",
                 it.get("source") or "llm", now),
            )
            ids.append(int(cur.lastrowid))
        return ids

    def get_quiz_items(self, session_id: str | None = None, knot_id: int | None = None,
                       only_unanswered: bool = False) -> list[sqlite3.Row]:
        sql = "SELECT * FROM quiz_items WHERE 1=1"
        args: list[Any] = []
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        if knot_id is not None:
            sql += " AND knot_id=?"
            args.append(int(knot_id))
        if only_unanswered:
            sql += " AND answered_index IS NULL"
        sql += " ORDER BY created_at DESC, id ASC"
        return self.query(sql, tuple(args))

    def answer_quiz_item(self, item_id: int, choice_index: int) -> None:
        self.execute(
            "UPDATE quiz_items SET answered_index=?, answered_at=? WHERE id=?",
            (int(choice_index), time.time(), int(item_id)),
        )

    def quiz_stats(self, session_id: str) -> dict:
        rows = self.query(
            """SELECT COUNT(*) n, SUM(answered_index IS NOT NULL) answered,
                      SUM(answered_index IS NOT NULL AND answered_index=correct_index) right_cnt
               FROM quiz_items WHERE session_id=?""", (session_id,))
        r = rows[0] if rows else None
        return {"total": int(r["n"] or 0) if r else 0,
                "answered": int(r["answered"] or 0) if r else 0,
                "correct": int(r["right_cnt"] or 0) if r else 0}

    # -- 待答问题 -------------------------------------------------------
    def add_question(self, session_id: str, text: str, t_sec: float = 0.0,
                     excerpt: str = "") -> int:
        cur = self.execute(
            """INSERT INTO pending_questions(session_id,t_sec,text,excerpt,status,created_at)
               VALUES(?,?,?,?, 'pending', ?)""",
            (session_id, float(t_sec or 0), text.strip(), excerpt[:400], time.time()),
        )
        return int(cur.lastrowid)

    def get_questions(self, session_id: str | None = None,
                      status: str | None = "pending") -> list[sqlite3.Row]:
        sql = "SELECT * FROM pending_questions WHERE 1=1"
        args: list[Any] = []
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY t_sec, id"
        return self.query(sql, tuple(args))

    def set_question_status(self, qid: int, status: str, answer: str = "") -> None:
        self.execute(
            "UPDATE pending_questions SET status=?, answer=?, answered_at=? WHERE id=?",
            (status, answer, time.time() if status != "pending" else 0, int(qid)),
        )

    def delete_question(self, qid: int) -> None:
        self.execute("DELETE FROM pending_questions WHERE id=?", (int(qid),))

    # -- 设置 KV --------------------------------------------------------
    def set_kv(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT OR REPLACE INTO settings_kv(key,value,updated_at) VALUES(?,?,?)",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )

    def get_kv(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM settings_kv WHERE key=?", (key,))
        if not rows:
            return default
        try:
            return json.loads(rows[0]["value"])
        except Exception:
            return default