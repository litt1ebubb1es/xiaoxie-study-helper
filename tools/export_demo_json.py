"""把一次真实分析的结果导出成只读 JSON，供在线演示站使用。

为什么单独一个脚本而不是让演示站跑管线：
1. 演示服务器只有 2 GB 内存，跑不动 numpy/PySide6 那一套；
2. 演示数据必须是**合成**的（seed_demo），不能把用户真实课堂录音带出去；
3. 演示站只需要"看起来是真的"——而它确实是真算法跑真数据算出来的，只是提前算好。

用法：
    ./.venv/Scripts/python.exe tools/export_demo_json.py [输出路径]

默认输出 webdemo/data/demo.json。
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_app.config import Settings            # noqa: E402
from dsh_app.db import DB                     # noqa: E402
from dsh_app.pipeline import Engine           # noqa: E402

LEVEL_CN = {
    "explicit": "显性回看点",
    "implicit_candidate": "隐性回看点候选",
    "implicit": "隐性回看点",
    "quiet": "平稳",
}
GRADE_CN = {"strong": "强", "medium": "中", "weak": "弱"}


def _series(ana) -> list[dict]:
    out = []
    for w in ana.windows:
        out.append({
            "idx": int(w.idx),
            "t": round(float(w.t_start), 1),
            "e": round(float(w.t_end), 1),
            "L": round(float(w.l_score), 4),
            "B": round(float(w.b_score), 4),
            "grade": str(w.grade),
            "quadrant": str(w.quadrant),
            "valid": bool(w.valid),
            "text": " ".join(str(w.text or "").split())[:400],
        })
    return out


def _quiz(db: DB, session_id: str) -> list[dict]:
    rows = db.query(
        "SELECT id, knot_id, prompt, choices_json, correct_index, evidence "
        "FROM quiz_items WHERE session_id=? ORDER BY id", (session_id,))
    out = []
    for r in rows:
        try:
            choices = json.loads(r["choices_json"] or "[]")
        except json.JSONDecodeError:
            continue
        if not choices:
            continue
        out.append({
            "id": int(r["id"]),
            "knot_id": int(r["knot_id"]) if r["knot_id"] else 0,
            "prompt": r["prompt"],
            "choices": choices,
            "correct": int(r["correct_index"]),
            "evidence": r["evidence"] or "",
        })
    return out


VIDEO_SLOTS = [
    ("intro", "介绍片", "动效短片",
     "它解决什么问题、凭什么敢说\"只重讲那几段\""),
    ("demo", "使用演示", "真机录屏",
     "从头走一遍：导入课程 → 分析 → 回看点 → 换个讲法 → 小测（含 B 站公开课程画面）"),
    ("lite", "演示视频（精简版）", "真机录屏 · 精简",
     "两分半走完同一条链路：导入 → 分析 → 划段重讲，适合先扫一眼"),
]


def _probe_videos(root: Path) -> list[dict]:
    """扫 webdemo/video/ 下约定名字的片子，有就带上，没有就不带。
    页面据此决定显示几个播放器——宁可不出现，也不给空壳。"""
    vdir = root / "video"
    if not vdir.is_dir():
        return []
    out: list[dict] = []
    for name, title, tag, note in VIDEO_SLOTS:
        mp4 = vdir / f"{name}.mp4"
        if not mp4.exists():
            continue
        item = {
            "id": name,
            "title": title,
            "tag": tag,
            "src": f"video/{name}.mp4",
            "note": note,
            "size_mb": round(mp4.stat().st_size / 1024 / 1024, 1),
        }
        poster = vdir / f"{name}.jpg"
        if poster.exists():
            item["poster"] = f"video/{name}.jpg"
        try:
            import subprocess
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(mp4)],
                capture_output=True, text=True, timeout=20,
            )
            sec = float(res.stdout.strip())
            item["duration_text"] = _hms(sec)
            item["duration"] = round(sec, 1)
        except Exception:                               # noqa: BLE001
            pass
        out.append(item)
    return out


def build(out_path: Path) -> dict:
    tmp_db = Path("tmp_demo/demo.db")
    tmp_db.parent.mkdir(parents=True, exist_ok=True)
    for p in tmp_db.parent.glob("demo.db*"):
        p.unlink()

    settings = Settings.load()
    # 演示站不接模型：重讲走 review.py 的离线规则版，JSON 里会标 degraded，
    # 页面照实写"离线规则版"，不假称是模型生成。
    settings.llm.backend = "off"

    db = DB(tmp_db)
    eng = Engine(db, settings)
    # seed_demo 每次都往同一门课里加一节，但节名写死成"第 1 讲"。演示要给"跨节关联"
    # 和"根因"这两块喂同一门课的多节数据，所以种三节、把节名改成真实序号。
    # 内容仍是同一段合成讲解——meta 里写明了，不假装是三堂不同的课。
    N_SESSIONS = 3
    seed_sid = ""
    for i in range(N_SESSIONS):
        sid = eng.seed_demo("演示课程")
        seed_sid = seed_sid or sid
        db.update_session(sid, title=f"第 {i + 1} 讲　随机变量与分布")

    analyzed: list[str] = []
    for row in db.list_sessions():
        try:
            eng.analyze_session(row["id"])
            analyzed.append(row["id"])
        except Exception as e:                      # noqa: BLE001
            print(f"  [skip] {row['id']}: {e}")
    session_id = seed_sid if seed_sid in analyzed else (analyzed[-1] if analyzed else seed_sid)

    ana = eng.analyze_session(session_id)
    diag = eng.diagnosis(session_id)

    # 给小测：make_quiz 走离线规则也能出题（evidence 必须在原文里查得到，否则作废）
    for k in diag.get("knots", [])[:2]:
        try:
            eng.make_quiz(int(k["id"]), 3)
        except Exception:                            # noqa: BLE001
            pass

    # 逐回看点生成"换个讲法"，最多存 3 条，页面按需切换
    rewords = []
    for k in diag.get("knots", [])[:3]:
        rw = eng.reword(int(k["id"]))
        if not rw.text:
            continue
        rewords.append({
            "knot_id": int(k["id"]),
            "strategy": rw.strategy_name or "补齐缺失前后文",
            "degraded": bool(rw.degraded),
            "text": rw.text,
        })

    knots = []
    spans = {int(r["id"]): (float(r["t_start"] or 0.0), float(r["t_end"] or 0.0))
             for r in db.query("SELECT id, t_start, t_end FROM knots WHERE session_id=?",
                               (session_id,))}
    for k in diag.get("knots", []):
        kid = int(k["id"])
        t_s, t_e = spans.get(kid, (float(k.get("t_start") or 0.0), 0.0))
        knots.append({
            "id": kid,
            "t": k.get("t"),
            "t_start": round(t_s, 1),
            "t_end": round(t_e, 1),
            "level": k.get("level"),
            "level_cn": LEVEL_CN.get(k.get("level"), "回看点"),
            "grade": k.get("grade"),
            "grade_cn": GRADE_CN.get(k.get("grade"), ""),
            "score": round(float(k.get("score") or 0.0), 3),
            "quadrant": k.get("quadrant"),
            "evidence": k.get("evidence") or "",
            "excerpt": (k.get("excerpt") or "").strip(),
            "terms": (k.get("terms") or [])[:12],
        })

    sessions = []
    for s in db.list_sessions():
        n = db.query("SELECT COUNT(*) AS c FROM knots WHERE session_id=?", (s["id"],))
        sessions.append({
            "idx": int(s["idx"] or 0),
            "title": s["title"] or "",
            "duration_text": _hms(float(s["duration"] or 0.0)),
            "n_knots": int(n[0]["c"]) if n else 0,
            "current": s["id"] == session_id,
        })

    course_rows = db.query("SELECT * FROM courses ORDER BY rowid")
    course = {"name": course_rows[0]["title"] if course_rows else "演示课程",
              "sessions": sessions}

    videos = _probe_videos(out_path.resolve().parents[1])

    data = {
        "meta": {
            "app": "小蟹学习助手",
            "version": "0.1.0",
            "generated_at": time.strftime("%Y-%m-%d %H:%M"),
            "data_source": "seed_demo 合成课程（非真实课堂录音）",
            "note": "只读快照：分析结果由真实算法在合成数据上算出；重讲为离线规则版（本次未接模型）",
            # 前端据此标注来源，不靠 RewordResult.degraded 猜——那个字段只表示"这次降级了"，
            # 模型没接时它也是 False，拿它当"模型生成"的判据会撒谎。
            "reword_source": "rules",
            "repo": "https://github.com/litt1ebubb1es/xiaoxie-study-helper",
        },
        "course": course,
        "session": diag.get("session", {}),
        "stats": ana.stats,
        "series": _series(ana),
        "knots": knots,
        "rewords": rewords,
        "diagnosis": {
            "n_knots": diag.get("n_knots", 0),
            "n_strong": diag.get("n_strong", 0),
            "distribution": diag.get("distribution", []),
            "root_causes": diag.get("root_causes", []),
            "related": diag.get("related", []),
            "plan": diag.get("plan", []),
        },
        "quiz": _quiz(db, session_id),
        # 有 webdemo/video/demo.mp4 才有，没有就是 None（页面隐藏播放器）
        # 输出在 webdemo/data/demo.json，往上两级才是站点根（video/ 在那下面）
        # videos: 多支片子按 VIDEO_SLOTS 顺序排；video 保留成第一支，兼容旧逻辑
        "videos": videos,
        "video": videos[0] if videos else None,
    }
    db.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def _hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("webdemo/data/demo.json")
    d = build(out)
    size = out.stat().st_size
    print(f"写出 {out}  {size/1024:.1f} KB")
    print(f"  会话      : {d['session'].get('title')} ({d['session'].get('duration_text')})")
    print(f"  窗口/回看点: {len(d['series'])} / {len(d['knots'])}")
    print(f"  重讲      : {len(d['rewords'])} 条")
    print(f"  根因/复习 : {len(d['diagnosis']['root_causes'])} / {len(d['diagnosis']['plan'])}")
    print(f"  小测      : {len(d['quiz'])} 题")
    for v in d.get("videos", []):
        print(f"  视频      : {v['title']} {v.get('duration_text','?')} / {v['size_mb']} MB / {v['src']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))