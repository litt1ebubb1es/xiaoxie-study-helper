"""把旧的演示会话按当前代码重建一次（术语抽取规则升级后的数据迁移）。

为什么需要：知识点是从转写里抽出来的。术语抽取规则改了以后，旧库里存的还是
旧规则算出来的知识点（「定义 / 必要条件 / 边界条件」这类元词），
而 `rebuild_knowledge` 只能按**现存转写**重算——演示会话的转写本身就是旧模板文本。

所以对 source='demo' 的会话：重新合成转写 → 重新分析 → 重新抽术语。
真实录音（source 不是 demo）一律不碰，只做术语重链接。

用法：
    .venv\\Scripts\\python.exe tools_migrate_demo.py            # 只看会改什么
    .venv\\Scripts\\python.exe tools_migrate_demo.py --apply    # 真的改（先自动备份）
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dsh_app.config import DATA_DIR, Settings          # noqa: E402
from dsh_app.db import DB                              # noqa: E402
from dsh_app.pipeline import Engine                    # noqa: E402


def main() -> int:
    apply = "--apply" in sys.argv
    db_path = DATA_DIR / "app.db"
    db = DB(db_path)
    eng = Engine(db, Settings.load())

    demo = [r for r in db.query("SELECT * FROM sessions ORDER BY created_at") or []
            if str(r["source"]) == "demo"]
    real = [r for r in db.query("SELECT * FROM sessions") or [] if str(r["source"]) != "demo"]
    print(f"库：{db_path}")
    print(f"演示会话 {len(demo)} 个，真实录音 {len(real)} 个")
    for r in demo:
        print(f"  demo  {r['id']}  {r['title']}")
    before = [dict(r) for r in db.query("SELECT name, course_id, attempts FROM knowledge_points")]
    print("现有知识点：", [b["name"] for b in before])

    if not apply:
        print("\n（没加 --apply，什么都没改）")
        return 0

    backup = db_path.with_suffix(f".bak-before-terms-{time.strftime('%Y%m%d-%H%M%S')}.db")
    shutil.copy2(db_path, backup)
    print("已备份：", backup)

    # 1) 真实录音：只按现存转写重链接术语
    for r in real:
        res = eng.rebuild_knowledge(str(r["course_id"]), use_llm=True)
        row = (res.get("courses") or [{}])[0]
        print(f"重链接 {r['id']} → 剪掉 {len(row.get('pruned') or [])} 个旧知识点")

    # 2) 演示会话：重新合成转写并重分析（同一门课只留一节，旧模板重复的删掉）
    by_course: dict[str, list] = {}
    for r in demo:
        by_course.setdefault(str(r["course_id"]), []).append(r)
    for cid, rows in by_course.items():
        keep = rows[0]
        for extra in rows[1:]:
            print(f"  删掉重复的演示会话 {extra['id']}")
            db.delete_session(str(extra["id"]))
        from dsh_app.asr import DemoASR

        res = DemoASR().transcribe(None)
        db.replace_segments(str(keep["id"]), res.as_dicts(), pass_no=1)
        db.update_session(str(keep["id"]),
                          duration=max(s.t_end for s in res.segments),
                          state="transcribed", error="")
        eng.analyze_session(str(keep["id"]))
        print(f"  重分析 {keep['id']} → 回看点 {len(db.get_knots(str(keep['id'])))} 个")

    after = [r["name"] for r in db.query("SELECT name FROM knowledge_points")]
    print("迁移后知识点：", after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())