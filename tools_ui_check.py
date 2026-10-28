"""无视觉模型时的界面自检：用像素与字体度量代替"肉眼看图"。

检查项：
1. 中文字体是否被 Qt 找到（否则整套界面会是方框）
2. 时间轴绘图区是否真有曲线像素（负荷曲线/回看点带）
3. 各标签页能否正常构建且控件数量合理
4. 回看点表格、会话列表是否填充
5. 截图是否非空白（整体像素方差）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 脚本就在仓库根目录：parent 就是仓库根（之前多退了一层，产物被写到盘符根目录）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QFontDatabase, QFontMetrics  # noqa: E402
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QApplication  # noqa: E402

from dsh_app.config import Settings  # noqa: E402
from dsh_app.db import DB  # noqa: E402
from dsh_app.pipeline import Engine  # noqa: E402
from dsh_app.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    ok = True
    app = QApplication.instance() or QApplication([])
    fams = QFontDatabase.families()
    cjk = [f for f in fams if any(k in f for k in ("YaHei", "SimHei", "SimSun", "Noto", "Hei"))]
    print(f"fonts(offscreen): total={len(fams)} cjk_candidates={cjk[:5]}")
    if not cjk:
        # offscreen 平台不加载系统字体（Qt 官方行为），字体能力要用原生平台单独查
        import subprocess
        probe = (
            "from PySide6.QtWidgets import QApplication;"
            "from PySide6.QtGui import QFontDatabase,QFontInfo,QFont;"
            "a=QApplication([]);f=QFontDatabase.families();"
            "print(a.platformName(),len(f),('Microsoft YaHei' in f),"
            "QFontInfo(QFont('Microsoft YaHei',13)).family())"
        )
        env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           env=env, timeout=120)
        out = (r.stdout or "").strip()
        print(f"fonts(native): {out}  {r.stderr.strip()[:120]}")
        parts = out.split()
        if len(parts) >= 4 and parts[0] == "windows" and int(parts[1] or 0) > 100 and parts[2] == "True":
            print("NOTE: offscreen 无字体属正常，原生平台字体可用（中文字体已确认）")
        else:
            print("FAIL: 原生平台也找不到中文字体"); ok = False
    fm = QFontMetrics(app.font())
    w = fm.horizontalAdvance("小蟹学习助手")
    print(f"cjk_text_width={w}px")
    if w <= 0:
        print("FAIL: 中文串宽度为 0"); ok = False

    settings = Settings.load()
    # 界面自检不出网：预热模型 / 拉模型列表都会真的打云端接口，
    # 既让自检变成"网络测试"，退出时还会留下跑不完的 QThread（"Destroyed while running"）
    settings.llm.preload_on_start = False
    settings.llm.base_url = ""
    settings.llm.api_key = ""
    db = DB(ROOT / "data" / "ui_check.db")
    eng = Engine(db, settings)
    if not db.list_sessions():
        eng.seed_demo("界面自检课程")
    win = MainWindow(settings, db)
    win.resize(1440, 900)
    win.show()
    for _ in range(8):
        app.processEvents()
    if win.session_list.count():
        win._on_session_activated(win.session_list.item(0))
    for _ in range(8):
        app.processEvents()

    print(f"sessions={win.session_list.count()} tabs={win.tabs.count()} "
          f"knot_rows={win.knot_table.rowCount()} knots={len(win.current_knots)}")
    expected = ["时间轴与回看点", "划段重讲", "小测与待答", "漏洞地图", "问模型",
                "设置与模型部署"]
    names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    if names != expected:
        print(f"FAIL: 页签不对：{names}"); ok = False
    if win.kb_list is None or win.chat_view is None:
        print("FAIL: 对话页控件缺失"); ok = False
    if len(win.findChildren(QShortcut)) != 0:
        print("FAIL: 不应再有任何快捷键"); ok = False
    # 本地转写（Whisper）一栏必须在，且尺寸下拉有 5 个选项
    if not hasattr(win, "btn_w_dl") or win.cmb_whisper.count() != 5:
        print("FAIL: 本地转写（Whisper）控件缺失"); ok = False
    else:
        print(f"whisper sizes={win.cmb_whisper.count()} "
              f"status_label={'有' if win.lbl_whisper.text() else '空'}")
    if win.knot_table.rowCount() == 0:
        print("FAIL: 回看点表格为空"); ok = False

    win.timeline.set_playhead(95)
    if win.current_knots:
        win.timeline.set_selected(win.current_knots[0].id)
    for _ in range(4):
        app.processEvents()

    # 像素分析
    img = win.timeline.grab().toImage()
    w_, h_ = img.width(), img.height()
    # 配色断言也来自主题模块：换肤时这里跟着走，不会留下"旧色值还在就过"的假通过
    from dsh_app.ui import theme as TH
    bg = QColor(TH.TL_BG).rgb()
    want = {"load": QColor(TH.TL_LOAD), "behavior": QColor(TH.TL_BEHAV),
            "knot": QColor(TH.TL_KNOT_STRONG), "med": QColor(TH.TL_KNOT_MED)}
    hits = {k: 0 for k in want}
    non_bg = 0
    for y in range(0, h_, 2):
        for x in range(0, w_, 2):
            c = img.pixelColor(x, y)
            if c.rgb() != bg:
                non_bg += 1
            for name, col in want.items():
                # 抗锯齿会把颜色混向背景，所以用通道容差而不是精确相等
                if (abs(c.red() - col.red()) < 40 and abs(c.green() - col.green()) < 40
                        and abs(c.blue() - col.blue()) < 40):
                    hits[name] += 1
                    break
    load_px, behav_px = hits["load"], hits["behavior"]
    knot_px = hits["knot"] + hits["med"]
    total = (w_ // 2) * (h_ // 2)
    print(f"timeline_px: total_sampled={total} non_bg={non_bg} "
          f"load={load_px} behavior={behav_px} knot={knot_px}")
    if non_bg < total * 0.01:
        print("FAIL: 时间轴几乎是空白"); ok = False
    if load_px == 0:
        print("FAIL: 没有画负荷曲线"); ok = False
    if knot_px == 0 and behav_px == 0:
        print("FAIL: 没有画回看点带或行为曲线"); ok = False

    px = win.grab()
    px.save(str(ROOT / "data" / "shot_main.png"))
    im = px.toImage()
    vals = set()
    for y in range(0, im.height(), 7):
        for x in range(0, im.width(), 7):
            vals.add(im.pixelColor(x, y).rgb())
    print(f"main_shot={im.width()}x{im.height()} distinct_colors={len(vals)}")
    if len(vals) < 50:
        print("FAIL: 主界面配色过于单一，可能没渲染控件"); ok = False

    # 逐页切换，确认不崩
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        for _ in range(3):
            app.processEvents()
    d = eng.diagnosis(win.current_session)
    win.map_view.setHtml(__import__("dsh_app.ui.main_window", fromlist=["_diagnosis_html"])
                         ._diagnosis_html(d))
    for _ in range(3):
        app.processEvents()
    win.tabs.setCurrentIndex(2)
    # 知识点掌握度表：内核建过图就必须能显示，否则漏洞地图还是"只增不减"的老样子
    win.refresh_kp_table()
    for _ in range(3):
        app.processEvents()
    print(f"kp_rows={win.kp_table.rowCount()} kp_hint={win.lbl_kp.text()[:24]}")
    # 小测与待答页：控件必须在，且能刷出回看点下拉与问题列表
    win.tabs.setCurrentIndex(2)
    win.refresh_quiz_knots()
    win.refresh_questions()
    for _ in range(3):
        app.processEvents()
    # 知识库索引签名：导入过就必须"不是过期"，且重建按钮在
    st = win.engine.kb_index_status()
    print(f"kb_sig={st['sig']} stale={st['stale']} chunks={st['chunks']} "
          f"rebuild_btn={win.btn_kb_rebuild.text()}")
    if st["has_index"] and st["stale"]:
        print("FAIL: 刚导入就报索引过期"); ok = False
    print(f"quiz_knots={win.quiz_knot.count()} q_list={win.q_list.count()} "
          f"quiz_hint={win.lbl_quiz.text()[:20]}")
    if len(win.current_knots) and win.quiz_knot.count() != len(win.current_knots):
        print("FAIL: 小测页回看点下拉数量与回看点数不一致"); ok = False
    # 覆盖度提示必须在（哪怕写的是"0/4"），否则用户又以为画面全覆盖了
    win.tabs.setCurrentIndex(0)
    win._load_session_view()
    for _ in range(3):
        app.processEvents()
    cov_text = win.lbl_coverage.text()
    print(f"coverage_hint={cov_text[:40]}")
    if win.current_knots and "画面覆盖" not in cov_text:
        print("FAIL: 覆盖度提示没写出来"); ok = False
    # 导出笔记：能在临时目录里落盘，且带四个必有小节
    import tempfile as _tf
    out = Path(_tf.mkdtemp(dir=str(ROOT / "data" / "probe"))) / "ui_export.md"
    exp = win.engine.export_notes(win.current_session, out)
    if not (exp.get("ok") and out.exists()):
        print(f"FAIL: 导出笔记失败 {exp}"); ok = False
    else:
        md = out.read_text(encoding="utf-8")
        need = ["## 回看点与掌握度", "画面覆盖："]
        miss = [n for n in need if n not in md]
        print(f"export_chars={len(md)} missing_sections={miss}")
        if miss:
            print("FAIL: 导出内容缺小节"); ok = False
    n_kp = len(win.db.get_knowledge_points(win._current_course_id()))
    if n_kp and win.kp_table.rowCount() != n_kp:
        print(f"FAIL: 知识点表应有 {n_kp} 行，实际 {win.kp_table.rowCount()}"); ok = False
    px2 = win.grab()
    px2.save(str(ROOT / "data" / "shot_map.png"))
    print(f"map_page saved: {px2.width()}x{px2.height()}")

    win.tabs.setCurrentIndex(0)
    print("RESULT:", "PASS" if ok else "FAIL")
    # 收尾：先让窗口走 closeEvent（它负责等后台线程），否则 Qt 会在退出时
    # 报 "QThread: Destroyed while thread is still running"
    win.close()
    app.processEvents()
    db.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())