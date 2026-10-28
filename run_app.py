"""启动入口：双击即开（不是 Web 服务）。

用法：
    run_app.py                 打开桌面应用
    run_app.py --selftest      无界面自检：建库 → 演示会话 → 分析 → 打印摘要
    run_app.py --shot out.png  离屏渲染主窗口并截图（用于交付验证）
    run_app.py --probe         命令行探测模型端点
    run_app.py --probe-vision  用自造的中文测试图探测视觉模型（屏幕识别）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 双击「小蟹学习助手.exe」时用的是 base 解释器（带螃蟹图标的那个副本），
# 它不认识本项目的 venv。这里把 venv 的 site-packages 补上，
# 省得依赖 PYTHONPATH 之类的环境变量——快捷方式里没法设环境变量。
_VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.append(str(_VENV_SITE))

# 混合环境（base 解释器 + venv 包）会踩到 "老 brotli + 新 httpx2"：
# br 压缩的响应会抛 TypeError: process() takes no keyword arguments。
# 必须在 openai/httpx2 被 import 之前修，见 dsh_app/netfix.py。
try:
    from dsh_app.netfix import patch_http_encodings as _patch_http_encodings

    _PATCH_NOTE = _patch_http_encodings()
except Exception:                                  # noqa: BLE001
    _PATCH_NOTE = ""


APP_USER_MODEL_ID = "Xiaoxie.LearningAssistant"


def _own_taskbar_identity() -> None:   # 见 main() 里的说明：默认不调用
    """告诉 Windows"这是一个独立应用"，任务栏才会用我们的图标。

    不设这个 ID 时，pythonw.exe 起的所有窗口会被并进 python 那一组，
    任务栏显示的是 Python 的小图标——窗口标题栏换得再漂亮也没用。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:                                   # noqa: BLE001
        pass


def _ensure_stdio() -> None:
    """打包成 exe 后保证输出有去处。

    windowed 版本没有控制台，`sys.stdout` 可能是 None（`print` 会炸）；
    就算有控制台，用户也看不到——所以要落盘到 `data/logs/app.log`。
    用户报 bug 时让他把这个文件发回来，我们 `--selftest` 也靠它验证打包结果。
    """
    frozen = bool(getattr(sys, "frozen", False))
    missing = sys.stdout is None or sys.stderr is None
    if not (frozen or missing):
        return
    from dsh_app.config import LOG_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_DIR / "app.log", "a", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = [s for s in streams if s is not None]

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:          # noqa: BLE001
                    pass
            return len(data)

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:          # noqa: BLE001
                    pass

        def isatty(self):
            return False

    out = _Tee(sys.stdout, fh)
    err = _Tee(sys.stderr, fh)
    sys.stdout = out
    sys.stderr = err
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====")


def _selftest() -> int:
    from dsh_app.config import DATA_DIR, Settings, ensure_dirs
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    _ensure_stdio()
    if _PATCH_NOTE:
        print(f"[http] {_PATCH_NOTE}")
    ensure_dirs()
    s = Settings.load()
    db = DB(DATA_DIR / "selftest.db")      # 打包后落在可写的数据目录
    eng = Engine(db, s)
    sid = eng.seed_demo("自检课程")
    ana = eng.analyze_session(sid)
    d = eng.diagnosis(sid)
    print(f"session={sid}")
    print(f"windows={len(ana.windows)} knots={len(ana.knots)}")
    for k in ana.knots[:5]:
        print(f"  {k.t_start:7.1f}-{k.t_end:7.1f} {k.grade:6s} {k.score:.3f} "
              f"{k.level:10s} {k.evidence}")
    print(f"diagnosis: knots={d['n_knots']} plan={len(d['plan'])} "
          f"related={len(d['related'])} root_causes={len(d['root_causes'])}")
    rw = eng.reword(ana.knots[0].id if False else d["knots"][0]["id"]) if d["knots"] else None
    if rw:
        print(f"reword: strategy={rw.strategy_name} degraded={rw.degraded} "
              f"chars={len(rw.text)}")
    db.close()
    return 0


def _shot(out: str) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from dsh_app.config import DATA_DIR, Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from dsh_app.ui.main_window import MainWindow

    _ensure_stdio()
    app = QApplication.instance() or QApplication([])
    s = Settings.load()
    db = DB(DATA_DIR / "selftest.db")
    eng = Engine(db, s)
    if not db.list_sessions():
        eng.seed_demo("自检课程")
    win = MainWindow(s, db)
    win.resize(1440, 900)
    win.show()
    for _ in range(6):
        app.processEvents()
    if win.session_list.count():
        win._on_session_activated(win.session_list.item(0))
    for _ in range(6):
        app.processEvents()
    win.timeline.set_playhead(95)
    win.timeline.set_selected(win.current_knots[0].id if win.current_knots else -1)
    for _ in range(4):
        app.processEvents()
    px = win.grab()
    px.save(out)
    print(f"saved {out} {px.width()}x{px.height()}")
    db.close()
    return 0


def _probe() -> int:
    from dsh_app.config import Settings
    from dsh_app.llm import build_llm

    s = Settings.load()
    c = build_llm(s)
    r = c.probe(deep=True)
    print(f"base_url: {r.base_url}")
    print(f"summary : {r.summary()}")
    if r.models:
        print("models  : " + ", ".join(r.models[:20]))
    return 0


def _probe_vision() -> int:
    """造一张带中文的图让视觉模型读，验证"截图→文字"这条链路。"""
    import tempfile
    from pathlib import Path

    from dsh_app.config import Settings
    from dsh_app.screen import build_extractor

    st = Settings.load()
    ext = build_extractor(st)
    ok, why = ext.available()
    print(f"视觉模型：{why}｜可用={ok}")
    if not ok:
        print("请先在设置页填 api_key 与视觉模型名（如 qwen-vl-max），保存后再试。")
        return 2
    try:
        from PySide6.QtCore import QBuffer, QByteArray
        from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
    except Exception as e:  # noqa: BLE001
        print(f"需要 PySide6 生成测试图：{e}")
        return 2
    app = QGuiApplication.instance() or QGuiApplication([])
    img = QImage(900, 260, QImage.Format_RGB32)
    img.fill(QColor("#ffffff"))
    pnt = QPainter(img)
    pnt.setPen(QColor("#111111"))
    pnt.setFont(QFont("Microsoft YaHei", 22))
    pnt.drawText(40, 80, "定理 3.2：若函数在该点可导，则梯度为零是取得极值的必要条件。")
    pnt.drawText(40, 140, "注意：梯度为零并不意味着取到最优解，反例可在边界条件上构造。")
    pnt.drawText(40, 200, "这是一个测试画面，用于验证视觉模型能否正确读出中文。")
    pnt.end()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    raw = bytes(ba.data())
    tmp = Path(tempfile.gettempdir()) / "dshk_vision_probe.png"
    img.save(str(tmp), "PNG")
    print(f"测试图：{tmp}（{len(raw)} 字节）")
    res = ext.extract(raw)
    print(f"耗时 {res.latency_s:.1f}s｜kind={res.kind}")
    if res.error:
        print(f"失败：{res.error}")
        return 1
    print("识别结果：")
    print(res.text or "（空）")
    got = (res.text or "").replace(" ", "")
    hits = [k for k in ("必要条件", "梯度", "最优解", "边界条件", "可导") if k in got]
    print(f"关键词命中 {len(hits)}/5：{hits}")
    return 0 if len(hits) >= 3 else 1


def _probe_webvideo() -> int:
    """探测"这台机器现在能不能自动抓网页视频"：看浏览器音频会话，不放声音就是没在播。"""
    from dsh_app import webvideo as wv

    print("音频会话探测（判断网页视频自动抓取是否可用）")
    print(f"pycaw 可用：{wv._has_pycaw()}")
    rep = wv.probe_report(n=3, gap_s=0.5)
    for i, s in enumerate(rep["samples"], 1):
        act = [x for x in s["sessions"] if x["state"] == 1]
        print(f"  第 {i} 次：Active 会话 {[(x['proc'], x['pid']) for x in act] or '（无）'}")
        hit = s.get("browser")
        print(f"    浏览器在放声音：{hit['proc'] if hit else '否'}")
    print(f"前台窗口标题：{rep['foreground'] or '（拿不到）'}")
    browsers = rep["browsers_seen"]
    if browsers:
        print(f"见过的浏览器进程：{browsers}")
    if not rep["pycaw"]:
        print("结论：缺 pycaw，自动抓取不可用（pip install pycaw comtypes psutil）")
        return 1
    print("结论：机制可用。现在去浏览器放一个视频，再跑一次这条命令；"
          "「浏览器在放声音：chrome/msedge」出现就说明能自动抓。")
    print(f"站点识别示例：{wv.site_of('第3讲 随机变量 - 哔哩哔哩_bilibili - Google Chrome', 'chrome')}")
    return 0


def main(argv: list[str]) -> int:
    # 注意：这里**故意不**调 _own_taskbar_identity()。
    # 实测（Windows 11）：设了未注册的 AppUserModelID 之后，任务栏反而回退到
    # 进程图标（pythonw 那个通用图标），窗口自己的 icon 就不显示了；
    # 不设时任务栏按窗口图标走，螃蟹反正在。要真做成独立应用外观，
    # 得给带 AUMID 的快捷方式注册图标——那属于打包问题，不是这里的事。
    if "--selftest" in argv:
        return _selftest()
    if "--shot" in argv:
        i = argv.index("--shot")
        out = argv[i + 1] if i + 1 < len(argv) else "data/shot.png"
        return _shot(out)
    if "--probe-vision" in argv:
        return _probe_vision()
    if "--probe-webvideo" in argv:
        return _probe_webvideo()
    if "--probe" in argv:
        return _probe()
    unknown = [a for a in argv if a.startswith("--")]
    if unknown:
        # 参数拼错时绝不能悄悄启动界面：那会让人以为命令在跑，实际卡在事件循环里
        print(f"不认识的参数：{', '.join(unknown)}\n"
              f"可选：--selftest / --shot <png> / --probe / --probe-vision / --probe-webvideo")
        return 2

    from dsh_app.ui.main_window import launch_app
    return launch_app()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))