"""B 站导入离线自检：不碰真网络，用假传输层把整条界面链路走一遍。

真网络只在不带参数交互使用时才碰；这里注入 `transport`，所以能反复跑：
登录态 → 收藏夹 → 勾选 → 导入 → 落库 → 界面刷新。

用法：`python tools_probe_bili.py`（输出到 data/probe/bili/）
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# 默认离屏跑（不弹窗）；加 --show 用真窗口跑，截图里才有中文字形
if "--show" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from dsh_app import bili  # noqa: E402
from dsh_app.config import Settings  # noqa: E402
from dsh_app.db import DB  # noqa: E402
from dsh_app.pipeline import Engine  # noqa: E402
from dsh_app.ui.bili_dialog import BiliImportDialog  # noqa: E402

OUT = ROOT / "data" / "probe" / "bili"

FAVOURITES = [
    {"title": "第1讲 样本空间与随机事件", "url": "https://www.bilibili.com/video/BV1aa?p=1",
     "duration": 1320, "bvid": "BV1aa", "page": 1, "n_pages": 1, "upper": "框框老师"},
    {"title": "第2讲 概率的定义与性质", "url": "https://www.bilibili.com/video/BV1bb?p=1",
     "duration": 1500, "bvid": "BV1bb", "page": 1, "n_pages": 1, "upper": "框框老师"},
    {"title": "第3讲 条件概率与独立性", "url": "https://www.bilibili.com/video/BV1cc?p=1",
     "duration": 1180, "bvid": "BV1cc", "page": 1, "n_pages": 1, "upper": "框框老师"},
]


SIMULATE_SCAN = "--simulate-scan" in sys.argv


def _router(url: str, timeout: float = 12.0) -> str:
    if "/x/web-interface/nav" in url:
        # 模拟扫码"成功但 cookie 是假的"：校验必须失败并说清原因
        if SIMULATE_SCAN:
            return json.dumps({"code": -101, "message": "账号未登录"})
        return json.dumps({"code": 0, "data": {
            "isLogin": True, "uname": "用户1", "mid": 4477, "vipStatus": 0}})
    if "/fav/folder/created/list-all" in url:
        return json.dumps({"code": 0, "data": {"list": [
            {"id": 90210, "title": "概率论与数理统计（速成）", "media_count": 3}]}})
    if "/fav/resource/list" in url:
        return json.dumps({"code": 0, "data": {"has_more": False, "medias": [
            {"type": 2, "title": i["title"], "bvid": i["bvid"],
             "duration": i["duration"], "page": 1,
             "upper": {"name": i["upper"]}} for i in FAVOURITES]}})
    if "/qrcode/generate" in url:
        return json.dumps({"code": 0, "data": {
            "url": "https://account.bilibili.com/h5/account-h5/auth/scan-web?probe=1",
            "qrcode_key": "probe-key"}})
    if "/qrcode/poll" in url:
        if SIMULATE_SCAN:
            return json.dumps({"code": 0, "data": {
                "code": 0, "message": "成功",
                "url": "https://www.bilibili.com/?SESSDATA=FAKE%2C1&bili_jct=x"
                       "&DedeUserID=4477"}})
        return json.dumps({"code": 0, "data": {"code": 86101, "message": "未扫码"}})
    if "/x/frontend/finger/spi" in url:
        return json.dumps({"code": 0, "data": {"b_3": "B3", "b_4": "B4"}})
    return json.dumps({"code": 0, "data": {}})


def _wait(app, cond=None, seconds: float = 3.0) -> bool:
    """等后台线程回填：processEvents + 小睡，直到条件成立或超时。"""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if cond is not None and cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(cond() if cond is not None else True)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv[:1])

    db = DB(OUT / "probe.db")
    eng = Engine(db, Settings.load())
    eng.db = db

    dlg = BiliImportDialog(transport=_router)   # 假传输层：离线可跑
    dlg.resize(836, 679)
    dlg._on_login({"logged_in": True, "via": "saved", "uname": "用户1",
                   "mid": 4477, "cookies": {"SESSDATA": "fake"}})
    dlg._load_sources()
    _wait(app, lambda: dlg.cb_folder.count() > 0 or "什么都" in dlg.cb_folder.currentText())
    dlg.cb_folder.setCurrentIndex(0)
    dlg._load_items()
    _wait(app, lambda: dlg.tbl.rowCount() > 0)
    rows = dlg.tbl.rowCount()
    dlg._start_qr()
    _wait(app, lambda: bool(dlg.lbl_qr.pixmap()) and not dlg.lbl_qr.pixmap().isNull())
    _wait(app, None, 0.3)
    shot = OUT / "dialog.png"
    dlg.grab().save(str(shot))

    dlg.ed_course.setText("B站 · 概率论与数理统计")
    dlg._accept_chosen()
    res = eng.import_series(dlg.course_title, dlg.chosen, source="bili")
    sessions = [dict(r) for r in db.list_sessions()]
    ok = (rows == 3 and len(dlg.chosen) == 3 and res["created"] == 3
          and all(s["video_url"] for s in sessions)
          and all(s["state"] == "imported" for s in sessions))
    if SIMULATE_SCAN:                     # 这个模式只看"校验失败说得清不清楚"
        ok = True
    print(f"清单行数 {rows}；勾选 {len(dlg.chosen)}；建课 {res['created']} 节")
    for s in sessions:
        print(f"  {s['title']}  {bili.hms(s['duration'])}  {s['video_url']}")
    print(f"截图 {shot}")

    if SIMULATE_SCAN:
        # 扫码后的校验链路：应该明确报"校验失败"，而不是默默什么都不发生
        polled = bili.qr_poll("probe-key", _router)
        print("模拟轮询:", polled["code"], polled["message"], sorted(polled["cookies"]))
        dlg2 = BiliImportDialog(transport=_router)
        dlg2._cookies = polled["cookies"] or {"SESSDATA": "FAKE"}
        dlg2._verify_after_login()
        _wait(app, lambda: "校验" in dlg2.lbl_hint.text(), 4.0)
        print("校验结果:", dlg2.lbl_hint.text())
        logfile = Path("data/logs/bili.log")
        logged = logfile.exists() and "poll code=0" in logfile.read_text(encoding="utf-8")
        print("日志有记录:", logged)
        ok = ok and "校验失败" in dlg2.lbl_hint.text() and logged
        dlg2.close()

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())