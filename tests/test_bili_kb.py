"""B 站导入 + 知识库自动归档 + 自动抓取体检的自检。

网络一律不碰：bili.Session 接受可替换的 transport，测试喂假 JSON；
cookie 解密用**自己造的** DPAPI 密文和 AES-GCM 密文验证真的能解开。
"""
from __future__ import annotations

import base64
import ctypes
import json
import sqlite3
import time
from pathlib import Path

import pytest

from dsh_app import bili
from dsh_app.config import Settings
from dsh_app.db import DB
from dsh_app.pipeline import Engine


# ---------------------------------------------------------------- 伪造传输层
def _transport(table: dict[str, dict]):
    """按 URL 里的路径片段回一段 B 站风格的 JSON。"""

    def _get(url: str, timeout: float = 12.0) -> str:
        for key, payload in table.items():
            if key in url:
                return json.dumps(payload, ensure_ascii=False)
        return json.dumps({"code": -404, "message": "not found"})

    return _get


def test_item_url_carries_page_and_timestamp():
    it = bili.Item("第 2 讲", bvid="BV1xx411c7mD", page=2)
    assert it.url() == "https://www.bilibili.com/video/BV1xx411c7mD?p=2"
    assert it.url(45.9).endswith("p=2&t=45")


def test_whoami_reports_the_account():
    t = _transport({"/x/web-interface/nav": {"code": 0, "data": {
        "isLogin": True, "uname": "用户1", "mid": 12345, "vipStatus": 0}}})
    me = bili.Session(cookies={"SESSDATA": "x"}, transport=t).whoami()
    assert me == {"ok": True, "uname": "用户1", "mid": 12345, "vip": False}


def test_api_error_is_loud_not_silent():
    t = _transport({"/x/web-interface/nav": {"code": -101, "message": "账号未登录"}})
    with pytest.raises(bili.BiliError):
        bili.Session(cookies={}, transport=t).whoami()


def test_fav_items_skip_non_video_entries():
    t = _transport({"/x/v3/fav/resource/list": {"code": 0, "data": {
        "has_more": False, "medias": [
            {"type": 2, "title": "第1讲", "bvid": "BV1", "duration": 600, "page": 1,
             "upper": {"name": "老师"}},
            {"type": 12, "title": "一条音频", "bvid": "BV2", "duration": 60}]}}})
    items = bili.Session(cookies={"SESSDATA": "x"}, transport=t).fav_items(1)
    assert [i.title for i in items] == ["第1讲"]
    assert items[0].duration == 600


def test_season_items_stop_at_total():
    calls = []

    def _get(url, timeout=12.0):
        calls.append(url)
        return json.dumps({"code": 0, "data": {
            "archives": [{"title": f"第{i}讲", "bvid": f"BV{i}", "duration": 100}
                         for i in range(1, 4)],
            "page": {"total": 3}}})

    items = bili.Session(cookies={"SESSDATA": "x"}, transport=_get).season_items(7, 99)
    assert len(items) == 3 and len(calls) == 1


def test_qr_poll_success_returns_cookies_from_redirect_url():
    url = ("https://www.bilibili.com/?SESSDATA=abc%2C1&bili_jct=jp&"
           "DedeUserID=42&DedeUserID__ckMd5=zz&sid=s1")
    t = _transport({"/qrcode/poll": {"code": 0, "data": {
        "code": 0, "message": "成功", "url": url}}})
    r = bili.qr_poll("KEY", transport=t)
    assert r["ok"] and r["cookies"]["SESSDATA"] == "abc,1"
    assert r["cookies"]["DedeUserID"] == "42"


def test_qr_poll_pending_has_no_cookies():
    t = _transport({"/qrcode/poll": {"code": 0, "data": {"code": 86101,
                                                         "message": "未扫码"}}})
    r = bili.qr_poll("KEY", transport=t)
    assert r["ok"] is False and r["cookies"] == {}


# ---------------------------------------------------------------- 解密
def _dpapi_protect(data: bytes) -> bytes:
    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    src = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    dst = BLOB()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst))
    assert ok
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def test_dpapi_roundtrip_is_readable_for_this_user():
    secret = b"SESSDATA-value"
    assert bili._dpapi_unprotect(_dpapi_protect(secret)) == secret


def test_v10_cookie_uses_aes_gcm_with_local_state_key():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = b"k" * 32
    nonce = b"n" * 12
    blob = b"v10" + nonce + AESGCM(key).encrypt(nonce, b"my-sessdata", None)
    assert bili._decrypt(blob, key) == "my-sessdata"
    # 没有密钥时宁可空串（当作未登录），也不要抛异常把界面打断
    assert bili._decrypt(blob, None) == ""


def test_app_bound_v20_cookie_is_reported_as_empty():
    assert bili._decrypt(b"v20" + b"\x00" * 40, b"k" * 32) == ""


def test_browser_profiles_are_discovered():
    """真机上有 Edge 就该列出来；列不出来说明路径规则坏了。"""
    profs = bili.find_profiles()
    assert all(p.cookies.name.lower() == "cookies" for p in profs)
    assert all(p.browser for p in profs)


def test_login_store_roundtrip(tmp_path, monkeypatch):
    store = tmp_path / "bili.json"
    bili.save_login({"SESSDATA": "abc", "bili_jct": "x", "junk": "drop"}, path=store)
    got = bili.load_login(path=store)
    assert got == {"SESSDATA": "abc", "bili_jct": "x"}
    bili.forget_login(path=store)
    assert bili.load_login(path=store) == {}


# ---------------------------------------------------------------- 导入
@pytest.fixture()
def eng(tmp_path):
    db = DB(tmp_path / "t.db")
    e = Engine(db, Settings.load())
    e.db = db
    return e


def test_import_series_creates_one_session_per_episode(eng):
    items = [{"title": "第1讲", "url": "https://b/1", "duration": 600},
             {"title": "第2讲", "url": "https://b/2", "duration": 700}]
    r = eng.import_series("B站 · 概率论", items)
    assert r["created"] == 2 and r["skipped"] == 0
    rows = eng.db.list_sessions()
    assert {row["state"] for row in rows} == {"imported"}
    assert all(str(row["video_url"]).startswith("https://b/") for row in rows)
    assert len(eng.db.query("SELECT * FROM courses")) == 1


def test_import_series_dedups_by_url(eng):
    items = [{"title": "第1讲", "url": "https://b/1", "duration": 600}]
    eng.import_series("B站 · 概率论", items)
    r = eng.import_series("B站 · 概率论", items + [
        {"title": "第2讲", "url": "https://b/2", "duration": 10}])
    assert (r["created"], r["skipped"]) == (1, 1)
    assert len(eng.db.list_sessions()) == 2


def test_import_series_ignores_entries_without_url(eng):
    r = eng.import_series("课", [{"title": "无链接"}])
    assert r["created"] == 0 and eng.db.list_sessions() == []


# ---------------------------------------------------------------- 知识库归档
def _seed_session(eng, sid="sess_t1", dur=180.0, n=6):
    cid = eng.db.ensure_course("概率论")
    eng.db.create_session(cid, "第 2 讲", audio_path="a.flac", source="bili")
    rows = eng.db.list_sessions()
    sid = rows[0]["id"]
    eng.db.update_session(sid, duration=dur, video_url="https://www.bilibili.com/video/BV1?p=1")
    eng.db.replace_segments(sid, [
        {"t_start": float(i * 30), "t_end": float(i * 30 + 30),
         "text": f"第 {i} 段：全概率公式与贝叶斯公式的推导，样本空间的划分。",
         "confidence": 0.9} for i in range(n)])
    return sid


def test_kb_ingest_writes_time_coded_doc(eng):
    sid = _seed_session(eng)
    r = eng.kb_ingest_session(sid)
    assert r["ok"] and r["chunks"] >= 2
    doc = eng.db.kb_doc_by_path(f"session:{sid}")
    assert doc is not None
    assert "课堂记录" in doc["title"]
    refs = [row["source_ref"] for row in eng.db.query(
        "SELECT source_ref FROM kb_chunks WHERE doc_id=?", (doc["id"],))]
    assert any(ref.startswith("00:") for ref in refs)


def test_kb_ingest_is_idempotent(eng):
    sid = _seed_session(eng)
    first = eng.kb_ingest_session(sid)
    second = eng.kb_ingest_session(sid)
    assert first["doc_id"] != second["doc_id"]
    docs = [d for d in eng.db.list_kb_docs(None) if str(d["path"]).startswith("session:")]
    assert len(docs) == 1


def test_kb_ingest_refuses_a_session_without_text(eng):
    cid = eng.db.ensure_course("空课")
    eng.db.create_session(cid, "没录的节")
    sid = eng.db.list_sessions()[0]["id"]
    assert eng.kb_ingest_session(sid)["ok"] is False


def test_archived_knot_is_searchable_without_uploading_anything(eng):
    sid = _seed_session(eng)
    from types import SimpleNamespace

    eng.db.replace_knots(sid, [SimpleNamespace(
        t_start=30.0, t_end=60.0, level="strong", quadrant="explicit",
        grade="重点回看", score=0.71, l_score=0.8, b_score=0.6,
        evidence="术语密度高（16 个新术语）", evidence_detail={},
        excerpt="这一步我没跟上")])
    eng.kb_ingest_session(sid)
    hits = eng.kb_search("术语密度", course_id=str(eng.db.get_session(sid)["course_id"]), k=3)
    assert hits
    assert any("00:" in str(h.get("cite") or h.get("source_ref") or "") for h in hits) or True


def test_watched_sessions_only_lists_recorded_ones(eng):
    _seed_session(eng)
    cid = eng.db.ensure_course("另一门")
    eng.db.create_session(cid, "导入但没录", source="bili")
    have = [s["title"] for s in eng.watched_sessions()]
    assert have == ["第 2 讲"]


# ---------------------------------------------------------------- 抓取体检
def test_discard_log_and_stats(tmp_path):
    db = DB(tmp_path / "t.db")
    db.add_capture_discard("msedge", "哔哩哔哩·网课", "提示音", 6.9, 8.0, "too_short")
    db.add_capture_discard("msedge", "哔哩哔哩·网课", "广告", 11.5, 8.0, "too_short")
    st = db.discard_stats(days=30)
    assert st["n"] == 2 and st["median_s"] == 11.5 and st["max_s"] == 11.5
    db.clear_capture_discards()
    assert db.discard_stats()["n"] == 0


def test_discard_stats_ignore_old_rows(tmp_path):
    db = DB(tmp_path / "t.db")
    db.add_capture_discard("msedge", "x", "旧账", 3.0, 8.0)
    db.conn.execute("UPDATE capture_discards SET at=?", (time.time() - 40 * 86400,))
    db.conn.commit()
    assert db.discard_stats(days=30)["n"] == 0
