"""B 站：借用浏览器里已有的登录态，把"我在看的课"整门导进来。

为什么要读浏览器 cookie：B 站的数据（收藏夹、合集、观看历史）都要登录态，
而这个软件**不做账号体系**——让用户在我们的窗口里再登一次 B 站，密码要经过
第三方程序，风险比收益大得多。浏览器的 cookie 库本来就在本机，DPAPI 是当前
用户级别的加密，读它等价于"用你已经登录的那个身份查一次自己的收藏夹"。

三条边界，写清楚了才好维护：

1. **只读**：只在 cookie 库里 SELECT，绝不写回、不调用任何写接口。
2. **只取 bilibili.com 的 cookie**，拿不到就明说"没读到登录"，让用户去浏览器登录。
3. **不落盘**：cookie 只在内存里活一次请求周期，不写数据库、不写日志。

解密链路（Windows / Chromium 系）：
`Local State` 的 `os_crypt.encrypted_key` → 去 `DPAPI` 前缀 → `CryptUnprotectData`
得到 32 字节 AES 密钥 → 每条 cookie 是 `v10` + 12 字节 nonce + 密文 + 16 字节 tag。
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import shutil
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0")
API = "https://api.bilibili.com"
REFERER = "https://www.bilibili.com/"

# 浏览器的用户数据目录（按"最可能装了 B 站"的顺序找）
BROWSER_ROOTS: tuple[tuple[str, str], ...] = (
    ("Edge", r"Microsoft\Edge\User Data"),
    ("Chrome", r"Google\Chrome\User Data"),
    ("Brave", r"BraveSoftware\Brave-Browser\User Data"),
    ("Chromium", r"Chromium\User Data"),
)

COOKIE_NAMES = ("SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5", "sid",
                "buvid3", "buvid4")
LOG_PATH = Path("data") / "logs" / "bili.log"


def log(msg: str) -> None:
    """扫码这条链路的问题隔着界面看不清，落一行日志比猜强。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:                                   # noqa: BLE001
        pass


# ---------------------------------------------------------------- 浏览器 cookie
@dataclass
class Profile:
    browser: str
    dir: Path              # ...\User Data\Default
    cookies: Path

    def label(self) -> str:
        return f"{self.browser} / {self.dir.name}"


def _dpapi_unprotect(blob: bytes) -> bytes:
    """CryptUnprotectData：当前用户才能解，换用户/换机器都不行。"""

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def to_blob(data: bytes) -> DATA_BLOB:
        buf = ctypes.create_string_buffer(data, len(data))
        return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    src, dst = to_blob(blob), DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise OSError("CryptUnprotectData 失败（不是当前用户加密的？）")
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def find_profiles() -> list[Profile]:
    """本机所有能读到的浏览器配置目录（含 Default / Profile 1..N）。"""
    local = Path(os.environ.get("LOCALAPPDATA") or "")
    out: list[Profile] = []
    for name, rel in BROWSER_ROOTS:
        root = local / rel
        if not root.is_dir():
            continue
        for prof in sorted(root.iterdir()):
            if not prof.is_dir():
                continue
            if prof.name != "Default" and not prof.name.startswith("Profile"):
                continue
            ck = prof / "Network" / "Cookies"
            if not ck.exists():
                ck = prof / "Cookies"          # 老版本 Chromium
            if ck.exists():
                out.append(Profile(name, prof, ck))
    return out


def _aes_key(profile: Profile) -> bytes | None:
    """从 Local State 拿 DPAPI 保护的 AES 密钥。"""
    try:
        state = json.loads((profile.dir.parent / "Local State").read_text(encoding="utf-8"))
        enc = base64.b64decode(state["os_crypt"]["encrypted_key"])
    except Exception:                                  # noqa: BLE001
        return None
    if enc[:5] != b"DPAPI":
        return None
    try:
        return _dpapi_unprotect(enc[5:])
    except Exception:                                  # noqa: BLE001
        return None


def _decrypt(enc: bytes, key: bytes | None) -> str:
    """v10/v11 = AES-256-GCM；更老的 = 直接 DPAPI；v20（App-Bound）= 不支持。"""
    if not enc:
        return ""
    if enc[:3] in (b"v10", b"v11"):
        if not key:
            return ""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce, payload = enc[3:15], enc[15:]
            return AESGCM(key).decrypt(nonce, payload, None).decode("utf-8", "ignore")
        except Exception:                              # noqa: BLE001
            return ""
    if enc[:3] == b"v20":
        return ""                                      # 新版 Chrome 绑定进程，读不了
    try:
        return _dpapi_unprotect(enc).decode("utf-8", "ignore")
    except Exception:                                  # noqa: BLE001
        return ""


def read_cookies(profiles: Iterable[Profile] | None = None,
                 hosts: tuple[str, ...] = ("bilibili.com",)) -> dict[str, str]:
    """把本机浏览器里 bilibili 的 cookie 读出来。

    数据库被浏览器独占（开着就锁），所以先复制到临时目录再读；
    同一站点的同名 cookie 可能有多份（不同 path），取**最新 expiry** 那份。
    """
    best: dict[str, tuple[float, str]] = {}
    for prof in (profiles if profiles is not None else find_profiles()):
        key = _aes_key(prof)
        tmp = Path(tempfile.mkdtemp(prefix="dshk_ck_")) / "Cookies"
        try:
            shutil.copy2(prof.cookies, tmp)
        except Exception:                              # noqa: BLE001
            continue
        try:
            con = sqlite3.connect(tmp)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT host_key,name,value,encrypted_value,expires_utc FROM cookies").fetchall()
        except Exception:                              # noqa: BLE001
            rows = []
        finally:
            try:
                con.close()
            except Exception:                          # noqa: BLE001
                pass
            try:
                tmp.unlink()
                tmp.parent.rmdir()
            except Exception:                          # noqa: BLE001
                pass
        for r in rows:
            host = str(r["host_key"] or "")
            if not any(h in host for h in hosts):
                continue
            name = str(r["name"] or "")
            if name not in COOKIE_NAMES:
                continue
            val = str(r["value"] or "") or _decrypt(bytes(r["encrypted_value"] or b""), key)
            if not val:
                continue
            exp = float(r["expires_utc"] or 0)
            if name not in best or exp >= best[name][0]:
                best[name] = (exp, val)
    return {k: v for k, (_, v) in best.items()}


# ---------------------------------------------------------------- 接口
class BiliError(RuntimeError):
    pass


@dataclass
class Item:
    """一节课（或一个合集/收藏夹里的一条）。"""
    title: str
    bvid: str = ""
    page: int = 1
    duration: float = 0.0
    n_pages: int = 1
    upper: str = ""
    season_id: int = 0
    media_id: int = 0

    def url(self, t: float | None = None) -> str:
        if not self.bvid:
            return ""
        q = {"p": self.page}
        if t:
            q["t"] = int(t)
        return f"https://www.bilibili.com/video/{self.bvid}?{urllib.parse.urlencode(q)}"

    def as_dict(self) -> dict:
        return {"title": self.title, "bvid": self.bvid, "page": self.page,
                "duration": self.duration, "n_pages": self.n_pages,
                "upper": self.upper, "url": self.url()}


@dataclass
class Session:
    """一次只读会话：cookie + 一个可替换的 HTTP 传输层（测试用假的）。"""
    cookies: dict[str, str] = field(default_factory=dict)
    transport: Callable[[str, float], str] | None = None   # (url, timeout) -> body
    last_set_cookies: list[str] = field(default_factory=list)

    def logged_in(self) -> bool:
        return bool(self.cookies.get("SESSDATA"))

    # -- HTTP --
    def _get(self, url: str, timeout: float = 12.0) -> str:
        self.last_set_cookies = []
        if self.transport is not None:
            return self.transport(url, timeout)
        req = urllib.request.Request(url, headers={
            "User-Agent": UA, "Referer": REFERER, "Origin": "https://www.bilibili.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cookie": "; ".join(f"{k}={v}" for k, v in self.cookies.items()),
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            try:
                self.last_set_cookies = list(r.headers.get_all("Set-Cookie") or [])
            except Exception:                           # noqa: BLE001
                self.last_set_cookies = []
            return r.read().decode("utf-8", "ignore")

    def api(self, path: str, **params) -> dict:
        url = f"{API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
        try:
            data = json.loads(self._get(url))
        except Exception as e:                          # noqa: BLE001
            raise BiliError(f"请求失败：{type(e).__name__}") from e
        code = int(data.get("code", -1))
        if code != 0:
            raise BiliError(f"B 站返回 {code}：{data.get('message') or '未知错误'}")
        return data.get("data") or {}

    # -- 数据 --
    def ensure_fingerprint(self) -> None:
        """补齐 buvid3/buvid4。

        只有 SESSDATA 而无浏览器指纹时，`x/web-interface/nav` 常被风控打回 -352，
        界面上就表现成"扫了码但没登录"。这一步拿一次指纹补上，成本一个请求。
        """
        if self.cookies.get("buvid3"):
            return
        try:
            data = json.loads(self._get(f"{API}/x/frontend/finger/spi"))
            d = data.get("data") or {}
            if d.get("b_3"):
                self.cookies["buvid3"] = str(d["b_3"])
            if d.get("b_4"):
                self.cookies["buvid4"] = str(d["b_4"])
            log(f"finger: buvid3={'有' if self.cookies.get('buvid3') else '无'}")
        except Exception as e:                          # noqa: BLE001
            log(f"finger 失败 {type(e).__name__}")

    def whoami(self) -> dict:
        self.ensure_fingerprint()
        d = self.api("/x/web-interface/nav")
        return {"ok": bool(d.get("isLogin")), "uname": d.get("uname") or "",
                "mid": int(d.get("mid") or 0), "vip": bool((d.get("vipStatus") or 0))}

    def fav_folders(self) -> list[dict]:
        me = self.whoami()
        if not me["mid"]:
            return []
        d = self.api("/x/v3/fav/folder/created/list-all", up_mid=me["mid"])
        return [{"id": int(f.get("id") or 0), "title": str(f.get("title") or ""),
                 "count": int(f.get("media_count") or 0)}
                for f in (d.get("list") or [])]

    def fav_items(self, media_id: int, pages: int = 4) -> list[Item]:
        out: list[Item] = []
        for pn in range(1, max(1, pages) + 1):
            d = self.api("/x/v3/fav/resource/list", media_id=int(media_id),
                         pn=pn, ps=20, platform="web")
            medias = d.get("medias") or []
            for m in medias:
                if int(m.get("type") or 2) != 2:        # 只要视频
                    continue
                out.append(Item(
                    title=str(m.get("title") or "").strip(),
                    bvid=str(m.get("bvid") or ""),
                    duration=float(m.get("duration") or 0),
                    n_pages=int(m.get("page") or 1),
                    upper=str((m.get("upper") or {}).get("name") or ""),
                    media_id=int(media_id)))
            if not d.get("has_more"):
                break
        return out

    def my_seasons(self) -> list[dict]:
        me = self.whoami()
        if not me["mid"]:
            return []
        try:
            d = self.api("/x/polymer/web-space/seasons_series_list",
                         mid=me["mid"], page_num=1, page_size=20)
        except BiliError:
            return []
        out = []
        for it in ((d.get("items_lists") or {}).get("seasons_list") or []):
            meta = it.get("meta") or {}
            out.append({"id": int(meta.get("season_id") or 0),
                        "title": str(meta.get("name") or ""),
                        "count": int((it.get("page") or {}).get("total") or 0),
                        "mid": me["mid"]})
        return out

    def season_items(self, season_id: int, mid: int, pages: int = 3) -> list[Item]:
        """合集里的每一 P。合集才是"一门课"最自然的单位。"""
        out: list[Item] = []
        for pn in range(1, max(1, pages) + 1):
            d = self.api("/x/polymer/web-space/seasons_archives_list",
                         mid=int(mid), season_id=int(season_id),
                         sort_reverse="false", page_num=pn, page_size=30)
            archives = d.get("archives") or []
            for a in archives:
                out.append(Item(
                    title=str(a.get("title") or "").strip(),
                    bvid=str(a.get("bvid") or ""),
                    duration=float(a.get("duration") or 0),
                    season_id=int(season_id)))
            total = int(d.get("page", {}).get("total") or 0)
            if total and len(out) >= total:
                break
        return out

    def history(self, pages: int = 1) -> list[Item]:
        """观看历史：最适合"我刚看的课，直接导进来"。"""
        out: list[Item] = []
        cursor = ""
        for _ in range(max(1, pages)):
            d = self.api("/x/web-interface/history/cursor", ps=30, type="archive",
                         max=0, view_at=cursor or 0, business="")
            lst = d.get("list") or []
            for h in lst:
                hist = h.get("history") or {}
                out.append(Item(
                    title=str(h.get("title") or "").strip(),
                    bvid=str(hist.get("bvid") or ""),
                    page=int(hist.get("page") or 1),
                    duration=float(h.get("duration") or 0),
                    upper=str((h.get("owner") or {}).get("name") or "")))
            if not lst:
                break
            cursor = str((d.get("cursor") or {}).get("view_at") or "")
        return out


def open_session(transport: Callable[[str, float], str] | None = None) -> Session:
    """读浏览器 cookie 建一个只读会话。"""
    return Session(cookies=read_cookies(), transport=transport)


def hms(sec: float) -> str:
    sec = max(0, int(sec or 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def open_in_browser(url: str) -> bool:
    """用系统默认浏览器打开（登录页、视频页、带时间点的深链）。"""
    if not url:
        return False
    try:
        os.startfile(url)                               # noqa: S606 (Windows 桌面程序)
        return True
    except Exception:                                   # noqa: BLE001
        try:
            import webbrowser

            return bool(webbrowser.open(url))
        except Exception:                               # noqa: BLE001
            return False


def touch(seconds: float = 0.0) -> None:
    """给测试用的时间戳（保持模块可注入，不依赖真时钟）。"""
    time.sleep(seconds)

# ---------------------------------------------------------------- 登录（扫码）
# 为什么有第二条路：Edge/Chrome 开着的时候，cookie 库是**独占锁**的
# （Windows 上连只读都打不开），此时"读浏览器 cookie"必然失败。与其让用户
# 关浏览器，不如自己走一次扫码登录：key 和结果都在我们手里，登录一次存下来
# 长期复用，不碰浏览器、不碰密码。
LOGIN_STORE = Path("data") / "bili_session.json"
PASSPORT = "https://passport.bilibili.com"


def qr_generate(transport: Callable[[str, float], str] | None = None) -> dict:
    """要一个登录二维码：返回 {url, key}。url 给用户扫（我们画成图）。"""
    s = Session(transport=transport)
    url = f"{PASSPORT}/x/passport-login/web/qrcode/generate"
    try:
        data = json.loads(s._get(url))
    except Exception as e:                              # noqa: BLE001
        raise BiliError(f"拿不到登录二维码：{type(e).__name__}") from e
    if int(data.get("code", -1)) != 0:
        raise BiliError(f"B 站返回 {data.get('message') or data.get('code')}")
    d = data.get("data") or {}
    key = str(d.get("qrcode_key") or "")
    url = _normalize_scan_url(str(d.get("url") or ""), key)
    log(f"generate key={'有' if key else '无'} url={url}")
    return {"url": url, "key": key}


def _normalize_scan_url(url: str, key: str) -> str:
    """把二维码地址的 `from` 补成 main-fe-header。

    接口返回的地址里 `from=` 是空的，而网页端扫码那一版永远是
    `from=main-fe-header`（和轮询的 `source` 对应）。空值扫码时容易落到
    另一个登录会话里：手机显示"登录成功"，我们这边轮询永远停在 86101。
    """
    if not url:
        return url
    parts = urllib.parse.urlsplit(url)
    q = dict(urllib.parse.parse_qsl(parts.query))
    if key and not q.get("qrcode_key"):
        q["qrcode_key"] = key
    if not q.get("from"):
        q["from"] = "main-fe-header"
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(q), parts.fragment))


def qr_poll(key: str, transport: Callable[[str, float], str] | None = None) -> dict:
    """轮询问"扫了没"。

    code: 0 成功 / 86101 未扫码 / 86090 已扫码待确认 / 86038 过期
    成功时返回的 `url` 里带着 SESSDATA 等参数——直接解析出 cookie 就行，
    不用再请求一次。
    """
    s = Session(transport=transport)
    url = (f"{PASSPORT}/x/passport-login/web/qrcode/poll"
           f"?qrcode_key={urllib.parse.quote(key)}&source=main-fe-header")
    try:
        data = json.loads(s._get(url))
    except Exception as e:                              # noqa: BLE001
        raise BiliError(f"轮询失败：{type(e).__name__}") from e
    d = data.get("data") or {}
    code = int(d.get("code", -1))
    cookies: dict[str, str] = {}
    if code == 0:
        cookies = _cookies_from_payload(d, s.last_set_cookies)
        if not cookies.get("SESSDATA"):
            cookies.update(harvest_cookies(str(d.get("url") or "")))
    raw_url = str(d.get("url") or "")
    qkeys = [k for k, _ in urllib.parse.parse_qsl(urllib.parse.urlparse(raw_url).query)]
    log(f"poll code={code} msg={d.get('message') or ''} url={raw_url[:28]!r} "
        f"qkeys={qkeys} cookie_info={bool((d.get('cookie_info') or {}).get('cookies'))} "
        f"setcookie={[c.split('=')[0] for c in s.last_set_cookies][:6]} "
        f"got={sorted(cookies)}")
    return {"code": code, "message": str(d.get("message") or ""),
            "cookies": cookies, "ok": code == 0 and bool(cookies)}


def _cookies_from_payload(d: dict, set_cookies: list[str] | None = None) -> dict[str, str]:
    """从扫码成功的结果里抠出 cookie。

    三条路都要试，因为 B 站不同端给的不一样：
    1. `url` 的查询串（老版网页扫码）；
    2. `cookie_info.cookies`（App/H5 那一套）；
    3. **响应头 Set-Cookie**——真浏览器就是靠这个种下 SESSDATA 的，
       实测 code=0 且 url 有值、查询串里却没有 SESSDATA，凭据在这儿。
    """
    out: dict[str, str] = {}

    raw = str(d.get("url") or "")
    for candidate in (raw, urllib.parse.unquote(raw)):
        for k, v in urllib.parse.parse_qsl(urllib.parse.urlparse(candidate).query):
            if k in COOKIE_NAMES and v:
                out.setdefault(k, v)

    for c in ((d.get("cookie_info") or {}).get("cookies") or []):
        name = str(c.get("name") or "")
        if name in COOKIE_NAMES and c.get("value"):
            out[name] = str(c["value"])

    for line in set_cookies or []:
        head = line.split(";")[0]
        if "=" not in head:
            continue
        name, _, value = head.partition("=")
        name = name.strip()
        if name in COOKIE_NAMES and value.strip():
            out[name] = value.strip()
    return out


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """自己走跳转：urllib 默认把 3xx 吞掉，中间的 Set-Cookie 就丢了。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D102
        return None


def harvest_cookies(url: str, timeout: float = 12.0,
                    max_hops: int = 6) -> dict[str, str]:
    """顺着登录跳转链把 Set-Cookie 一路收下来。

    为什么需要这一步：轮询返回 `code=0` 时只给一个**跳转地址**，
    真正的 SESSDATA 是访问这个地址时由 `passport.biligame.com/crossDomain`
    这类接口用 Set-Cookie 种下的（真浏览器就是这么拿到登录态的）。
    只解析 url 查询串，会得到"扫码成功但什么也没拿到"。
    """
    if not url:
        return {}
    opener = urllib.request.build_opener(_NoRedirect)
    cur, jars = url, []
    for _ in range(max(1, max_hops)):
        req = urllib.request.Request(cur, headers={
            "User-Agent": UA, "Referer": REFERER, "Origin": "https://www.bilibili.com",
            "Accept": "text/html,application/xhtml+xml,*/*",
        })
        try:
            resp = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:            # 3xx 也走这里，正是我们要的头
            resp = e
        except Exception as e:                         # noqa: BLE001
            log(f"harvest 失败 {type(e).__name__}")
            break
        try:
            jars += list(resp.headers.get_all("Set-Cookie") or [])
        except Exception:                              # noqa: BLE001
            pass
        loc = resp.headers.get("Location") if getattr(resp, "headers", None) else None
        try:
            resp.close()
        except Exception:                              # noqa: BLE001
            pass
        if not loc:
            break
        cur = urllib.parse.urljoin(cur, loc)
    cookies = _cookies_from_payload({}, jars)
    log(f"harvest hops={len(jars)} got={sorted(cookies)}")
    return cookies


def save_login(cookies: dict[str, str], path: Path | None = None) -> Path:
    """把登录态存到本机（当前用户可读）。不写日志、不进数据库。"""
    p = Path(path or LOGIN_STORE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cookies": cookies, "saved_at": time.time()},
                            ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:                                   # noqa: BLE001
        pass
    return p


def load_login(path: Path | None = None) -> dict[str, str]:
    p = Path(path or LOGIN_STORE)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}
    ck = d.get("cookies") or {}
    return {k: str(v) for k, v in ck.items() if k in COOKIE_NAMES and v}


def forget_login(path: Path | None = None) -> None:
    try:
        Path(path or LOGIN_STORE).unlink(missing_ok=True)
    except Exception:                                   # noqa: BLE001
        pass


def login_state(transport: Callable[[str, float], str] | None = None) -> dict:
    """现在的登录状态：先试自己存过的，再试浏览器 cookie。

    失败时把原因一起带出去（`error`），否则界面只能说"没登录"，
    而"没登录"和"被风控成 -352"在处理上是两件事。
    """
    err = ""
    saved = load_login()
    if saved:
        try:
            sess = Session(cookies=dict(saved), transport=transport)
            me = sess.whoami()
            if me.get("ok"):
                if sess.cookies != saved:
                    save_login(sess.cookies)        # 补上的指纹一起存下来
                return {"logged_in": True, "via": "saved", "uname": me.get("uname", ""),
                        "mid": me.get("mid", 0), "cookies": sess.cookies}
        except BiliError as e:
            err = str(e)
    ck = read_cookies()
    if ck.get("SESSDATA"):
        try:
            me = Session(cookies=dict(ck), transport=transport).whoami()
            if me.get("ok"):
                return {"logged_in": True, "via": "browser", "uname": me.get("uname", ""),
                        "mid": me.get("mid", 0), "cookies": ck}
        except BiliError as e:
            err = str(e)
    return {"logged_in": False, "via": "", "uname": "", "mid": 0, "cookies": ck,
            "error": err}


def qr_png(url: str, scale: int = 6, border: int = 2) -> bytes:
    """二维码 → PNG 字节（界面直接丢给 QPixmap）。

    box_size 直接定死：默认 10 会画出 400 多像素的图，塞进对话框要么被压扁
    要么被裁一半——扫不出来。6 像素/模块 + 2 模块白边，220 像素上下正好。
    """
    import io

    import qrcode

    q = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
                      box_size=max(2, int(scale)), border=max(1, int(border)))
    q.add_data(url)
    q.make(fit=True)
    buf = io.BytesIO()
    q.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
    return buf.getvalue()


def open_browser_login() -> bool:
    """跳转到 B 站网页登录（扫码/密码都行）。用户登完再回来点"读浏览器"。"""
    return open_in_browser(f"{PASSPORT}/login")
