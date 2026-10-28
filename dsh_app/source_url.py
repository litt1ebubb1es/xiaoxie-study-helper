"""从链接导入课程（方案 3.4 的替代路径：不录屏、不下载视频）。

两条路，按代价从低到高：

1. **只取字幕**（最省）：B 站有 CC/AI 字幕时，直接拿字幕 JSON → 真正的逐句时间戳。
   不需要下载视频、不需要转写模型。代价：AI 字幕需要登录 Cookie（实测未登录返回 0 条）。
2. **只下音频**（兜底）：没有字幕就只抓音轨（`yt-dlp -f bestaudio`），交给转写后端。
   一节 45 分钟的课，纯音频约 20–40 MB，比整段视频小一个量级。

明确不做：下载视频画面。分析用不到画面，下载只会拖慢且吃磁盘。
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 单元测试要把接口指到本地假服务；正式运行用默认值
API_BASE = os.environ.get("DSHK_BILI_API", "https://api.bilibili.com")

BILI_PATTERNS = [
    re.compile(r"bilibili\.com/video/(BV[0-9A-Za-z]+)"),
    re.compile(r"bilibili\.com/video/av(\d+)", re.I),
    re.compile(r"^BV[0-9A-Za-z]{10}$"),
    re.compile(r"^av\d+$", re.I),
]
YT_HOSTS = ("youtube.com", "youtu.be", "m.bilibili.com")


@dataclass
class Probe:
    """链接探测结果——界面先把这些摆给用户看，再决定导不导。"""
    ok: bool
    platform: str = ""            # bilibili | generic
    title: str = ""
    duration: float = 0.0
    n_pages: int = 1
    has_subtitle: bool = False
    subtitle_langs: list[str] = field(default_factory=list)
    note: str = ""
    error: str = ""
    url: str = ""


def platform_of(url: str) -> str:
    u = (url or "").strip()
    if any(p.search(u) for p in BILI_PATTERNS) or "bilibili.com" in u:
        return "bilibili"
    return "generic"


def bvid_of(url: str) -> str:
    """从各种 B 站链接里抠出 BV 号；也接受直接贴 BV 号。

    注意：BILI_PATTERNS 里有不带分组的模式，不能无条件取 group(1)。
    """
    u = (url or "").strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", u)
    if m:
        return m.group(1)
    m = re.search(r"av(\d+)", u, re.I)
    if m:
        return "av" + m.group(1)
    return ""


def _headers(sessdata: str = "") -> dict[str, str]:
    h = {"User-Agent": UA, "Referer": "https://www.bilibili.com",
         "Origin": "https://www.bilibili.com"}
    if sessdata:
        h["Cookie"] = f"SESSDATA={sessdata}"
    return h


# ---------------------------------------------------------------- B 站
def _bili_view(bvid: str, sessdata: str = "") -> dict[str, Any]:
    import httpx

    r = httpx.get(f"{API_BASE}/x/web-interface/view",
                  params={"bvid": bvid}, headers=_headers(sessdata), timeout=12)
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"B站接口返回 {data.get('code')}：{data.get('message')}")
    return data["data"]


def _bili_subtitle_list(bvid: str, cid: int, sessdata: str = "") -> list[dict]:
    import httpx

    r = httpx.get(f"{API_BASE}/x/player/v2",
                  params={"bvid": bvid, "cid": cid},
                  headers=_headers(sessdata), timeout=12)
    data = r.json()
    sub = ((data.get("data") or {}).get("subtitle") or {})
    return list(sub.get("subtitles") or [])


def probe(url: str, sessdata: str = "") -> Probe:
    """探测：能拿到什么、需不需要登录、要不要下音频。"""
    plat = platform_of(url)
    try:
        if plat == "bilibili":
            bvid = bvid_of(url)
            if not bvid:
                return Probe(False, plat, error="没认出 BV 号，请贴完整的视频链接", url=url)
            d = _bili_view(bvid, sessdata)
            pages = d.get("pages") or []
            subs = _bili_subtitle_list(bvid, int(d["cid"]), sessdata)
            langs = [str(s.get("lan") or "") for s in subs]
            note = ""
            if not subs:
                note = ("未拿到字幕。B 站的 AI 字幕需要登录 Cookie（设置页填 SESSDATA）；"
                        "没有字幕就走「只下音频 + 转写」这条路。")
            return Probe(True, plat, title=str(d.get("title") or ""),
                         duration=float(d.get("duration") or 0),
                         n_pages=max(1, len(pages)), has_subtitle=bool(subs),
                         subtitle_langs=langs, note=note, url=url)
        # 通用站点：问 yt-dlp 要元信息（不下载）
        info = ytdlp_info(url)
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        langs = list(subs.keys()) + [k + "(自动)" for k in auto.keys()]
        return Probe(True, "generic", title=str(info.get("title") or ""),
                     duration=float(info.get("duration") or 0),
                     has_subtitle=bool(subs or auto), subtitle_langs=langs[:8],
                     note="" if (subs or auto) else "该链接没提供字幕，只能下音频转写",
                     url=url)
    except Exception as e:  # noqa: BLE001
        return Probe(False, plat, error=f"{type(e).__name__}: {str(e)[:160]}", url=url)


def _pick_sub_url(subs: list[dict]) -> str:
    """优先中文，其次任意。"""
    if not subs:
        return ""
    def score(s: dict) -> int:
        lan = str(s.get("lan") or "").lower()
        if lan.startswith("zh"): return 0
        if "ai" in lan: return 1
        if lan.startswith("en"): return 2
        return 3
    return str(sorted(subs, key=score)[0].get("subtitle_url") or "")


def bili_subtitle_segments(url: str, sessdata: str = "") -> tuple[list[dict], str]:
    """返回 (segments, 说明)。segments 带真实时间戳。"""
    import httpx

    bvid = bvid_of(url)
    if not bvid:
        raise RuntimeError("没认出 BV 号")
    d = _bili_view(bvid, sessdata)
    subs = _bili_subtitle_list(bvid, int(d["cid"]), sessdata)
    sub_url = _pick_sub_url(subs)
    if not sub_url:
        raise RuntimeError(
            "这个视频没有可用的字幕（B站 AI 字幕需要登录 Cookie）。"
            "请在设置页填 SESSDATA，或改用「下音频 + 转写」。")
    if sub_url.startswith("//"):
        sub_url = "https:" + sub_url
    r = httpx.get(sub_url, headers=_headers(sessdata), timeout=20)
    body = r.json()
    segs = []
    for item in body.get("body") or []:
        t0 = float(item.get("from") or 0)
        t1 = float(item.get("to") or t0 + 1)
        txt = str(item.get("content") or "").strip()
        if txt:
            segs.append({"t_start": t0, "t_end": max(t1, t0 + 0.4), "text": txt,
                         "speaker": "", "confidence": 0.9})
    if not segs:
        raise RuntimeError("字幕文件是空的")
    return segs, f"B站字幕 {len(body.get('body') or [])} 句"


# ---------------------------------------------------------------- B 站故事板（零下载取画面）
def _index_seconds(idx: list, duration: float) -> list[float]:
    """把故事板 index 归一成秒。

    实测：BV1d8pU6xEZX 的 index 末值是 155，视频 157 秒 —— 即**秒**为单位；
    但官方文档与部分视频给的是毫秒。所以按量级自适应，不硬编码单位。
    """
    if not idx:
        return []
    vals = [float(x) for x in idx]
    mx = max(vals)
    if duration > 0 and mx > duration * 2:
        return [v / 1000.0 for v in vals]      # 毫秒
    return vals                                 # 已经是秒


def bili_storyboard(bvid: str, cid: int, sessdata: str = "") -> dict:
    """取故事板：整张雪碧图 + 每帧时间 + 网格几何。不需要登录。

    返回 {"sprite_urls": [...], "times": [...], "tile": (w,h), "grid": (cols,rows)}
    """
    import httpx

    r = httpx.get(f"{API_BASE}/x/player/videoshot",
                  params={"bvid": bvid, "cid": cid, "index": 1},
                  headers=_headers(sessdata), timeout=15)
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"故事板接口返回 {data.get('code')}：{data.get('message')}")
    d = data.get("data") or {}
    images = list(d.get("image") or [])
    idx = list(d.get("index") or [])
    if not images or not idx:
        raise RuntimeError("这个视频没有故事板（可能是直播或未生成）")
    try:
        dur = float(_bili_view(bvid, sessdata).get("duration") or 0)
    except Exception:  # noqa: BLE001
        dur = 0.0
    urls = [("https:" + u if u.startswith("//") else u) for u in images]
    return {"sprite_urls": urls,
            "times": _index_seconds(idx, dur),
            "tile": (int(d.get("img_x_size") or 480), int(d.get("img_y_size") or 270)),
            "grid": (int(d.get("img_x_len") or 10), int(d.get("img_y_len") or 10))}


def storyboard_frame_png(sb: dict, t: float, sessdata: str = "") -> bytes:
    """取离 t 最近的那一帧，裁成独立 PNG。只下载它所在的那张雪碧图。"""
    import httpx
    from PIL import Image

    times = sb["times"]
    if not times:
        raise RuntimeError("故事板没有帧时间")
    k = min(range(len(times)), key=lambda i: abs(times[i] - float(t)))
    cols, rows = sb["grid"]
    per_sheet = max(1, cols * rows)
    sheet_i = min(len(sb["sprite_urls"]) - 1, k // per_sheet)
    pos = k % per_sheet
    gx, gy = pos % cols, pos // cols
    tw, th = sb["tile"]
    raw = httpx.get(sb["sprite_urls"][sheet_i], headers=_headers(sessdata),
                    timeout=20).content
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    box = (gx * tw, gy * th, gx * tw + tw, gy * th + th)
    if box[2] > im.width or box[3] > im.height:   # 末张图可能不满格
        raise RuntimeError(f"故事板几何不一致：想裁 {box}，图只有 {im.size}")
    tile = im.crop(box)
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- 通用（yt-dlp）
def ytdlp_info(url: str, cookie_file: str = "") -> dict[str, Any]:
    try:
        import yt_dlp
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "没装 yt-dlp：.venv\\Scripts\\python.exe -m pip install yt-dlp") from e
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True,
                            "noplaylist": True, "socket_timeout": 20}
    if cookie_file:
        opts["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _parse_vtt(text: str) -> list[dict]:
    """极简 WebVTT 解析：取时间轴行与紧跟的文字。"""
    segs: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        s = line.strip()
        if "-->" in s:
            m = re.match(r"([\d:.]+)\s*-->\s*([\d:.]+)", s)
            if m:
                cur = {"t_start": _vtt_time(m.group(1)),
                       "t_end": _vtt_time(m.group(2)), "text": ""}
            continue
        if not s or s.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        if cur is not None:
            if cur["text"]:
                cur["text"] += " "
            cur["text"] += re.sub(r"<[^>]+>", "", s)
            if len(cur["text"]) > 0:
                segs.append({"t_start": cur["t_start"], "t_end": cur["t_end"],
                             "text": cur["text"].strip(), "speaker": "",
                             "confidence": 0.85})
                cur = None
    return segs


def _vtt_time(s: str) -> float:
    parts = s.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:  # noqa: BLE001
        return 0.0


def ytdlp_subtitle_segments(url: str, lang: str = "zh-Hans",
                            cookie_file: str = "") -> tuple[list[dict], str]:
    """让 yt-dlp 只下字幕（skip_download），解析成 segments。"""
    try:
        import yt_dlp
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "没装 yt-dlp：.venv\\Scripts\\python.exe -m pip install yt-dlp") from e
    tmp = Path("data/tmp_subs")
    tmp.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": [lang, "zh", "zh-CN", "zh-TW", "en"],
        "subtitlesformat": "vtt", "outtmpl": str(tmp / "%(id)s.%(ext)s"),
        "noplaylist": True, "socket_timeout": 20,
    }
    if cookie_file:
        opts["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = sorted(tmp.glob("*.vtt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("没下到字幕文件（该链接可能没有字幕）")
    text = files[-1].read_text(encoding="utf-8", errors="replace")
    segs = _parse_vtt(text)
    if not segs:
        raise RuntimeError("字幕解析后为空")
    return segs, f"yt-dlp 字幕 {len(segs)} 句（{files[-1].name}）"


def download_audio(url: str, out_dir: str | Path = "data/media",
                   cookie_file: str = "") -> tuple[Path, str]:
    """只下载音轨。返回 (文件路径, 说明)。"""
    try:
        import yt_dlp
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "没装 yt-dlp：.venv\\Scripts\\python.exe -m pip install yt-dlp") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.glob("*")}
    opts: dict[str, Any] = {
        "quiet": True, "no_warnings": True, "format": "bestaudio/best",
        "outtmpl": str(out / "%(title).60s-%(id)s.%(ext)s"),
        "noplaylist": True, "socket_timeout": 30, "retries": 2,
    }
    if cookie_file:
        opts["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        new = [p for p in out.glob("*") if p.name not in before]
        if not new:
            raise RuntimeError("音频下载失败（找不到输出文件）")
        path = new[-1]
    mb = path.stat().st_size / 1024 / 1024
    return path, f"已下载音轨 {path.name}（{mb:.1f} MB，画面未下载）"


def estimate_audio_mb(duration_s: float, kbps: int = 96) -> float:
    """给用户一个下载前的体积估计。"""
    return max(0.0, duration_s) * kbps / 8 / 1024