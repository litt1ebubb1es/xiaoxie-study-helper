"""网页视频自动抓取：不靠浏览器插件，也不靠站点 API，走系统回环。

为什么不用插件：插件只能覆盖愿意做插件的站点（B 站、YouTube 有字幕/接口，
学堂在线、学校自建平台、腾讯会议回放、B 站以外的各种网课平台大多没有）。
而且插件要用户自己装、自己点，等于把"别忘了开录"这件事又还给用户。

系统回环（WASAPI loopback）不关心声音从哪来——浏览器、播放器、会议软件都一样。
所以这一层的判据只有一个：**浏览器进程有没有在放声音**。

判定用 Windows 音频会话（IAudioSessionManager2，经 pycaw）：每个在放声音的进程
都有一个 `AudioSessionState.Active` 的会话。会话是系统级的，任何 Chromium/Gecko
系浏览器都覆盖，不需要知道具体是哪个网站。

宁可漏、不可误：只认浏览器进程名（chrome/msedge/firefox/…），
别的程序在放声音（音乐、会议、游戏）不会触发录制。
"""
from __future__ import annotations

import ctypes
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# 浏览器进程名（小写、无 .exe 也行）。覆盖 Chromium/Gecko/国产壳。
BROWSER_PROCS: frozenset[str] = frozenset({
    "chrome", "chromium", "msedge", "firefox", "brave", "opera", "vivaldi",
    "arc", "librewolf", "waterfox", "360se", "360chrome", "360chromex",
    "qqbrowser", "sogouexplorer", "liebao", "maxthon", "theworld", "2345explorer",
    "quark", "uppc", "huaweibrowser",
})

# 窗口标题里的站点特征 → 规范站点名。只用于给这次录制起名字，不影响抓不抓。
SITE_HINTS: tuple[tuple[str, str], ...] = (
    ("bilibili", "哔哩哔哩"), ("哔哩哔哩", "哔哩哔哩"),
    ("youtube", "YouTube"),
    ("douyin", "抖音"), ("抖音", "抖音"),
    ("ixigua", "西瓜视频"), ("西瓜视频", "西瓜视频"),
    ("youku", "优酷"), ("优酷", "优酷"),
    ("iqiyi", "爱奇艺"), ("爱奇艺", "爱奇艺"),
    ("v.qq.com", "腾讯视频"), ("腾讯视频", "腾讯视频"),
    ("xuetangx", "学堂在线"), ("学堂在线", "学堂在线"),
    ("icourse163", "中国大学MOOC"), ("中国大学", "中国大学MOOC"),
    ("chaoxing", "超星学习通"), ("学习通", "超星学习通"),
    ("zhihuishu", "智慧树"), ("智慧树", "智慧树"),
    ("ke.qq.com", "腾讯课堂"), ("腾讯课堂", "腾讯课堂"),
    ("open.163", "网易公开课"), ("网易公开课", "网易公开课"),
    ("coursera", "Coursera"), ("edx", "edX"), ("udemy", "Udemy"),
    ("ted.com", "TED"), ("TED", "TED"),
)

# 浏览器往标题后面加的后缀：「标题 - Google Chrome」「标题 - Microsoft​ Edge」
_BROWSER_SUFFIX = tuple(f" - {n}" for n in (
    "Google Chrome", "Microsoft​ Edge", "Microsoft Edge", "Mozilla Firefox",
    "Brave", "Opera", "Vivaldi", "360安全浏览器", "360极速浏览器", "QQ浏览器",
    "搜狗高速浏览器", "猎豹浏览器", "Maxthon", "Chromium", "Arc",
))


@dataclass
class AudioSessionInfo:
    proc: str = ""        # 进程名（小写，无 .exe）
    pid: int = 0
    state: int = 0        # 0=Inactive 1=Active 2=Expired
    display: str = ""     # 会话显示名（浏览器常为空）

    @property
    def active(self) -> bool:
        return self.state == 1

    def as_dict(self) -> dict[str, Any]:
        return {"proc": self.proc, "pid": self.pid, "state": self.state,
                "display": self.display}


def _norm_proc(name: str) -> str:
    n = (name or "").strip().lower()
    return n[:-4] if n.endswith(".exe") else n


def list_sessions() -> list[AudioSessionInfo]:
    """当前所有音频会话。pycaw 不可用时返回空表（不是抛异常）。"""
    try:
        from pycaw.utils import AudioUtilities  # 只在需要时才 import
    except Exception:
        return []
    out: list[AudioSessionInfo] = []
    try:
        for s in AudioUtilities.GetAllSessions():
            try:
                proc = _norm_proc(s.Process.name()) if s.Process is not None else ""
            except Exception:
                proc = ""
            try:
                pid = int(getattr(s, "ProcessId", 0) or 0)
            except Exception:
                pid = 0
            try:
                state = int(s.State)
            except Exception:
                state = 0
            out.append(AudioSessionInfo(proc=proc, pid=pid, state=state,
                                        display=str(getattr(s, "DisplayName", "") or "")))
    except Exception:
        return out
    return out


def _match_browser(proc: str, procs: frozenset[str] | set[str]) -> bool:
    p = _norm_proc(proc)
    if not p:
        return False
    if p in procs:
        return True
    # 有些壳浏览器带版本后缀（chrome_1、msedge_preview）
    return any(p.startswith(x) for x in procs if len(x) >= 4)


def browser_playing(sessions: Sequence[AudioSessionInfo] | None = None,
                    procs: frozenset[str] | set[str] = BROWSER_PROCS,
                    ) -> AudioSessionInfo | None:
    """哪个浏览器正在放声音。没有就返回 None。"""
    ss = list_sessions() if sessions is None else list(sessions)
    act = [s for s in ss if s.active and _match_browser(s.proc, procs)]
    if not act:
        return None
    act.sort(key=lambda s: (s.proc, s.pid))
    return act[0]


def window_title_of_pid(pid: int) -> str:
    """属于这个进程的、最"像正文"的窗口标题。

    为什么不能只看前台窗口：自动抓取是被浏览器里的声音触发的，用户此刻的
    前台窗口可能是任何东西（编辑器、聊天、甚至另一个浏览器窗口）。拿前台
    标题当节名，库里就会出现"明明在放概率论、节名却是某某文档"。
    音频会话带 pid，顺着 pid 找它自己的窗口才准。
    """
    if not pid:
        return ""
    try:
        u = ctypes.windll.user32
        best = ""
        buf = ctypes.create_unicode_buffer(512)

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def _cb(hwnd, _lparam):
            nonlocal best
            if not u.IsWindowVisible(hwnd):
                return True
            owner = ctypes.c_ulong()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if int(owner.value) != int(pid):
                return True
            n = u.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            u.GetWindowTextW(hwnd, buf, min(512, n + 2))
            t = (buf.value or "").strip()
            if len(t) > len(best):
                best = t
            return True

        u.EnumWindows(_cb, 0)
        return best
    except Exception:
        return ""


# 这些词出现在标题里，基本就是"一节网课"而不是别的网页。
# 不做成站点白名单（那会漏站点），只用来在同进程的多个窗口之间挑更像的那一个。
LESSON_MARKERS = (
    "哔哩哔哩", "bilibili", "学堂在线", "xuetangx", "中国大学mooc", "icourse163",
    "网易公开课", "腾讯课堂", "超星", "学习通", "智慧树", "慕课", "mooc",
    "网课", "公开课", "第", "讲", "课",
)


def _score_lesson_title(title: str) -> int:
    """给一个窗口标题打分：越像"正在上的那节课"分越高。"""
    t = (title or "").lower()
    if not t:
        return 0
    score = 0
    for m in LESSON_MARKERS:
        if m in t:
            score += 2 if len(m) >= 3 else 1
    if len(t) >= 12:
        score += 1
    return score


def _process_parent(pid: int) -> int:
    """这个进程的父进程 id（拿不到返回 0）。

    Chromium 系浏览器的音频会话挂在**渲染/音频子进程**上，窗口却在浏览器主
    进程里。所以"从音频会话 pid 找窗口"必须先往上走一级，否则永远找不到。
    """
    if not pid:
        return 0
    try:
        k32 = ctypes.windll.kernel32
        TH32CS_SNAPPROCESS = 0x00000002

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                        ("th32ProcessID", ctypes.c_ulong),
                        ("th32DefaultHeapID", ctypes.c_void_p),
                        ("th32ModuleID", ctypes.c_ulong),
                        ("cntThreads", ctypes.c_ulong),
                        ("th32ParentProcessID", ctypes.c_ulong),
                        ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", ctypes.c_ulong),
                        ("szExeFile", ctypes.c_char * 260)]

        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap:
            return 0
        try:
            e = PROCESSENTRY32()
            e.dwSize = ctypes.sizeof(PROCESSENTRY32)
            ok = k32.Process32First(snap, ctypes.byref(e))
            while ok:
                if int(e.th32ProcessID) == int(pid):
                    return int(e.th32ParentProcessID)
                ok = k32.Process32Next(snap, ctypes.byref(e))
        finally:
            k32.CloseHandle(snap)
    except Exception:
        return 0
    return 0


def lesson_title_of_pid(pid: int) -> str:
    """在这一进程（含最多两级父进程）的窗口里，挑最像"一节网课"的标题。

    浏览器所有窗口都由主进程持有，光靠 pid 分不出用户在听哪个标签页。于是
    按标题打分：带站点名/章节字样的优先。都不像就返回空串，让调用方退回
    前台窗口标题（那种情况下前台大概就是对的）。
    """
    if not pid:
        return ""
    titles: list[str] = []
    cur = int(pid)
    for _ in range(3):                        # 音频子进程 → 浏览器主进程
        titles.extend(_window_titles_of_pid(cur))
        cur = _process_parent(cur)
        if not cur:
            break
    best, best_score = "", 0
    try:
        for title in titles:
            sc = _score_lesson_title(title)
            if sc > best_score or (sc == best_score and len(title) > len(best)):
                best, best_score = title, sc
    except Exception:
        return ""
    return best if best_score >= 2 else ""


def _window_titles_of_pid(pid: int) -> list[str]:
    """这个进程所有可见顶层窗口的标题。"""
    if not pid:
        return []
    out: list[str] = []
    try:
        u = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(512)

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def _cb(hwnd, _lparam):
            if not u.IsWindowVisible(hwnd):
                return True
            owner = ctypes.c_ulong()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if int(owner.value) != int(pid):
                return True
            n = u.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            u.GetWindowTextW(hwnd, buf, min(512, n + 2))
            t = (buf.value or "").strip()
            if t:
                out.append(t)
            return True

        u.EnumWindows(_cb, 0)
    except Exception:
        return out
    return out


def foreground_title() -> str:
    """当前前台窗口标题。拿不到就返回空串（不是在放视频的时候没用）。"""
    try:
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return ""
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 2)
        u.GetWindowTextW(hwnd, buf, n + 2)
        return buf.value.strip()
    except Exception:
        return ""


def clean_title(title: str) -> str:
    """去掉浏览器和账号加的后缀，留下页面标题。

    Windows 上浏览器标题的典型样子是
    `页面标题 - bilibili - 用户1 - Microsoft​ Edge`：
    浏览器名在最后、账号在它前面，所以要**反复剥**，一次剥一层。
    """
    t = (title or "").strip()
    for _ in range(4):
        before = t
        for suf in _BROWSER_SUFFIX:
            if t.lower().endswith(suf.lower()):
                t = t[: -len(suf)].strip()
                break
        t = re.sub(r"\s*[-–—]\s*(?:用户|Person|Profile)\s*\S*\s*$", "", t).strip()
        if t == before:
            break
    return t.strip(" -—_|·")


def site_of(title: str, proc: str = "") -> str:
    """从标题（和进程名）里认出站点。认不出就回落到进程名，再不认识就"网页视频"。"""
    t = (title or "").lower()
    for key, name in SITE_HINTS:
        if key.lower() in t:
            return name
    p = _norm_proc(proc)
    for key, name in (("msedge", "Edge"), ("chrome", "Chrome"), ("firefox", "Firefox"),
                      ("brave", "Brave"), ("opera", "Opera"), ("qqbrowser", "QQ浏览器"),
                      ("360", "360浏览器"), ("sogou", "搜狗浏览器")):
        if p.startswith(key):
            return f"网页视频（{name}）"
    return "网页视频"


def course_of(site: str) -> str:
    """自动录制时归到哪门课：按站点归，比按标题归稳定（同一站点的课放一起）。"""
    s = (site or "").strip() or "网页视频"
    return f"{s}·网课"


@dataclass
class CaptureMeta:
    proc: str = ""
    pid: int = 0
    page_title: str = ""
    site: str = ""
    course: str = ""
    started_wall: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"proc": self.proc, "pid": self.pid, "page_title": self.page_title,
                "site": self.site, "course": self.course,
                "started_wall": self.started_wall}


def describe(meta: CaptureMeta) -> str:
    t = meta.page_title or "（没拿到页面标题）"
    return f"{meta.site}：{t}"


@dataclass
class WatcherConfig:
    start_after_s: float = 2.5      # 连续出声这么久才开始录（躲开"点开就关"的误触发）
    stop_after_s: float = 25.0      # 安静这么久才停（课上暂停、缓冲都算）
    poll_s: float = 1.0
    min_keep_s: float = 5.0         # 短于这个的录制标成"太短"，但仍保留（可能就是想截一段）
    procs: frozenset[str] = BROWSER_PROCS


class WebVideoWatcher:
    """盯着浏览器音频会话，自己起停录制。

    刻意做成"纯状态机 + 注入探针"，这样能用一个假探针把时序全部测出来，
    不需要真的在机器上放一段视频。
    """

    def __init__(self, config: WatcherConfig | None = None,
                 probe: Callable[[], AudioSessionInfo | None] | None = None,
                 title_fn: Callable[[], str] | None = None,
                 pid_title_fn: Callable[[int], str] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 on_start: Callable[[CaptureMeta], None] | None = None,
                 on_stop: Callable[[CaptureMeta, str, float], None] | None = None,
                 on_state: Callable[[dict], None] | None = None,
                 is_quiet: Callable[[], bool] | None = None):
        self.cfg = config or WatcherConfig()
        self._probe = probe or (lambda: browser_playing(procs=self.cfg.procs))
        self._title = title_fn or foreground_title
        # 页面标题优先按"出声的那个进程"取，取不到才退回前台窗口
        self._pid_title = pid_title_fn or lesson_title_of_pid
        self._clock = clock
        self.on_start = on_start or (lambda m: None)
        self.on_stop = on_stop or (lambda m, why, dur: None)
        self.on_state = on_state or (lambda d: None)
        # 会话状态会"赖着"Active（浏览器把流开着但没声音）。所以停录要两个条件同时满足：
        # 会话不活跃 **且** 回环里真的没声音了。is_quiet 由上层用录音电平实现。
        self.is_quiet = is_quiet
        self.state = "idle"        # idle / arming / recording
        self.meta = CaptureMeta()
        self._armed_at = 0.0
        self._last_play = 0.0
        self._started_at = 0.0
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error = ""

    # -- 一次轮询（可单独调用，测试就测它） ------------------------------
    def poll_once(self) -> dict:
        now = self._clock()
        info = None
        try:
            info = self._probe()
        except Exception as e:                     # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            info = None
        playing = info is not None and info.active

        if self.state == "idle":
            if playing:
                self.state = "arming"
                self._armed_at = now
                self.meta = CaptureMeta(proc=info.proc, pid=info.pid)
        elif self.state == "arming":
            if not playing:
                self.state = "idle"
            elif now - self._armed_at >= self.cfg.start_after_s:
                title = ""
                try:
                    if info.pid:
                        title = self._pid_title(int(info.pid))
                except Exception:
                    title = ""
                if not title:
                    try:
                        title = self._title()
                    except Exception:
                        title = ""
                self.meta.page_title = clean_title(title) or self.meta.page_title
                self.meta.site = site_of(self.meta.page_title, info.proc)
                self.meta.course = course_of(self.meta.site)
                self.meta.started_wall = time.time()
                self.state = "recording"
                self._started_at = now
                self._last_play = now
                try:
                    self.on_start(self.meta)
                except Exception as e:             # noqa: BLE001
                    self.last_error = f"on_start: {e}"
        elif self.state == "recording":
            # 电平优先、会话其次：会话可能赖着 Active 但页面已经静音，
            # 也可能会话短暂消失但声音还在（切 tab、播放器重载）。
            audible = playing
            if self.is_quiet is not None:
                try:
                    quiet = self.is_quiet()
                except Exception:
                    quiet = False         # 判不了就当有声，宁可多录也不切段
                if quiet:
                    audible = False       # 会话还在响但页面静音了（被 muted / 视频暂停）
                elif not playing:
                    audible = True        # 会话掉了但电平证明还有声音（切 tab、播放器重载）
            if audible:
                self._last_play = now
            elif now - self._last_play >= self.cfg.stop_after_s:
                dur = now - self._started_at
                self.state = "idle"
                try:
                    self.on_stop(self.meta, "silence", dur)
                except Exception as e:             # noqa: BLE001
                    self.last_error = f"on_stop: {e}"

        snap = {"state": self.state, "proc": (info.proc if info else ""),
                "playing": playing, "title": self.meta.page_title,
                "site": self.meta.site,
                "elapsed": (now - self._started_at) if self.state == "recording" else 0.0,
                "error": self.last_error}
        try:
            self.on_state(snap)
        except Exception:
            pass
        return snap

    # -- 线程包装 --------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, name="webvideo-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_flag.wait(self.cfg.poll_s):
            try:
                self.poll_once()
            except Exception as e:                 # noqa: BLE001
                self.last_error = f"poll: {e}"

    def stop(self, reason: str = "manual") -> None:
        """停止监听；如果正在录，按"手动停"收尾（让上层把这一节的音频收好）。"""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        if self.state == "recording":
            dur = self._clock() - self._started_at
            self.state = "idle"
            try:
                self.on_stop(self.meta, reason, dur)
            except Exception as e:                 # noqa: BLE001
                self.last_error = f"on_stop: {e}"

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


def probe_report(n: int = 3, gap_s: float = 0.4) -> dict:
    """诊断用：连续看几次会话状态，回答"现在能不能自动抓"。"""
    seen: list[dict] = []
    for _ in range(max(1, int(n))):
        ss = list_sessions()
        hit = browser_playing(ss)
        seen.append({"sessions": [s.as_dict() for s in ss],
                     "browser": hit.as_dict() if hit else None})
        time.sleep(gap_s)
    return {"ok": True, "samples": seen,
            "pycaw": _has_pycaw(),
            "browsers_seen": sorted({s["proc"] for x in seen for s in x["sessions"]
                                     if _match_browser(s["proc"], BROWSER_PROCS)}),
            "foreground": foreground_title()}


def _has_pycaw() -> bool:
    try:
        import pycaw  # noqa: F401
        return True
    except Exception:
        return False