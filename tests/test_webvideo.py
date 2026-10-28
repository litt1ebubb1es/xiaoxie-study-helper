"""网页视频自动抓取：不靠插件，靠"浏览器在放声音"这件事。

这一层的价值全在**时序正确**上：太早开始录会把点开就关的广告录进去，
太早停止会把课中间的暂停切成两段。所以用注入的假探针把时序全部钉住。
"""
from __future__ import annotations

import pytest

from dsh_app import webvideo as wv


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeProbe:
    """按剧本返回"浏览器在不在放声音"。"""

    def __init__(self, plan: list[bool]):
        self.plan = list(plan)

    def __call__(self):
        playing = self.plan.pop(0) if self.plan else False
        if not playing:
            return None
        return wv.AudioSessionInfo(proc="chrome", pid=4321, state=1)


@pytest.fixture()
def rig():
    clock = FakeClock()
    events: list[tuple] = []

    def build(plan, start_after=2.5, stop_after=25.0, title="【第3讲】随机变量 - 哔哩哔哩_bilibili"):
        w = wv.WebVideoWatcher(
            wv.WatcherConfig(start_after_s=start_after, stop_after_s=stop_after),
            probe=FakeProbe(plan), title_fn=lambda: title, clock=clock,
            on_start=lambda m: events.append(("start", m.site, m.page_title)),
            on_stop=lambda m, why, dur: events.append(("stop", why, round(dur, 1))),
        )
        return w

    return clock, events, build


# ---------------------------------------------------------------- 时序
def test_starts_after_the_arming_delay_not_immediately(rig):
    clock, events, build = rig
    w = build([True, True, True, True, True], start_after=2.5)
    w.poll_once()                       # 第一次看到声音：只记时，不开始
    assert events == [] and w.state == "arming"
    clock.advance(1.0)
    w.poll_once()
    assert events == [], "还没到 2.5 秒就开始录了"
    clock.advance(1.6)
    w.poll_once()
    assert [e[0] for e in events] == ["start"]
    assert w.state == "recording"


def test_short_blip_never_starts_recording(rig):
    """点开就关（广告、误点）不该留下一个空会话。"""
    clock, events, build = rig
    w = build([True, False, False], start_after=2.5)
    w.poll_once()
    clock.advance(0.5)
    w.poll_once()                        # 声音没了 → 回到 idle
    clock.advance(0.5)
    w.poll_once()
    assert events == [] and w.state == "idle"


def test_stops_only_after_the_silence_window(rig):
    clock, events, build = rig
    w = build([True, True, False, False], start_after=2.5, stop_after=25.0)
    w.poll_once()
    clock.advance(3.0)
    w.poll_once()                        # recording
    assert w.state == "recording"
    clock.advance(10.0)
    w.poll_once()                        # 停 10 秒：课间暂停，不该停录
    assert w.state == "recording" and events[-1][0] == "start"
    clock.advance(20.0)
    w.poll_once()                        # 累计安静 30 秒 > 25 → 停
    assert events[-1][0] == "stop"
    assert events[-1][1] == "silence"
    assert w.state == "idle"


def test_pause_shorter_than_window_keeps_one_session(rig):
    clock, events, build = rig
    w = build([True, True, True, False, True, True, True], start_after=1.0, stop_after=25.0)
    w.poll_once()
    clock.advance(1.5)
    w.poll_once()                        # recording
    clock.advance(5.0)
    w.poll_once()                        # 静音 5 秒
    clock.advance(5.0)
    w.poll_once()                        # 又出声
    assert [e[0] for e in events] == ["start"], "中间暂停不该把一节课切成两段"


def test_manual_stop_finishes_an_active_recording(rig):
    clock, events, build = rig
    w = build([True, True, True], start_after=1.0)
    w.poll_once()
    clock.advance(1.5)
    w.poll_once()
    assert w.state == "recording"
    clock.advance(120.0)
    w.stop(reason="手动停止")
    assert events[-1] == ("stop", "手动停止", 120.0)
    assert w.state == "idle"


# ---------------------------------------------------------------- 判据
def test_non_browser_audio_does_not_trigger():
    """会议/音乐/游戏在放声音不该触发——只认浏览器进程。"""
    ss = [wv.AudioSessionInfo(proc="spotify", pid=1, state=1),
          wv.AudioSessionInfo(proc="gameviewerserver", pid=2, state=1),
          wv.AudioSessionInfo(proc="oopz", pid=3, state=0)]
    assert wv.browser_playing(ss) is None


@pytest.mark.parametrize("proc", ["chrome", "msedge", "firefox", "brave",
                                  "qqbrowser", "360se", "chromium"])
def test_known_browsers_are_recognised(proc):
    ss = [wv.AudioSessionInfo(proc=proc, pid=9, state=1)]
    got = wv.browser_playing(ss)
    assert got is not None and got.proc == proc


def test_inactive_browser_session_is_not_playing():
    ss = [wv.AudioSessionInfo(proc="chrome", pid=9, state=0)]
    assert wv.browser_playing(ss) is None


def test_probe_errors_are_contained(rig):
    clock, events, build = rig

    def boom():
        raise RuntimeError("COM 挂了")

    w = wv.WebVideoWatcher(wv.WatcherConfig(), probe=boom, clock=clock,
                           title_fn=lambda: "x")
    snap = w.poll_once()                  # 不该抛出去
    assert snap["playing"] is False
    assert "COM" in w.last_error


# ---------------------------------------------------------------- 标题与站名
@pytest.mark.parametrize("title,site", [
    ("【概率论】第3讲 随机变量 - 哔哩哔哩_bilibili", "哔哩哔哩"),
    ("Random Variables - YouTube", "YouTube"),
    ("概率论与数理统计 - 学堂在线", "学堂在线"),
    ("第2章 随机变量 - 超星学习通", "超星学习通"),
    ("课程视频 - 中国大学MOOC(慕课)", "中国大学MOOC"),
    ("Lecture 3 - Coursera", "Coursera"),
])
def test_site_detection(title, site):
    assert wv.site_of(title, "chrome") == site


@pytest.mark.parametrize("raw,want", [
    ("第3讲 随机变量 - 哔哩哔哩_bilibili - Google Chrome", "第3讲 随机变量 - 哔哩哔哩_bilibili"),
    ("第3讲 随机变量 - Microsoft\u200b Edge", "第3讲 随机变量"),
    ("第3讲 随机变量 - Mozilla Firefox", "第3讲 随机变量"),
])
def test_browser_suffix_is_stripped(raw, want):
    assert wv.clean_title(raw) == want


def test_unknown_site_falls_back_to_browser_name():
    assert "Edge" in wv.site_of("某个学校自建平台 第3讲", "msedge")
    assert wv.site_of("", "") == "网页视频"


def test_course_groups_by_site_not_by_title():
    a = wv.course_of(wv.site_of("第3讲 - 哔哩哔哩", "chrome"))
    b = wv.course_of(wv.site_of("第4讲 - 哔哩哔哩", "chrome"))
    assert a == b, "同一站点的课要归到同一门课，否则一学期散成几十门"


def test_describe_reads_naturally():
    m = wv.CaptureMeta(site="哔哩哔哩", page_title="第3讲 随机变量")
    assert wv.describe(m) == "哔哩哔哩：第3讲 随机变量"

def test_lingering_active_session_does_not_record_dead_air(rig):
    """会话会赖着 Active（浏览器把音频流开着但没声音）——电平还在就不该停，
    电平没了才停。这是"录了三小时空气"的防线。"""
    clock, events, build = rig
    loud = {"v": 0.3}
    w = wv.WebVideoWatcher(
        wv.WatcherConfig(start_after_s=1.0, stop_after_s=20.0),
        probe=lambda: wv.AudioSessionInfo(proc="chrome", pid=1, state=1),
        title_fn=lambda: "第3讲 - 哔哩哔哩",
        clock=clock,
        on_start=lambda m: events.append(("start",)),
        on_stop=lambda m, why, dur: events.append(("stop", why)),
        is_quiet=lambda: loud["v"] < 0.005,
    )
    w.poll_once()
    clock.advance(1.2)
    w.poll_once()
    assert w.state == "recording"

    # 会话仍 Active 但页面静音了（比如被 muted）：电平掉下去 → 开始计静音时间
    loud["v"] = 0.0
    clock.advance(10.0)
    w.poll_once()
    assert w.state == "recording"
    clock.advance(15.0)
    w.poll_once()
    assert events[-1] == ("stop", "silence"), "会话赖着 Active 时，靠电平判断收尾"


def test_still_audible_keeps_recording_even_if_session_drops(rig):
    """会话短暂消失但声音还在（切 tab、整屏播放器重载）：不该切段。"""
    clock, events, build = rig
    loud = {"v": 0.4}
    plan = [True, True, False, False, False]
    w = wv.WebVideoWatcher(
        wv.WatcherConfig(start_after_s=1.0, stop_after_s=20.0),
        probe=FakeProbe(plan), title_fn=lambda: "第3讲 - 哔哩哔哩", clock=clock,
        on_start=lambda m: events.append(("start",)),
        on_stop=lambda m, why, dur: events.append(("stop", why)),
        is_quiet=lambda: loud["v"] < 0.005,
    )
    w.poll_once()
    clock.advance(1.2)
    w.poll_once()                     # recording
    clock.advance(30.0)
    w.poll_once()                     # 会话没了，但电平还在
    assert [e[0] for e in events] == ["start"], "还有声音就不该停"
    loud["v"] = 0.0
    clock.advance(30.0)
    w.poll_once()
    assert events[-1][0] == "stop"


# ---------------------------------------------------------------- 接到引擎上
class FakeRecorder:
    """替身录音机：不碰声卡，只记 state，让"自动录"这条链能端到端测。"""

    def __init__(self, out_path, device_name="", channels=2, **kw):
        self.out_path = out_path
        self.duration = 0.0
        self.error = ""
        self.on_event = None
        self.on_level = None
        self.last_data_at = 0.0
        self.started_at = 0.0
        FakeRecorder.last = self

    def start(self):
        import time as _t
        self.started_at = _t.time()
        self.last_data_at = 0.0

    def stop(self):
        self.duration = 42.0
        return self.out_path

    @property
    def receiving(self):
        return False

    @property
    def idle_seconds(self):
        return 0.0

    def drain_audio(self):
        import numpy as np
        return np.zeros(0, dtype="float32")


@pytest.fixture()
def eng(tmp_path, monkeypatch):
    from dsh_app import capture as cap
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    monkeypatch.setattr(cap, "LoopbackRecorder", FakeRecorder)
    db = DB(tmp_path / "t.db")
    st = Settings.load()
    st.llm.base_url = ""
    st.llm.api_key = ""
    e = Engine(db, st)
    return e, db


def test_start_web_capture_files_the_session_under_the_site(eng):
    e, db = eng
    meta = {"course": "哔哩哔哩·网课", "page_title": "第3讲 随机变量", "site": "哔哩哔哩"}
    sid = e.start_web_capture(meta)
    row = db.get_session(sid)
    assert row is not None
    assert row["title"] == "第3讲 随机变量"
    assert row["source"] == "webvideo", "自动录的会话要能认出来，别和手动录混在一起"
    course = db.query("SELECT title FROM courses WHERE id=?", (row["course_id"],))[0]
    assert "哔哩哔哩" in course["title"]


def test_stop_web_capture_logs_why_it_stopped(eng):
    e, db = eng
    sid = e.start_web_capture({"course": "学堂在线·网课", "page_title": "第2章 分布函数"})
    res = e.stop_web_capture(reason="silence")
    assert res["ok"] and res["session_id"] == sid and res["duration"] == 42.0
    evs = [dict(r) for r in db.get_events(sid)]
    kinds = [ev["kind"] for ev in evs]
    assert "web_capture_stop" in kinds, "要能回答'这段为什么结束'，否则用户以为是软件出错"
    payload = [ev for ev in evs if ev["kind"] == "web_capture_stop"][0]["payload_json"]
    assert "silence" in str(payload)


def test_second_start_while_recording_is_idempotent(eng):
    e, db = eng
    sid1 = e.start_web_capture({"course": "A·网课", "page_title": "第1讲"})
    sid2 = e.start_web_capture({"course": "B·网课", "page_title": "第2讲"})
    assert sid1 == sid2, "已经在录的时候不该另起一节"
    assert len(db.list_sessions()) == 1


# -- 自动抓取的两条回归：别把一节切成碎片、别拿别人的窗口当节名 -------------
def test_quiet_check_is_false_when_level_is_stale(tmp_path):
    """电平判不了（没在录 / 数据是陈的）时必须说"有声音"。

    真机上就是这样：`is_quiet` 恒为 True，一节 40 分钟的网课被切成
    20 秒一段的碎片，每段都去跑一次分析。
    """
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    eng = Engine(DB(tmp_path / "q.db"), Settings.load())
    assert eng.is_capture_quiet() is True            # 没在录：没有声音可录

    class FakeRec:
        level = 0.0
        level_at = 123.0                              # 很久以前的一块

    eng.recorder = FakeRec()
    # 收不到数据 = WASAPI 在没声音时不给回调 = 该停。写反过一次，结果是
    # 浏览器停播后会话永远不结束。
    assert eng.is_capture_quiet() is True

    FakeRec.level_at = __import__("time").time()       # 新鲜的静音：真静音
    assert eng.is_capture_quiet() is True

    FakeRec.level = 0.25                              # 新鲜的有声：不能停
    assert eng.is_capture_quiet() is False


def test_level_callback_reaches_the_engine(tmp_path, monkeypatch):
    """录音线程每块的电平必须能到 UI——电平条和停录判定都靠它。"""
    from dsh_app import capture as cap
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    seen = []

    class FakeRec:
        level = 0.0
        level_at = 0.0
        duration = 0.0
        on_event = None
        on_level = None

        def __init__(self, *a, **kw):
            pass

        def start(self):
            self.level, self.level_at = 0.25, __import__("time").time()
            if self.on_level:
                self.on_level(0.25)

        def stop(self):
            return None

    monkeypatch.setattr(cap, "LoopbackRecorder", FakeRec)
    eng = Engine(DB(tmp_path / "l.db"), Settings.load())
    eng.on_level = seen.append
    eng.start_recording("测试课", "第 1 讲")
    assert seen == [0.25]
    assert eng.capture_level()[0] == 0.25
    assert eng.is_capture_quiet() is False          # 0.25 显然不是静音


def test_watcher_titles_the_session_from_the_audio_process():
    """节名要取"出声那个进程"的窗口标题，而不是当前前台窗口。"""
    from dsh_app.webvideo import AudioSessionInfo, WatcherConfig, WebVideoWatcher

    started = []
    w = WebVideoWatcher(
        WatcherConfig(start_after_s=0.0, stop_after_s=10.0),
        probe=lambda: AudioSessionInfo(proc="msedge", pid=23024, state=1),
        title_fn=lambda: "某个不相干的文档 - 编辑器",
        pid_title_fn=lambda pid: "第一章 随机事件及其概率（二）_哔哩哔哩_bilibili",
        clock=lambda: 100.0,
        on_start=lambda m: started.append(m),
    )
    w.poll_once()
    w.poll_once()                      # 第一次只是 arm，第二次才真的起录
    assert started and "哔哩哔哩" in started[0].page_title
    assert started[0].site          # 站点识别出来了


def test_watcher_falls_back_to_foreground_when_pid_title_missing():
    from dsh_app.webvideo import AudioSessionInfo, WatcherConfig, WebVideoWatcher

    started = []
    w = WebVideoWatcher(
        WatcherConfig(start_after_s=0.0),
        probe=lambda: AudioSessionInfo(proc="chrome", pid=7, state=1),
        title_fn=lambda: "概率论 第 3 讲 - Google Chrome",
        pid_title_fn=lambda pid: "",
        clock=lambda: 5.0,
        on_start=lambda m: started.append(m),
    )
    w.poll_once()
    w.poll_once()
    assert started and "概率论" in started[0].page_title


def test_clean_title_strips_browser_and_profile_suffixes():
    """Windows 浏览器标题是"页面 - 站点 - 用户1 - Microsoft Edge"，得连着剥。"""
    from dsh_app.webvideo import clean_title

    raw = "第一章：随机事件及其概率（二）_哔哩哔哩_bilibili - 用户1 - Microsoft\u200b Edge"
    assert clean_title(raw) == "第一章：随机事件及其概率（二）_哔哩哔哩_bilibili"
    assert clean_title("概率论 第 3 讲 - Google Chrome") == "概率论 第 3 讲"
    assert clean_title("随便一个页面") == "随便一个页面"


def test_lesson_title_scoring_prefers_the_lesson_window():
    """同一进程开着好几个窗口时，选"像课"的那个，别选那个开着文档的。"""
    from dsh_app.webvideo import _score_lesson_title

    lesson = "第一章：随机事件及其概率（二）_哔哩哔哩_bilibili"
    doc = "根据这个方案构建应用，先看 — DeepSeek Harness 和另外 4 个页面"
    assert _score_lesson_title(lesson) > _score_lesson_title(doc)
    assert _score_lesson_title("") == 0
