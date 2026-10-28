"""主界面骨架的回归测试：左栏导航 ⇄ 页签 ⇄ 页头。

以前是"顶部一条工具栏 + 一排页签"，按钮挤在一起、两套导航抢注意力。现在导航只剩
左栏一处，页签栏藏起来。这种"结构不靠肉眼"的改动必须有测试盯着，否则某天有人把
tabBar().show() 加回来，界面又变成两套导航。
"""
from __future__ import annotations

import pytest

from dsh_app.ui import icons, theme
from dsh_app.ui.main_window import PAGES


@pytest.fixture()
def win(qapp, tmp_path):
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.ui.main_window import MainWindow

    # 关键：设置也落在 tmp_path。真实 config.json 绝不能因为跑测试被改
    # （有个测试会点「按记录调阈值」，它内部就调 settings.save()）
    settings = Settings.load(tmp_path / "config.json")
    db = DB(tmp_path / "ui.db")
    w = MainWindow(settings, db)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def test_nav_icons_all_exist(qapp):
    for _title, _sub, ic in PAGES:
        assert icons.get(ic) is not None
    assert "timeline" in icons.names() and "settings" in icons.names()


def test_icon_renders_at_two_sizes(qapp):
    pm = icons.get("play", theme.ACCENT, 18).pixmap(18, 18)
    assert not pm.isNull() and pm.width() == 18


def test_icons_need_an_app_first():
    """没有 QApplication 时给个能看懂的错，而不是让 Qt 直接 abort。"""
    import dsh_app.ui.icons as ic

    ic._cached.cache_clear()
    if ic.QGuiApplication.instance() is None:
        with pytest.raises(RuntimeError):
            ic.get("play")
    else:                     # 同进程里已经有 app（别的用例建过），这条就跳过
        assert ic.get("play") is not None


def test_app_badge_is_drawn(qapp):
    badge = icons.app_badge(30)
    assert not badge.isNull() and badge.width() == 30


def test_nav_and_tabs_stay_in_sync(win):
    assert win.nav.count() == win.tabs.count() == len(PAGES) == 6
    for row in range(win.tabs.count()):
        win.nav.setCurrentRow(row)
        assert win.tabs.currentIndex() == row
        assert win.lbl_page_title.text() == PAGES[row][0]
        assert win.windowTitle().endswith(PAGES[row][0])


def test_switching_tab_in_code_moves_nav(win):
    win.tabs.setCurrentIndex(3)
    assert win.nav.currentRow() == 3
    assert win.lbl_page_sub.text() == PAGES[3][1]


def test_tab_bar_is_hidden(win):
    """页签栏必须藏着：导航只在左栏，顶部再来一排就是两套导航。"""
    assert win.tabs.tabBar().isVisible() is False


def test_top_actions_are_few(win):
    """页头的按钮数量要少：这是"别都挤在上面"这条要求的可测形式。"""
    visible = [w for w in win.findChildren(type(win.btn_more))
               if w.isVisibleTo(win) and w.parent() is not None]
    assert sum(1 for w in visible if w.objectName() != "primary") < 40   # 粗筛，防止失控加按钮
    assert win.btn_chat_pop.isVisibleTo(win)
    assert win.btn_more.isVisibleTo(win)


def test_more_menu_routes_to_the_old_buttons(win):
    """菜单项就是老按钮：这样所有已有逻辑不用改一行。"""
    acts = [a for a in win.menu_more.actions() if not a.isSeparator()]
    assert len(acts) == 7
    assert acts[0].text().startswith("从 B 站导入课程")
    # 断开真实处理函数，否则触发"导出笔记"会弹保存对话框，测试就卡在那儿了
    win.btn_export.clicked.disconnect()
    hit = []
    win.btn_export.clicked.connect(lambda: hit.append(1))
    next(a for a in acts if a.text() == "导出笔记").trigger()
    assert hit == [1]


def test_record_controls_live_in_sidebar(win):
    """录制三件套属于"正在上课"，应该跟着课程列表待着，不在页头。"""
    assert win.btn_record.isVisibleTo(win)
    assert win.btn_stop.isVisibleTo(win)
    assert win.btn_mark.isVisibleTo(win)
    assert win.chk_autoweb.isVisibleTo(win)
    assert win.level.isVisibleTo(win)


def test_nav_icon_of_current_page_is_accent(win):
    """当前页图标染橙、其余灰——QSS 管不到图标颜色，只能代码换。"""
    win.nav.setCurrentRow(2)
    icon = win.nav.item(2).icon()
    other = win.nav.item(0).icon()
    assert not icon.isNull() and not other.isNull()
    px = icon.pixmap(18, 18).toImage()
    px2 = other.pixmap(18, 18).toImage()
    assert px != px2          # 两张图必须不一样，否则说明没染色

# -- 设置页：存储 / 灵敏度 --------------------------------------------------
def test_storage_controls_are_collected_into_settings(win):
    """存储开关必须真的写进配置——写了但存不下来，等于没有。"""
    win.sp_keep_days.setValue(3)
    win.chk_compress.setChecked(False)
    win.chk_keep_raw.setChecked(True)
    win._collect_settings()
    assert win.settings.capture.keep_audio_days == 3
    assert win.settings.capture.audio_compress is False
    assert win.settings.capture.save_raw_audio is True


def test_sensitivity_preset_overwrites_the_two_thresholds(win):
    """三档灵敏度只动两个阈值：判定分和象限分位。"""
    win.cmb_sens.setCurrentIndex(0)          # 保守 0.55 / 75
    assert round(win.sp_grade.value(), 2) == 0.55
    assert round(win.sp_pct.value()) == 75
    win.cmb_sens.setCurrentIndex(2)          # 积极 0.35 / 65
    assert round(win.sp_grade.value(), 2) == 0.35
    assert round(win.sp_pct.value()) == 65
    win._collect_settings()
    assert round(win.settings.cognitive.grade_medium, 2) == 0.35
    assert round(win.settings.cognitive.quadrant_percentile) == 65


def test_knot_action_bar_exists_below_the_table(win):
    """复制时间点/确认/不算/撤销这四个动作要能点，默认没有数据时也可点（点了给提示）。"""
    for b in (win.btn_copy_ts, win.btn_confirm_knot,
              win.btn_dismiss_knot, win.btn_undo_knot):
        assert b.isVisibleTo(win)
        assert b.isEnabled()


def test_zoom_chip_starts_as_full_view(win):
    # 空窗口的时间轴没有时长，缩放是空操作；先给它一节 10 分钟的课
    win.timeline.set_data([], [], 600.0)
    assert win.lbl_zoom_state.text() == "全貌"
    win.timeline.zoom_at(4.0, 30.0)
    win._on_view_changed(*win.timeline.view_range())
    assert "–" in win.lbl_zoom_state.text()      # 变成"起点–终点（多长）"


def test_settings_page_scrolls(win):
    """设置页越来越长，必须包在滚动区里；否则小屏幕上最后一块永远点不到。"""
    from PySide6.QtWidgets import QScrollArea

    assert isinstance(win._settings_scroll, QScrollArea)
    assert win._settings_scroll.widget() is not None


# -- 回看点纠错：反馈之后行还选着 --------------------------------------------
def _seed_one_knot(db, t0=60.0, t1=90.0):
    """建一节带一条回看点的会话，返回 (session_id, knot_id)。"""
    cid = db.ensure_course("测试课")
    sid = db.create_session(cid, "第 1 讲", "")
    db.replace_segments(sid, [{"idx": 0, "t_start": t0, "t_end": t1,
                               "text": "全概率公式", "confidence": 0.9, "source": "asr"}])

    class K:
        def __init__(self):
            self.t_start, self.t_end = t0, t1
            self.level, self.quadrant, self.grade = "explicit", "high_low", "strong"
            self.score, self.l_score, self.b_score = 0.8, 0.8, 0.7
            self.evidence, self.excerpt = "你回退了约 30 秒", "全概率公式"
            self.evidence_detail = {"why": "回溯"}

    ids = db.replace_knots(sid, [K()])
    return sid, int(ids[0])


def test_undo_after_dismiss_keeps_the_row_selected(win):
    """标了「不算回看点」之后要能立刻撤销：行得还选着。

    真机上试出来的：重画表格会把选中清掉，撤销按钮拿到 None，只弹一句
    "先选一个回看点"——用户以为按钮坏了。
    """
    w = win
    db = w.db
    sid, kid = _seed_one_knot(db)
    w.current_session = sid
    w._load_session_view()
    assert w.knot_table.rowCount() == 1

    w._select_knot_row(kid)
    assert w._current_knot_row() is not None

    w._on_dismiss_knot()
    assert db.get_knot(kid)["status"] == "dismissed"
    assert w._current_knot_row() is not None, "重画之后应还选着这一行"
    assert int(w._current_knot_row().id) == kid

    w._on_restore_knot()
    assert db.get_knot(kid)["status"] == "pending"


def test_copy_timestamp_puts_the_moment_on_the_clipboard(win):
    """没有画面就别装回放：按钮只把时刻抄给你，回视频里自己拖。"""
    from PySide6.QtWidgets import QApplication

    w = win
    sid, kid = _seed_one_knot(w.db, t0=75.0, t1=95.0)
    w.current_session = sid
    w._load_session_view()
    w._select_knot_row(kid)
    assert "01:15" in w.lbl_ts.text()            # 提示里就该写着这一刻

    w._on_copy_timestamp()
    assert QApplication.clipboard().text() == "01:15"
    assert "01:15" in w.lbl_ts.text()


def test_copy_timestamp_without_selection_only_hints(win):
    w = win
    w._on_copy_timestamp()                        # 没有数据也不该炸
    assert "先选一个回看点" in w.status.currentMessage()


def test_double_click_row_does_not_try_to_play_anything(win):
    """回放已经拿掉了：双击只进重讲，不能偷偷去开音频设备。"""
    w = win
    sid, _kid = _seed_one_knot(w.db)
    w.current_session = sid
    w._load_session_view()
    assert not hasattr(w, "player")
    assert not hasattr(w, "btn_play_knot")


# -- 自动抓取的垃圾与失败：不许留垃圾节，不许弹框 ---------------------------
def _fake_web_capture(db, dur: float = 6.9) -> str:
    cid = db.ensure_course("哔哩哔哩·网课")
    sid = db.create_session(cid, "某节课", "")
    db.set_session_meta(sid, source="webvideo", duration=dur) if hasattr(
        db, "set_session_meta") else db.execute(
        "UPDATE sessions SET source='webvideo', duration=? WHERE id=?", (dur, sid))
    return sid


def test_short_auto_capture_is_discarded_entirely(win, tmp_path):
    """只响了几秒的自动抓取（提示音/广告前奏）不该留成一节失败的课。"""
    w = win
    wav = tmp_path / "blip.wav"
    wav.write_bytes(b"RIFF")
    sid = _fake_web_capture(w.db)
    w.db.execute("UPDATE sessions SET audio_path=? WHERE id=?", (str(wav), sid))

    w.current_session = sid
    w._discard_short_capture(sid, 6.9)

    assert w.db.get_session(sid) is None, "垃圾节应该整个删掉"
    assert not wav.exists(), "音频也别留着"
    assert "当提示音忽略" in w.status.currentMessage()


def test_auto_capture_failure_never_opens_a_modal(win):
    """自动抓取的分析失败只在状态栏说一声：用户正在看视频，别盖模态框。"""
    from PySide6.QtWidgets import QMessageBox

    w = win
    called = []
    orig = QMessageBox.critical
    QMessageBox.critical = staticmethod(lambda *a, **k: called.append(a) or QMessageBox.Ok)
    try:
        w._pipeline_failed("所有转写后端都失败：…", sid="", auto=True)
    finally:
        QMessageBox.critical = orig

    assert called == []
    assert "自动抓取" in w.status.currentMessage()

    called.clear()
    QMessageBox.critical = staticmethod(lambda *a, **k: called.append(a) or QMessageBox.Ok)
    try:
        w._pipeline_failed("手动点的就照实报", sid="", auto=False)
    finally:
        QMessageBox.critical = orig
    assert len(called) == 1, "手动触发失败仍然要报出来"


def test_copy_with_evidence_makes_a_note_ready_line(win):
    """贴到笔记里的那种复制：时间点 + 等级 + 依据 + 原话。"""
    from PySide6.QtWidgets import QApplication

    w = win
    sid, kid = _seed_one_knot(w.db, t0=45.0, t1=75.0)
    w.current_session = sid
    w._load_session_view()
    w._select_knot_row(kid)

    w._on_copy_timestamp(with_evidence=True)
    text = QApplication.clipboard().text()
    assert text.startswith("00:45")
    assert "重点回看" in text and "0.80" in text
    assert "回退了" in text                    # 依据
    assert "原话" in text


def test_copy_hint_names_where_to_watch(win):
    """自动抓的网页视频要说出"回哪儿的播放器"，手动录的只说"视频"。"""
    w = win
    sid, kid = _seed_one_knot(w.db)
    w.db.execute("UPDATE sessions SET source='webvideo' WHERE id=?", (sid,))
    cid = w.db.get_session(sid)["course_id"]
    w.db.execute("UPDATE courses SET title='哔哩哔哩·网课' WHERE id=?", (cid,))
    w.current_session = sid
    w._load_session_view()
    w._select_knot_row(kid)
    assert "播放器" in w.lbl_ts.text()


def test_short_capture_threshold_comes_from_settings(win):
    """8 秒是默认值，但要能在 config.json 里调（网课平台片头差异很大）。"""
    w = win
    w.settings.capture.web_min_keep_s = 3.0
    assert float(w.settings.capture.web_min_keep_s) == 3.0


# ---------------------------------------------------------------- B 站导入接线
def _seed_bili_session(w) -> str:
    """一节"已导入、带原始链接、还没录"的课。UI 上的新入口都围着它转。"""
    db = w.db
    cid = db.ensure_course("概率论")
    db.create_session(cid, "第 2 讲", audio_path="a.flac", source="bili")
    sid = db.list_sessions()[0]["id"]
    db.update_session(sid, duration=180.0,
                      video_url="https://www.bilibili.com/video/BV1xx411c7mD?p=1")
    return sid


def test_session_menu_opens_the_video_in_browser(win, monkeypatch):
    from dsh_app import bili

    sid = _seed_bili_session(win)
    opened = []
    monkeypatch.setattr(bili, "open_in_browser", lambda url: opened.append(url) or True)
    win._open_session_video(sid)
    assert opened and opened[0].startswith("https://www.bilibili.com/video/BV1")


def test_open_session_video_appends_the_timestamp(win, monkeypatch):
    from dsh_app import bili

    sid = _seed_bili_session(win)
    opened = []
    monkeypatch.setattr(bili, "open_in_browser", lambda url: opened.append(url) or True)
    win._open_session_video(sid, 75.4)
    assert opened[0].endswith("t=75")


def test_copy_timestamp_prefers_the_deep_link(win):
    from types import SimpleNamespace

    sid = _seed_bili_session(win)
    win.db.replace_knots(sid, [SimpleNamespace(
        t_start=45.0, t_end=70.0, level="strong", quadrant="explicit",
        grade="重点回看", score=0.71, l_score=0.8, b_score=0.6,
        evidence="术语密度高（16 个新术语）；你回退了约 36 秒",
        evidence_detail={}, excerpt="这一步我没跟上")])
    win.current_session = sid
    win._load_session_view()
    kid = win.db.get_knots(sid)[0].id
    win._select_knot_row(int(kid))
    win._on_copy_timestamp()
    from PySide6.QtWidgets import QApplication

    clip = QApplication.clipboard().text()
    assert "bilibili.com" in clip and clip.endswith("t=45")


def test_imported_session_hints_where_to_start(win):
    res = win.engine.import_series("B站 · 概率论", [
        {"title": "第1讲", "url": "https://www.bilibili.com/video/BV9?p=1",
         "duration": 600}])
    win._refresh_sessions()
    sid = win.db.list_sessions()[0]["id"]
    win.current_session = sid
    win._load_session_view()
    assert "看这一节" in win.status.currentMessage()
    assert res["created"] == 1


def test_discard_short_capture_leaves_a_sample(win):
    win.current_session = ""
    sid = win.db.create_session(win.db.ensure_course("哔哩哔哩·网课"), "提示音",
                                source="webvideo")
    win._web_last_proc = "msedge"
    win._discard_short_capture(sid, 6.9)
    st = win.db.discard_stats(days=30)
    assert st["n"] == 1 and st["median_s"] == 6.9
    assert win.db.get_session(sid) is None


def test_discard_hint_moves_the_threshold_above_what_was_thrown_away(win):
    win.db.add_capture_discard("msedge", "哔哩哔哩·网课", "提示音", 12.0, 8.0, "too_short")
    win._on_apply_discard_hint()
    assert float(win.settings.capture.web_min_keep_s) >= 12.0


def test_settings_show_how_many_captures_were_dropped(win):
    win.db.add_capture_discard("msedge", "哔哩哔哩·网课", "广告", 5.0, 8.0, "too_short")
    win._refresh_discard_info()
    assert "1 次" in win.lbl_discard.text()


def test_archive_all_button_feeds_the_knowledge_base(win):
    sid = _seed_bili_session(win)
    win.db.replace_segments(sid, [
        {"t_start": 0.0, "t_end": 30.0, "text": "先讲样本空间的划分。"},
        {"t_start": 30.0, "t_end": 60.0, "text": "再讲全概率公式。"}])
    out = win.engine.kb_ingest_all()
    assert out["done"] == 1
    docs = [d for d in win.engine.kb_docs(None) if str(d["path"]).startswith("session:")]
    assert len(docs) == 1
    win._refresh_kb_auto_info()
    assert "已归档 1 节" in win.lbl_kb_auto.text()
