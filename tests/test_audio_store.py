"""音频收口层：读、量、压缩。

为什么这套测试值得写：录音是 WAV，但课后会被压成 16k 单声道 FLAC。所有读音频的
老代码都写死了 `wave.open`，一旦转成 FLAC，"时长 0 秒""波形全零"这种错不会报错，
只会让回看点识别悄悄失效。所以这里逐个断言：换格式之后，读出来的东西必须一样。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsh_app import audio_io, capture


def _tone(path, seconds=6.0, rate=44100, freq=440.0, channels=2):
    import soundfile as sf

    t = np.arange(int(rate * seconds)) / float(rate)
    x = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([x] * channels, axis=1) if channels > 1 else x
    sf.write(str(path), data, rate, subtype="PCM_16")
    return path


@pytest.fixture()
def wav(tmp_path):
    return _tone(tmp_path / "lesson.wav")


def test_is_audio_knows_the_formats_we_meet(wav, tmp_path):
    assert audio_io.is_audio(wav)
    assert audio_io.is_audio(tmp_path / "a.FLAC")
    assert audio_io.is_audio(tmp_path / "导入的视频.m4a")
    assert not audio_io.is_audio(tmp_path / "notes.txt")


def test_duration_matches_the_source(wav):
    assert audio_io.duration(wav) == pytest.approx(6.0, abs=0.05)


def test_read_mono16k_downsamples_and_mixes_down(wav):
    mono = audio_io.read_mono16k(wav)
    assert mono.ndim == 1
    assert mono.shape[0] == pytest.approx(6 * 16000, rel=0.02)
    # 立体声两路相同，混成单声道后幅度不变
    assert 0.3 < float(np.abs(mono).max()) < 0.45


def test_read_mono_can_keep_the_native_rate(wav):
    raw = audio_io.read_mono(wav, rate=None)
    assert raw.shape[0] == pytest.approx(6 * 44100, rel=0.01)


def test_compress_shrinks_a_lot_and_keeps_the_timeline(wav, tmp_path):
    out = audio_io.compress_to_flac(wav, tmp_path / "lesson.flac")
    assert out.exists() and out.stat().st_size > 0
    # 16k 单声道 FLAC  gegenüber 44.1k 立体声 WAV：至少小 5 倍
    assert out.stat().st_size < wav.stat().st_size / 5
    assert audio_io.duration(out) == pytest.approx(6.0, abs=0.1)
    assert audio_io.info(out)["rate"] == 16000
    # 压缩是全损？不：波形峰值还在，能听、能重新分析
    mono = audio_io.read_mono16k(out)
    assert 0.3 < float(np.abs(mono).max()) < 0.45


def test_compress_streams_in_blocks_so_long_lessons_do_not_explode(tmp_path):
    """关键性质：分块读写。整段读进内存的话，90 分钟的课要 2 GB。

    这里用"块边界处没有静音或爆音"来间接证明接缝处理对了：
    分 1 秒一块压缩一段连续正弦，块边界若跳变，相邻样本差会出现尖峰。
    """
    src = _tone(tmp_path / "long.wav", seconds=4.0, freq=220.0)
    dst = audio_io.compress_to_flac(src, tmp_path / "long.flac", block_s=1.0)
    mono = audio_io.read_mono16k(dst)
    jumps = np.abs(np.diff(mono))
    assert float(jumps.max()) < 0.2       # 220 Hz 正弦相邻样本差本来就很小
    # 没有出现"某一块丢了/重复了"：长度对得上
    assert mono.shape[0] == pytest.approx(4 * 16000, rel=0.05)


def test_read_segment_returns_native_rate(wav):
    seg, rate = audio_io.read_segment(wav, 1.0, 2.5)
    assert rate == 44100                 # 不重采样：重新转写用的就是原样数据
    assert seg.shape[0] == pytest.approx(1.5 * rate, rel=0.02)


def test_missing_or_broken_file_never_raises(tmp_path):
    assert audio_io.duration(tmp_path / "nope.wav") == 0.0
    broken = tmp_path / "broken.flac"
    broken.write_bytes(b"not really flac")
    assert audio_io.duration(broken) == 0.0
    assert audio_io.info(broken)["rate"] == 0


def test_capture_helpers_now_accept_compressed_audio(wav, tmp_path):
    """老调用点（pipeline / smoke / 回溯检测）用的是 capture 里那几个函数名，
    它们必须跟着认 FLAC，否则压缩一次就把回溯检测废掉。"""
    dst = audio_io.compress_to_flac(wav, tmp_path / "c.flac")
    assert capture.audio_duration(dst) == pytest.approx(6.0, abs=0.1)
    arr = capture.read_wav_mono16k(dst)
    assert arr.shape[0] > 0 and float(np.abs(arr).max()) > 0.2


def test_detect_audio_events_reads_flac_too(tmp_path):
    """一段"有声—静音—有声"的音频，压缩后仍能测出静音段。"""
    import soundfile as sf

    rate = 16000
    loud = (0.4 * np.sin(2 * np.pi * 300 * np.arange(rate * 2) / rate)).astype(np.float32)
    silent = np.zeros(rate * 5, np.float32)
    mono = np.concatenate([loud, silent, loud])
    src = tmp_path / "gap.wav"
    sf.write(str(src), mono, rate, subtype="PCM_16")
    dst = audio_io.compress_to_flac(src, tmp_path / "gap.flac")
    assert audio_io.duration(dst) == pytest.approx(9.0, abs=0.15)
    events = capture.detect_audio_events(dst, min_silence_s=2.0, long_silence_s=4.0)
    assert events, "5 秒静音应该被认成暂停/回溯事件"

def test_analysis_compresses_the_lesson_audio(tmp_path, monkeypatch):
    """分析跑完要顺手把 WAV 压成 FLAC，并把库里的路径改过去。

    这是"音频占 0.89 GB"那条 P0 的落地点：压缩挂在 analyze_session 的收尾，
    而不是让用户记得手动点。
    """
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "a.db")
    settings = Settings.load()
    settings.capture.audio_compress = True
    settings.capture.save_raw_audio = True
    cid = db.ensure_course("测试课")
    wav = _tone(tmp_path / "lesson.wav", seconds=6.0)
    sid = db.create_session(cid, "第 1 讲", str(wav))
    db.replace_segments(sid, [{"idx": 0, "t_start": 0.0, "t_end": 6.0,
                               "text": "概率密度函数的定义", "confidence": 0.9, "source": "asr"}])
    Engine(db, settings).analyze_session(sid)

    row = db.get_session(sid)
    new_path = Path(str(row["audio_path"]))
    assert new_path.suffix == ".flac"
    assert new_path.exists() and new_path.stat().st_size < 96 * 1024   # 源 WAV 约 1 MB
    assert not wav.exists(), "压完要把原始 WAV 删掉，否则等于没省空间"
    # 压完还能读：重新转写和回溯检测都靠它
    assert audio_io.duration(new_path) == pytest.approx(6.0, abs=0.2)


def test_analysis_can_throw_the_audio_away(tmp_path):
    """不保留原始音频时，分析完直接删文件，但课还在。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine
    from pathlib import Path

    db = DB(tmp_path / "b.db")
    settings = Settings.load()
    settings.capture.save_raw_audio = False
    cid = db.ensure_course("测试课")
    wav = _tone(tmp_path / "lesson2.wav", seconds=4.0)
    sid = db.create_session(cid, "第 1 讲", str(wav))
    db.replace_segments(sid, [{"idx": 0, "t_start": 0.0, "t_end": 4.0,
                               "text": "分布函数右连续", "confidence": 0.9, "source": "asr"}])
    Engine(db, settings).analyze_session(sid)
    row = db.get_session(sid)
    assert row["audio_path"] == ""
    assert not wav.exists()
    assert len(db.get_segments(sid)) == 1, "删音频不能连带删转写"


def test_failed_compression_keeps_the_original(tmp_path, monkeypatch):
    """压缩失败只记错误，原始 WAV 必须原样留着——省空间不能以丢证据为代价。"""
    from dsh_app.config import Settings
    from dsh_app.db import DB
    from dsh_app.pipeline import Engine

    db = DB(tmp_path / "c.db")
    settings = Settings.load()
    cid = db.ensure_course("测试课")
    wav = _tone(tmp_path / "lesson3.wav", seconds=4.0)
    sid = db.create_session(cid, "第 1 讲", str(wav))
    db.replace_segments(sid, [{"idx": 0, "t_start": 0.0, "t_end": 4.0,
                               "text": "期望与方差", "confidence": 0.9, "source": "asr"}])

    def boom(*a, **kw):
        raise RuntimeError("磁盘满了")

    monkeypatch.setattr(audio_io, "compress_to_flac", boom)
    Engine(db, settings).analyze_session(sid)
    assert wav.exists()
    assert "音频处理失败" in (db.get_session(sid)["error"] or "")
