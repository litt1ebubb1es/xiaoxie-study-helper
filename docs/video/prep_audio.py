"""happy.mp3 -> work/audio.json（节拍网格 + 逐帧音频特征）

做三件事：
1. 用自相关估出 BPM，再用检测到的 onset 做线性回归，把 BPM 和相位钉准；
2. 输出节拍 / 小节线时间表（视频里的切点、脉冲全部锚在这张表上）；
3. 输出 60 fps 的 rms / low / high / onset 特征，供画面做呼吸和闪动。

视频截取窗口由 CLIP_START / CLIP_LEN 决定，所有时间都以"截取后"为准。
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WAV = HERE / "work" / "happy_mono.wav"
OUT = HERE / "work" / "audio.json"

FPS = 60
CLIP_START = 12.0          # 从第 12 秒开始（响度提上来之后）
CLIP_LEN = 99.0            # 99 秒 = 198 拍 = 49.5 小节，卡在 120 BPM 上


def load(path: Path) -> tuple[np.ndarray, int]:
    w = wave.open(str(path))
    sr, n = w.getframerate(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def onset_envelope(x: np.ndarray, sr: int, hop: int = 256, nfft: int = 1024) -> tuple[np.ndarray, float]:
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    flux = np.maximum(0.0, np.diff(spec, axis=0)).sum(1)
    flux /= flux.max() + 1e-9
    return flux, sr / hop


def band_energy(x: np.ndarray, sr: int, lo: float, hi: float, hop: int = 1024) -> np.ndarray:
    win = np.hanning(hop).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, hop)[:: hop // 2] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    band = spec[:, (freqs >= lo) & (freqs < hi)]
    return np.sqrt((band ** 2).mean(1))


def resample(t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    return np.interp(t, src_t, src_v)


def main() -> None:
    x, sr = load(WAV)
    flux, fps_env = onset_envelope(x, sr)
    t_env = np.arange(len(flux)) / fps_env

    # --- 粗估 BPM（70..190），再用 onset 峰做线性回归钉准 ---------------------
    env = flux - flux.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = int(fps_env * 60 / 190), int(fps_env * 60 / 70)
    lag = lo + int(np.argmax(ac[lo:hi]))
    bpm0 = 60 * fps_env / lag

    # 二维网格搜索：在 bpm0 附近同时找"最准的一拍长度"和"相位"，
    # 判据是网格点上 onset 包络的总和（越贴合鼓点越高）。比线性回归稳得多。
    def score(period: float, offset: float) -> float:
        grid = np.arange(offset, x.size / sr, period)
        idx = np.clip((grid * fps_env).astype(int), 0, len(flux) - 1)
        return float(flux[idx].sum())

    best = (bpm0, 0.0, -1.0)
    for cand_bpm in np.arange(bpm0 * 0.97, bpm0 * 1.03, 0.01):
        per = 60 / cand_bpm
        for off in np.arange(0, per, per / 60):
            sc = score(per, off)
            if sc > best[2]:
                best = (float(cand_bpm), float(off), sc)
    bpm, offset, _ = best
    period = 60 / bpm

    beats = np.arange(offset, x.size / sr, period)
    downbeats = beats[::4]

    # --- 逐帧特征（截取窗口内） -------------------------------------------
    n_frames = int(round(CLIP_LEN * FPS))
    tf = CLIP_START + np.arange(n_frames) / FPS

    rms_t = np.arange(len(band_energy(x, sr, 20, 20000))) * (1024 / 2) / sr
    rms = band_energy(x, sr, 20, 20000)
    low = band_energy(x, sr, 20, 180)
    high = band_energy(x, sr, 4000, 11000)

    def norm(v: np.ndarray) -> np.ndarray:
        p95 = np.percentile(v, 95) + 1e-9
        return np.clip(v / p95, 0, 1.4)

    feats = {
        "rms": resample(tf, rms_t, norm(rms)),
        "low": resample(tf, rms_t, norm(low)),
        "high": resample(tf, rms_t, norm(high)),
        "onset": resample(tf, t_env, np.clip(flux / (np.percentile(flux, 98) + 1e-9), 0, 1.5)),
    }

    data = {
        "fps": FPS,
        "clip_start": CLIP_START,
        "clip_len": CLIP_LEN,
        "frames": n_frames,
        "song_duration": x.size / sr,
        "bpm": float(bpm),
        "period": float(period),
        "offset": float(offset),
        "beats": [float(b - CLIP_START) for b in beats if CLIP_START - 1 <= b <= CLIP_START + CLIP_LEN + 1],
        "downbeats": [float(b - CLIP_START) for b in downbeats if CLIP_START - 1 <= b <= CLIP_START + CLIP_LEN + 1],
        "features": {k: [round(float(v), 4) for v in arr] for k, arr in feats.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    print(f"时长 {data['song_duration']:.2f}s ｜ BPM {bpm:.3f} ｜ 每拍 {period:.4f}s ｜ 首拍相位 {offset:.3f}s")
    print(f"窗口 {CLIP_START:.1f}–{CLIP_START + CLIP_LEN:.1f}s ｜ {n_frames} 帧 @ {FPS}fps")
    print(f"窗口内拍点 {len([b for b in data['beats'] if 0 <= b <= CLIP_LEN])} 个，"
          f"小节线 {len([b for b in data['downbeats'] if 0 <= b <= CLIP_LEN])} 个")
    print(f"写出 {OUT}")


def _peaks(v: np.ndarray, height: float, distance: int):
    """极简峰值检测：先卡阈值，再按最小间距贪心取大。"""
    cand = np.where(v >= height)[0]
    if cand.size == 0:
        return cand, {}
    order = cand[np.argsort(-v[cand])]
    keep: list[int] = []
    for i in order:
        if all(abs(int(i) - k) >= distance for k in keep):
            keep.append(int(i))
    keep.sort()
    return np.array(keep), {}


if __name__ == "__main__":
    main()