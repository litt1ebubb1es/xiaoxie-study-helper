"""本地转写的安装与自检（对应设置页「本地转写 Whisper」一栏）。

三件事分开做，每件都能单独重试，也都如实报告结果：
1. 依赖：`faster-whisper` 有没有装（没装就告诉你确切命令，不替你静默安装）
2. 模型：某个尺寸的权重在不在本机（走 hf-mirror；模型落 data/models/）
3. 能跑：用一段真实音频跑一遍，报用时与文字

为什么不用云端转写：实测这台机器配的 MaaS 端点没有 /audio/transcriptions（404），
而本地转写离线、免费、不限量，也符合"端侧"的定位。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"
HF_MIRROR = "https://hf-mirror.com"

# 体积是各尺寸权重文件的实际量级，用来让用户心里有数
SIZES = [
    {"key": "tiny",     "label": "tiny（75 MB）最快，准确率低，只适合试流程"},
    {"key": "base",     "label": "base（145 MB）快，中文一般"},
    {"key": "small",    "label": "small（484 MB）中文够用，推荐默认"},
    {"key": "medium",   "label": "medium（1.5 GB）更准，GPU 友好"},
    {"key": "large-v3", "label": "large-v3（3.1 GB）最准，需显存/较慢"},
]
DEFAULT_SIZE = "small"

# 实测坑：ctranslate2 构造 CUDA 模型时不报错，**第一次推理**才抛
# "Library cublas64_12.dll is not found"（缺 cuBLAS/cuDNN）。所以回落逻辑
# 必须覆盖"运行期"，并在本进程内记下来，避免每次都白试一次 GPU。
_GPU_BROKEN = False

# 钉住简体中文的书面语风格（见 transcribe_retry 里的说明）
SIMPLIFIED_HINT = "以下是普通话课程讲解，请用简体中文书面语转写。"


def _hf_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", HF_MIRROR)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def model_dir(size: str) -> Path:
    return MODELS_DIR / f"faster-whisper-{size}"


def _dir_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 1)


def deps_status() -> tuple[bool, str]:
    """依赖是否可用。返回 (可用, 说明)。"""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False, "未安装 faster-whisper"
    try:
        import ctranslate2
        ver = getattr(faster_whisper, "__version__", "?")
        cuda = 0
        try:
            cuda = int(ctranslate2.get_cuda_device_count())
        except Exception:  # noqa: BLE001
            cuda = 0
        return True, f"faster-whisper {ver}｜ctranslate2 {ctranslate2.__version__}｜CUDA 设备 {cuda}"
    except Exception as e:  # noqa: BLE001
        return True, f"faster-whisper 已装（{type(e).__name__}）"


def models_status() -> list[dict]:
    out = []
    for item in SIZES:
        d = model_dir(item["key"])
        installed = d.exists() and any(d.glob("*.bin"))
        out.append({"key": item["key"], "label": item["label"],
                    "installed": bool(installed), "mb": _dir_mb(d),
                    "path": str(d)})
    return out


def status() -> dict:
    ok, msg = deps_status()
    models = models_status()
    have = [m for m in models if m["installed"]]
    gpu_note = ""
    if ok:
        gpu_note = "｜GPU 可用" if gpu_usable() else "｜CPU 推理"
    return {"deps_ok": ok, "deps": msg + gpu_note, "models": models,
            "gpu_usable": gpu_usable(),
            "installed_sizes": [m["key"] for m in have],
            "any_model": bool(have),
            "hf_endpoint": os.environ.get("HF_ENDPOINT", HF_MIRROR),
            "models_dir": str(MODELS_DIR),
            "ready": bool(ok and have)}


def install_deps(timeout_s: int = 900) -> tuple[bool, str]:
    """装 faster-whisper。返回 (成功, 日志尾)。不猜、不静默：失败原因原样返回。"""
    cmd = [sys.executable, "-m", "pip", "install", "-q", "faster-whisper",
           "-i", PIP_MIRROR]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"安装超时（>{timeout_s}s），可重试"
    log = ((r.stdout or "") + (r.stderr or "")).strip()
    ok, msg = deps_status()
    return ok, (log[-800:] or msg)


def repo_for(size: str) -> str:
    return f"Systran/faster-whisper-{size}"


# 这些文件对推理没用，不下（省时省流量）
_SKIP = (".gitattributes", "README.md", ".md", ".png", ".jpg", ".gif")

# 下载源。实测（同一台机器、同一个 model.bin）：
#   ModelScope 单连接 4.7 MB/s ｜ hf-mirror 单连接 1.6 MB/s
#   两个源同时开 12 线程合计 7.1、ModelScope 自己 12 线程 7.4 → 本地出口约 8 MB/s 封顶
# 所以：ModelScope 优先，hf-mirror 兜底；大文件再切块并行。
SOURCES = ("modelscope", "hf-mirror")

_UA = {"User-Agent": "Mozilla/5.0"}


def _url_for(source: str, repo: str, name: str) -> str:
    if source == "modelscope":
        return f"https://www.modelscope.cn/models/{repo}/resolve/master/{name}"
    return f"{HF_MIRROR}/{repo}/resolve/main/{name}?download=true"


def _list_files(repo: str) -> list[str]:
    """列出要下的文件。先问 ModelScope，问不到再问 hf-mirror。"""
    import httpx

    try:
        r = httpx.get(
            f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files",
            params={"Revision": "master", "Recursive": "true"},
            headers=_UA, timeout=30)
        if r.status_code == 200:
            files = [(f.get("Path") or "") for f in
                     ((r.json().get("Data") or {}).get("Files") or [])]
            files = [f for f in files if f and not f.endswith(_SKIP)]
            if files:
                return files
    except Exception:  # noqa: BLE001
        pass
    r = httpx.get(f"{HF_MIRROR}/api/models/{repo}", timeout=30, headers=_UA)
    r.raise_for_status()
    files = [s.get("rfilename", "") for s in (r.json().get("siblings") or [])]
    return [f for f in files if f and not f.endswith(_SKIP)]


def _remote_size(url: str) -> int:
    """用 Range: bytes=0-0 问总大小（顺便验证这个源真的可下）。"""
    import httpx

    r = httpx.get(url, headers={**_UA, "Range": "bytes=0-0"},
                  timeout=httpx.Timeout(20.0, read=30.0), follow_redirects=True)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code}")
    cr = r.headers.get("content-range") or ""
    if "/" in cr:
        return int(cr.split("/")[-1])
    return int(r.headers.get("content-length") or 0)


def _fmt_speed(bps: float) -> str:
    return f"{bps / 1048576:.1f} MB/s" if bps > 0 else "-"


def download_file(urls: list[str], dest: Path, progress=None,
                  threads: int = 12, chunk_mb: int = 8) -> None:
    """多线程分块下载一个文件，支持断点续传；失败抛异常。

    每个块落到 `<dest>.chunks/<i>`：中断后再跑会跳过已完整的块，
    也省掉"整文件重下"。合并时顺序拼回，最后校验总大小。
    """
    import threading

    import httpx

    if not urls:
        raise RuntimeError("没有可用的下载地址")
    total = 0
    errs = []
    for u in urls:
        try:
            total = _remote_size(u)
            if total > 0:
                break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{u[:40]}… {type(e).__name__}: {str(e)[:70]}")
    if total <= 0:
        raise RuntimeError("所有源都拿不到文件大小：" + " ｜ ".join(errs))

    chunk = max(1, chunk_mb) * 1024 * 1024
    # spans 是 (字节偏移, 长度)：偏移本身就是块号，续传时块名不会因为改 chunk 而错位
    spans = [(off, min(off + chunk, total) - off)
             for off in range(0, total, chunk)]
    cdir = dest.with_suffix(dest.suffix + ".chunks")
    cdir.mkdir(parents=True, exist_ok=True)

    done_lock = threading.Lock()
    done_bytes = [0]
    t0 = time.time()
    last_emit = [0.0]
    remain = [len(spans)]

    def emit(force: bool = False) -> None:
        if not progress:
            return
        now = time.time()
        with done_lock:
            done = done_bytes[0]
        if not force and now - last_emit[0] < 0.6:
            return
        last_emit[0] = now
        el = max(0.001, now - t0)
        speed = done / el
        eta = (total - done) / speed if speed > 0 else 0
        pct = done * 100 // total
        progress(f"{dest.name} {pct}% ｜ {done / 1048576:.0f}/{total / 1048576:.0f} MB"
                 f" ｜ {_fmt_speed(speed)} ｜ 剩 {eta:.0f}s")

    def fetch(offset: int, length: int) -> None:
        cpath = cdir / f"{offset:012d}"
        if cpath.exists() and cpath.stat().st_size == length:
            with done_lock:
                done_bytes[0] += length
                remain[0] -= 1
            emit()
            return
        want = (f"bytes={offset}-{offset + length - 1}")
        last_err = ""
        for u in urls:                       # 换源重试
            for attempt in range(3):
                got = 0
                try:
                    with httpx.stream("GET", u, headers={**_UA, "Range": want},
                                      timeout=httpx.Timeout(20.0, read=120.0),
                                      follow_redirects=True) as r:
                        if r.status_code not in (200, 206):
                            raise RuntimeError(f"HTTP {r.status_code}")
                        with open(cpath, "wb") as f:
                            for b in r.iter_bytes(262144):
                                f.write(b)
                                got += len(b)
                                with done_lock:
                                    done_bytes[0] += len(b)
                                emit()
                    if got == length:
                        with done_lock:
                            remain[0] -= 1
                        return
                    raise RuntimeError(f"只收到 {got}/{length} 字节")
                except Exception as e:  # noqa: BLE001
                    last_err = f"{u[:32]}… {type(e).__name__}: {str(e)[:60]}"
                    with done_lock:
                        done_bytes[0] -= got
                    cpath.unlink(missing_ok=True)
                    time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"偏移 {offset} 的块下载失败：{last_err}")

    try:
        with ThreadPoolExecutor(max_workers=min(threads, len(spans))) as ex:
            futs = [ex.submit(fetch, off, ln) for off, ln in spans]
            for f in as_completed(futs):
                f.result()                  # 有块失败就整体失败，别假装成功
    except Exception:
        emit(force=True)
        raise

    # 合并
    tmp = dest.with_suffix(dest.suffix + ".joining")
    with open(tmp, "wb") as out:
        for off, _ln in spans:
            out.write((cdir / f"{off:012d}").read_bytes())
    size = tmp.stat().st_size
    if size != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"大小不对：{size} != {total}")
    tmp.replace(dest)
    shutil.rmtree(cdir, ignore_errors=True)
    emit(force=True)


def download_model(size: str = DEFAULT_SIZE,
                   progress=None) -> tuple[bool, str]:
    """把某个尺寸的权重下到 data/models/。

    不用 huggingface_hub 的 snapshot_download：实测它对 hf-mirror 走的 CAS 重建接口
    返回 401，而**逐文件直连 GET 是通的**。所以自己拉：
    ModelScope 优先（实测快 3 倍）→ hf-mirror 兜底，大文件分块并行 + 断点续传。
    """
    target = model_dir(size)
    if _stamp_ok(target):
        return True, f"{size} 已在本机且校验通过（{_dir_mb(target):.0f} MB），跳过下载"
    repo = repo_for(size)
    try:
        files = _list_files(repo)
    except Exception as e:  # noqa: BLE001
        return False, f"拿不到文件清单：{type(e).__name__}: {str(e)[:160]}"
    if not any(f.endswith(".bin") for f in files):
        return False, f"清单里没有 .bin：{files[:5]}"

    target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fetched = 0          # 真下了多少字节（只做校验时是 0，别报一个假速度）
    for name in files:
        dest = target / name
        if dest.exists() and dest.stat().st_size > 0:
            # 已有文件也要对大小：下坏一半的文件会被当成"已装"，
            # 到加载模型时才报莫名其妙的错，很难查。
            try:
                want = _remote_size(_url_for(SOURCES[0], repo, name))
            except Exception:  # noqa: BLE001
                want = 0
            if want <= 0 or dest.stat().st_size == want:
                continue
            if progress:
                progress(f"{name} 大小不对（{dest.stat().st_size} ≠ {want}），重下")
            dest.unlink(missing_ok=True)
        urls = [_url_for(s, repo, name) for s in SOURCES]
        try:
            if name.endswith(".bin"):        # 大文件才值得分块
                download_file(urls, dest, progress=progress, threads=8)
            else:
                _download_small(urls, dest, progress)
            fetched += dest.stat().st_size
        except Exception as e:  # noqa: BLE001
            return False, f"下载 {name} 失败：{type(e).__name__}: {str(e)[:200]}"
    _write_stamp(target)
    el = time.time() - t0
    mb = _dir_mb(target)
    if fetched <= 0:
        return True, f"{size} 已在本机且校验通过（{mb:.0f} MB），不用重下"
    return True, (f"{size} 下载完成 {mb:.0f} MB，用时 {el:.0f}s"
                  f"（{fetched / 1048576 / max(el, 0.1):.1f} MB/s）")


STAMP = ".dsh-ok.json"


def _stamp_ok(target: Path) -> bool:
    """目录里有没有"下完并校验过"的标记，且文件大小仍然对得上。

    只看"有没有 *.bin"是不够的：中途断掉的半截文件同样满足那个条件，
    之后加载模型才炸，报错还看不懂。
    """
    import json

    stamp = target / STAMP
    if not stamp.is_file():
        return False
    try:
        want = json.loads(stamp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    for name, size in want.items():
        f = target / name
        if not f.is_file() or f.stat().st_size != size:
            return False
    return bool(want)


def _write_stamp(target: Path) -> None:
    import json

    sizes = {p.name: p.stat().st_size for p in target.iterdir()
             if p.is_file() and p.name != STAMP}
    (target / STAMP).write_text(json.dumps(sizes, ensure_ascii=False, indent=1),
                                encoding="utf-8")


def _download_small(urls: list[str], dest: Path, progress=None) -> None:
    """小文件（config.json 之类）单流拿，成败都直说。"""
    import httpx

    last = ""
    for u in urls:
        try:
            with httpx.stream("GET", u, headers=_UA, timeout=httpx.Timeout(20.0, read=60.0),
                              follow_redirects=True) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                part = dest.with_suffix(dest.suffix + ".part")
                with open(part, "wb") as f:
                    for b in r.iter_bytes(65536):
                        f.write(b)
                part.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            last = f"{u[:36]}… {type(e).__name__}: {str(e)[:60]}"
    raise RuntimeError(last or "所有源都失败")


def resolve_model(size: str = "") -> str:
    """优先用本机已下载的目录，避免运行时再去联网。"""
    want = size or DEFAULT_SIZE
    d = model_dir(want)
    if d.exists() and any(d.glob("*.bin")):
        return str(d)
    # 该尺寸没下，但本机有别的：用已装的最好的那个，别为了一个尺寸卡住
    order = ["large-v3", "medium", "small", "base", "tiny"]
    for k in order:
        dd = model_dir(k)
        if dd.exists() and any(dd.glob("*.bin")):
            return str(dd)
    return want


def _register_cuda_dlls() -> list[str]:
    """把 pip 装的 CUDA 运行库目录挂给 Windows 的 DLL 搜索路径。

    实测：ctranslate2 的 CUDA 推理要 cublas64_12.dll / cudnn*_9.dll，
    而 `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` 把 DLL 放在
    site-packages/nvidia/<lib>/bin —— Windows 默认**不会**去那里找。
    """
    if os.name != "nt":
        return []
    import sysconfig

    added: list[str] = []
    roots = set()
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            roots.add(Path(p) / "nvidia")
    for root in roots:
        if not root.exists():
            continue
        for b in root.glob("*/bin"):
            try:
                os.add_dll_directory(str(b))
                added.append(str(b))
            except Exception:  # noqa: BLE001
                pass
    return added


def gpu_usable() -> bool:
    """当前进程里 GPU 到底能不能用。

    "有 CUDA 设备"不等于"能跑"：ctranslate2 还需要 cublas64_12.dll 等运行库。
    早先这里只数设备数，状态栏就敢写"GPU 可用"，属于误报 —— 现在真的去加载一次。
    """
    global _GPU_BROKEN
    if _GPU_BROKEN:
        return False
    try:
        import ctranslate2
        if int(ctranslate2.get_cuda_device_count()) <= 0:
            return False
    except Exception:  # noqa: BLE001
        return False
    _register_cuda_dlls()
    if os.name != "nt":
        return True
    try:
        import ctypes
        ctypes.WinDLL("cublas64_12.dll")
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_gpu_lib_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "not found or cannot be loaded"))


def _collect(model, audio_path, language: str, want_words: bool,
             **kwargs) -> tuple[list[dict], object]:
    """把生成器吃干净。

    关键：faster-whisper 的 transcribe() 只返回生成器，**真正的推理发生在迭代时**。
    GPU 缺库的异常是在 for 循环里抛的 —— 只在外面 try 住 transcribe() 抓不到。
    """
    import math

    segments, info = model.transcribe(str(audio_path), language=language, **kwargs)
    rows: list[dict] = []
    for s in segments:
        words = []
        if want_words:
            words = [{"t_start": float(w.start), "t_end": float(w.end),
                      "text": getattr(w, "word", "")}
                     for w in (getattr(s, "words", None) or [])]
        rows.append({"t_start": float(s.start), "t_end": float(s.end),
                     "text": (s.text or "").strip(), "words": words,
                     "confidence": float(math.exp(float(s.avg_logprob or 0.0)))})
    return rows, info


def transcribe_retry(model_size: str, audio_path: str | Path,
                     language: str = "zh", prefer_gpu: bool = True,
                     want_words: bool = False, **kwargs):
    """转写并回落。返回 (rows, info, 说明)。GPU 缺库时改 CPU 重试一次。"""
    global _GPU_BROKEN
    # 实测：medium 模型对普通话常输出**繁体**（"這是我的第一個項目"）。
    # Whisper 官方给的解法是用 initial_prompt 把语言风格钉住，这里固定成简体书面语。
    kwargs.setdefault("initial_prompt", SIMPLIFIED_HINT)
    # 每段都接着上一段的上下文会让长课越译越飘，关掉更稳
    kwargs.setdefault("condition_on_previous_text", False)
    model, how = load_model(model_size, prefer_gpu)
    try:
        rows, info = _collect(model, audio_path, language, want_words, **kwargs)
        return rows, info, how
    except Exception as e:  # noqa: BLE001
        if not prefer_gpu or not _is_gpu_lib_error(e):
            raise
        _GPU_BROKEN = True
        model, how2 = load_model(model_size, False)
        rows, info = _collect(model, audio_path, language, want_words, **kwargs)
        return rows, info, f"{how2}｜GPU 缺库已回落 CPU（{str(e)[:60]}）"


def pick_device(prefer_gpu: bool = True) -> tuple[str, str]:
    """给 ctranslate2 选 (device, compute_type)。

    CUDA 需要 cuDNN 9，缺了会在加载时报错 —— 所以这里只做"想不想用 GPU"的判断，
    真失败由 load_model 兜底回 CPU。
    """
    if prefer_gpu and gpu_usable():
        return "cuda", "float16"
    return "cpu", "int8"


def load_model(size: str = "", prefer_gpu: bool = True):
    """加载模型，GPU 起不来就退回 CPU（并说明为什么）。返回 (model, 说明)。"""
    from faster_whisper import WhisperModel

    path = resolve_model(size)
    device, compute = pick_device(prefer_gpu)
    if device == "cuda":
        dlls = _register_cuda_dlls()
        if not dlls:
            _GPU_BROKEN = True          # 没有 CUDA 运行库就别硬试
            device, compute = "cpu", "int8"
    tried = []
    for dev, ct in ((device, compute), ("cpu", "int8")):
        try:
            t0 = time.time()
            m = WhisperModel(path, device=dev, compute_type=ct)
            tried.append(f"{dev}/{ct} 加载 {time.time() - t0:.1f}s")
            return m, "｜".join(tried) + f"｜模型 {Path(path).name}"
        except Exception as e:  # noqa: BLE001
            tried.append(f"{dev}/{ct} 失败：{type(e).__name__}: {str(e)[:90]}")
    raise RuntimeError("模型加载失败：" + "｜".join(tried))


def transcribe_file(audio_path: str | Path, size: str = "",
                    language: str = "zh", prefer_gpu: bool = True) -> dict:
    """真实转写一段音频，返回文字/分段/用时。用于设置页的「测试转写」。"""
    p = Path(audio_path)
    if not p.exists():
        return {"ok": False, "error": f"音频不存在：{p}"}
    t0 = time.time()
    try:
        rows0, info, how = transcribe_retry(
            size, p, language=language, prefer_gpu=prefer_gpu,
            vad_filter=True, without_timestamps=False)
        rows = [{"t_start": round(r["t_start"], 2), "t_end": round(r["t_end"], 2),
                 "text": r["text"]} for r in rows0 if r["text"]]
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"转写失败：{type(e).__name__}: {str(e)[:200]}"}
    dur = time.time() - t0
    media = float(getattr(info, "duration", 0.0) or (rows[-1]["t_end"] if rows else 0))
    return {"ok": bool(rows), "segments": rows, "n": len(rows),
            "seconds": round(dur, 1), "media_seconds": round(media, 1),
            "speed": round(media / dur, 2) if dur > 0 else 0.0,
            "model": Path(resolve_model(size)).name, "device": how,
            "error": "" if rows else "没识别出任何分段"}