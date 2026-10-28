"""本地模型：检测 + 一键部署（用户裁决：N1X 机器可选本地模型，挂检测与一键部署）。

设计原则：
- **不假装模型一定存在**。检测只看三件事：本机有没有权重文件、对应端口有没有服务在跑、
  显存/内存够不够。三者都明确回报，不做乐观推断。
- **不静默下载几十 GB**。部署脚本先落盘给你看，点击"启动"才执行，日志实时写入。
- **普通电脑自动建议云端**。`recommend()` 按显存给建议，而不是按型号硬编码。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import DATA_DIR, DEPLOY_DIR, LOCAL_MODEL_REGISTRY

MODEL_ROOT = DATA_DIR / "models"
CACHE_DIRS = [
    Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope" / "hub")),
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")),
    Path.home() / ".cache" / "modelscope",
]


@dataclass
class ModelStatus:
    key: str
    name: str
    role: str
    plan_ref: str
    size_gb: float
    min_mem_gb: float
    serve: str
    port: int
    modelscope_id: str
    downloaded: bool = False
    download_path: str = ""
    serving: bool = False
    endpoint: str = ""
    models: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def state(self) -> str:
        if self.serving:
            return "运行中"
        if self.downloaded:
            return "已下载未启动"
        if self.note.startswith("端口"):
            return "端口被占用"
        return "未下载"


def _port_ok(port: int, timeout: float = 1.2) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return True
    except Exception:
        return False


def _probe_openai(port: int) -> tuple[str, list[str]]:
    """端口探测分三态，避免把"端口被别的程序占着"谎报成"模型在跑"。

    - "closed"    没人监听
    - "listening" 有进程在监听，但 /v1/models 取不到模型清单（不像 OpenAI 兼容服务）
    - "openai"    真的返回了 OpenAI 兼容的模型清单
    """
    if not _port_ok(port):
        return "closed", []
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=2.5) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        models = [str(m.get("id", "")) for m in data.get("data", [])]
        models = [m for m in models if m]
        return ("openai" if models else "listening"), models
    except Exception:
        return "listening", []


class _ScanTimeout(Exception):
    pass


def _walk_limited(root: Path, max_depth: int, deadline: float,
                  max_dirs: int = 4000):
    """限深限时的目录遍历。

    之前的 rglob 会把整棵模型缓存扫一遍（几万目录），导致"检测"按钮永远转圈。
    这里限制深度与总目录数，并设硬截止时间。
    """
    root_depth = len(root.parts)
    seen = 0
    for dirpath, dirnames, _files in os.walk(root, onerror=lambda e: None):
        seen += 1
        if seen > max_dirs or time.time() > deadline:
            raise _ScanTimeout()
        if len(Path(dirpath).parts) - root_depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath)


def _find_local_weights(entry: dict, budget_s: float = 1.2) -> str:
    """在约定目录与模型缓存里找权重。只看目录名，限深限时。"""
    name = entry["key"]
    cand = MODEL_ROOT / name
    try:
        if cand.exists() and any(cand.iterdir()):
            return str(cand)
    except Exception:
        pass
    mid = entry.get("modelscope_id", "")
    short = (mid.split("/")[-1] or "").lower()
    if not short:
        return ""
    deadline = time.time() + budget_s
    for base in CACHE_DIRS:
        try:
            if not base.exists():
                continue
        except Exception:
            continue
        try:
            for d in _walk_limited(base, max_depth=3, deadline=deadline):
                if short in d.name.lower():
                    return str(d)
        except _ScanTimeout:
            break
        except Exception:
            continue
    return ""


def gpu_info() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        lines = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
        gpus = []
        for l in lines:
            parts = [p.strip() for p in l.split(",")]
            if len(parts) >= 4:
                gpus.append({"name": parts[0], "total_mb": float(parts[1] or 0),
                             "used_mb": float(parts[2] or 0), "driver": parts[3]})
        if gpus:
            return {"available": True, "gpus": gpus,
                    "total_gb": round(sum(g["total_mb"] for g in gpus) / 1024, 1)}
    except Exception:
        pass
    return {"available": False, "gpus": [], "total_gb": 0.0, "note": "未检测到 NVIDIA GPU"}


def mem_info() -> dict:
    total = 0.0
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        total = st.ullTotalPhys / (1024 ** 3)
        avail = st.ullAvailPhys / (1024 ** 3)
        return {"total_gb": round(total, 1), "available_gb": round(avail, 1)}
    except Exception:
        pass
    try:
        import os as _os
        total = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        total = 0.0
    return {"total_gb": round(total, 1), "available_gb": 0.0}


def detect_all() -> dict:
    from concurrent.futures import ThreadPoolExecutor

    # 端口探测串行时 5 个模型最坏要等 5×1.2s，界面上看就是"卡住"
    def _probe(port: int):
        try:
            return _probe_openai(port)
        except Exception:
            return "closed", []

    ports = [int(e["port"]) for e in LOCAL_MODEL_REGISTRY]
    with ThreadPoolExecutor(max_workers=min(8, len(ports))) as ex:
        probes = dict(zip(ports, ex.map(_probe, ports)))

    statuses: list[ModelStatus] = []
    for e in LOCAL_MODEL_REGISTRY:
        st = ModelStatus(
            key=e["key"], name=e["name"], role=e["role"], plan_ref=e["plan_ref"],
            size_gb=float(e["size_gb"]), min_mem_gb=float(e["min_mem_gb"]),
            serve=e["serve"], port=int(e["port"]), modelscope_id=e["modelscope_id"],
        )
        state, models = probes.get(st.port, ("closed", []))
        st.serving = state == "openai"
        st.models = models
        if st.serving:
            st.endpoint = f"http://127.0.0.1:{st.port}/v1"
        elif state == "listening":
            st.note = (f"端口 {st.port} 有进程在监听，但 /v1/models 取不到清单"
                       f"（多半是别的程序占用了这个端口）")
        path = _find_local_weights(e)
        if path:
            st.downloaded = True
            st.download_path = path
        elif st.serving:
            st.downloaded = True
            st.note = "端口在返回 OpenAI 兼容清单，但没在约定目录找到权重（可能是别的方式部署的）"
        statuses.append(st)
    gpu = gpu_info()
    mem = mem_info()
    capacity = max(gpu.get("total_gb", 0.0), mem.get("total_gb", 0.0) * 0.55)
    return {
        "gpu": gpu,
        "mem": mem,
        "models": [st.__dict__ | {"state": st.state} for st in statuses],
        "capacity_gb": round(capacity, 1),
        "recommendation": recommend(capacity),
        "checked_at": time.time(),
    }


def recommend(capacity_gb: float) -> dict:
    """按可用容量给后端建议。capacity 取显存与本机内存可用量的较大者近似。"""
    if capacity_gb >= 96:
        return {"backend": "local", "model": "主脑 122B + 全栈本地",
                "reason": f"可用容量约 {capacity_gb:.0f} GB，足以按方案 4.4 全本地部署。",
                "level": "full_local"}
    if capacity_gb >= 16:
        return {"backend": "local", "model": "9B 级以下本地模型",
                "reason": (f"可用容量约 {capacity_gb:.0f} GB：可跑 4B 级量化模型与本地 ASR；"
                           "9B 需要大量 offload 到内存，重讲会明显变慢；"
                           "122B 主脑不成立，重讲质量会低于方案目标。建议生成层走云端。"),
                "level": "partial_local"}
    return {"backend": "cloud", "model": "云端 API（OpenAI 兼容）",
            "reason": (f"可用容量约 {capacity_gb:.0f} GB：不足以常驻本地大模型。"
                       "建议生成层走云端 API，采集与认知层仍在本地（隐私与成本边界不变）。"),
            "level": "cloud"}


# --------------------------------------------------------------------------
# 部署脚本生成
# --------------------------------------------------------------------------
_SERVE_CMD = {
    "vllm": ('python -m vllm.entrypoints.openai.api_server --served-model-name {name} '
             '--host 127.0.0.1 --port {port} --model {path}'),
    "llama.cpp": ('llama-server -m {path}\\*.gguf --host 127.0.0.1 --port {port} '
                  '--alias {name} -c 8192 -ngl 999'),
    "funasr": ('python -m funasr.bin.serve --model {path} --port {port}'),
}


def build_deploy_script(entry: dict) -> Path:
    """生成一份可审阅的 PowerShell 脚本：下载（ModelScope）→ 启动（OpenAI 兼容端口）。"""
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    key = entry["key"]
    mid = entry["modelscope_id"]
    local_dir = MODEL_ROOT / key
    serve_cmd = _SERVE_CMD.get(entry["serve"], _SERVE_CMD["vllm"]).format(
        name=key, port=entry["port"], path=str(local_dir))
    script = f"""# 由「小蟹学习助手」生成 —— {entry['name']}
# 用途：{entry['plan_ref']}；量化后约 {entry['size_gb']} GB；建议可用容量 ≥ {entry['min_mem_gb']} GB
# 审阅后执行。下载走 ModelScope（国内可达），权重落在 {local_dir}
$ErrorActionPreference = 'Stop'
$env:MODELSCOPE_CACHE = "{local_dir.parent}"

Write-Host "[1/3] 检查 modelscope 命令行" -ForegroundColor Cyan
if (-not (Get-Command modelscope -ErrorAction SilentlyContinue)) {{
    python -m pip install -U modelscope
}}

Write-Host "[2/3] 下载 {mid}" -ForegroundColor Cyan
modelscope download --model {mid} --local_dir "{local_dir}"

Write-Host "[3/3] 启动 OpenAI 兼容服务（端口 {entry['port']}）" -ForegroundColor Cyan
{serve_cmd}
"""
    path = DEPLOY_DIR / f"deploy_{key}.ps1"
    path.write_text(script, encoding="utf-8")
    return path


def launch_script(path: Path, log_name: str = "") -> subprocess.Popen:
    """后台启动部署脚本，输出重定向到日志。返回 Popen。"""
    log = DATA_DIR / "logs" / (log_name or f"{path.stem}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a", encoding="utf-8")
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)]
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def one_click_deploy(entry_key: str, dry_run: bool = True) -> dict:
    entry = next((e for e in LOCAL_MODEL_REGISTRY if e["key"] == entry_key), None)
    if not entry:
        return {"ok": False, "error": f"未知模型：{entry_key}"}
    script = build_deploy_script(entry)
    if dry_run:
        return {"ok": True, "dry_run": True, "script": str(script),
                "text": script.read_text(encoding="utf-8")}
    proc = launch_script(script)
    return {"ok": True, "dry_run": False, "script": str(script), "pid": proc.pid,
            "log": str(DATA_DIR / "logs" / f"{script.stem}.log")}