"""探 qwen-image-2.1-pro 到底走哪个端点能出图。

provider qw（阿里云 MaaS，openai 兼容）：
    base = https://ws-gymcmdektyoi7zpn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    model = qwen-image-2.1-pro

依次试三条路，谁先 200 就停：
  1) OpenAI 风格  POST {base}/images/generations
  2) DashScope 原生 POST {host}/api/v1/services/aigc/multimodal-generation/generation
  3) DashScope 原生 POST {host}/api/v1/services/aigc/text2image/image-synthesis（异步）
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

CRED = Path.home() / ".dsh" / ".credentials.yaml"
BASE = ("https://ws-gymcmdektyoi7zpn.cn-beijing.maas.aliyuncs.com"
        "/compatible-mode/v1")
HOST = "https://ws-gymcmdektyoi7zpn.cn-beijing.maas.aliyuncs.com"
MODEL = "qwen-image-2.1-pro"
PROMPT = ("一只戴圆框眼镜的极简小人，黑白粗墨线、圆形半调网点，"
          "纯白背景，漫画风格，构图居中")
OUT = Path("data/probe/oilvisual")


def keys() -> dict[str, str]:
    out = {}
    for line in CRED.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([A-Z0-9_]+):\s*(\S+)\s*$", line)
        if m and m.group(1).endswith("API_KEY"):
            out[m.group(1)] = m.group(2).strip().strip('"\'')
    return out


def post(url: str, key: str, payload: dict, extra: dict | None = None,
         timeout: int = 180) -> tuple[int, object]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    for k, v in (extra or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:                                    # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def save_image(obj: object, tag: str) -> str | None:
    """把响应里的图存下来：支持 b64_json / url / DashScope results。"""
    if not isinstance(obj, dict):
        return None
    cands: list[dict] = []
    if isinstance(obj.get("data"), list):
        cands += [c for c in obj["data"] if isinstance(c, dict)]
    out = obj.get("output") or {}
    if isinstance(out.get("results"), list):
        cands += [c for c in out["results"] if isinstance(c, dict)]
    if isinstance(out.get("choices"), list):
        for ch in out["choices"]:
            msg = ch.get("message") or {}
            for part in (msg.get("content") or []):
                if isinstance(part, dict) and part.get("image"):
                    cands.append({"url": part["image"]})
    OUT.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cands):
        b64 = c.get("b64_json")
        if b64:
            p = OUT / f"{tag}-{i}.png"
            p.write_bytes(base64.b64decode(b64))
            return str(p)
        url = c.get("url") or c.get("image_url")
        if url:
            p = OUT / f"{tag}-{i}.png"
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    p.write_bytes(r.read())
                return str(p)
            except Exception as e:                            # noqa: BLE001
                return f"下载失败 {e}"
    return None


def main() -> int:
    ks = keys()
    if not ks:
        print("没找到 API key")
        return 1
    order = [k for k in ("QW_API_KEY", "QWEN_TOKEN_PLAN_CN_API_KEY") if k in ks] + \
            [k for k in ks if k not in ("QW_API_KEY", "QWEN_TOKEN_PLAN_CN_API_KEY")]

    plans = [
        ("images/generations", f"{BASE}/images/generations",
         {"model": MODEL, "prompt": PROMPT, "n": 1, "size": "1024x1024"}, None),
        ("multimodal", f"{HOST}/api/v1/services/aigc/multimodal-generation/generation",
         {"model": MODEL,
          "input": {"messages": [{"role": "user", "content": [{"text": PROMPT}]}]},
          "parameters": {"size": "1024*1024", "n": 1, "watermark": False}}, None),
        ("text2image-async", f"{HOST}/api/v1/services/aigc/text2image/image-synthesis",
         {"model": MODEL, "input": {"prompt": PROMPT},
          "parameters": {"size": "1024*1024", "n": 1, "watermark": False}},
         {"X-DashScope-Async": "enable"}),
    ]

    for keyname in order[:2]:
        key = ks[keyname]
        print(f"\n########## 用 {keyname}（{key[:6]}…，{len(key)} 位）")
        for name, url, payload, extra in plans:
            t0 = time.time()
            code, resp = post(url, key, payload, extra)
            dt = time.time() - t0
            body = json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else str(resp)
            print(f"\n--- {name}: HTTP {code}（{dt:.1f}s）")
            print(body[:600])
            if code == 200:
                if "task_id" in body and "results" not in body:
                    print("   异步任务已建，task_id 见上（这一版先只验证能不能建单）")
                    return 0
                path = save_image(resp, name.replace('/', '-'))
                if path:
                    print(f"\n[成功] 图已存：{path}")
                    return 0
        print("   （这条路没出图，换下一把 key）")
    print("\n[全部失败]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())