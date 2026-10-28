"""F3：屏幕内容 → 视觉模型 → 文字（方案 3.4）。

设计要点：
1. **按需抓取，不做持续录屏**。用户按 Ctrl+Shift+S 抓一帧，或按固定间隔低频抓取；
   抓到的帧只用于这一次识别，默认不留存图片（可配置留档）。
2. **识别交给视觉模型**（OpenAI 兼容协议的 image_url），不引入离线 OCR 依赖。
3. 识别结果必须**可被用户修正**：模型认错字很常见，界面里给一个可编辑框。
4. 屏幕文字与语音转写语义不同：语音有"语速/停顿"，屏幕只有"这页停留多久"。
   这个差异在 cognitive.features 里用 content_kind 区分，不在这里糊弄。
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import HARDEN_HINT, call_with_retry

PROMPT = """这是一张网课屏幕截图。请把画面里的**学习内容文字**提取出来，供后续分析用。

规则：
- 只提取课程内容：标题、正文、公式、代码、板书、幻灯片文字、批注。
- 忽略浏览器地址栏、播放器控件、进度条、弹幕、系统任务栏、聊天框、广告。
- 手写/公式尽量转成可读的线性文本；数学符号用常见写法（如 x^2、∫、≤）。
- 如果画面里几乎没有课程文字（纯视频画面、纯图、黑屏），text 返回空字符串。

只输出一个 JSON 对象，不要任何解释：
{"text": "提取到的文字，段落之间用换行", "kind": "slide|board|notes|code|formula|empty"}"""


@dataclass
class ScreenResult:
    text: str
    kind: str = "slide"
    model: str = ""
    latency_s: float = 0.0
    raw: str = ""
    error: str = ""
    retried: bool = False       # 靠「降档重问」才拿到结果
    image_b64: str = ""
    image_path: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


def encode_jpeg_bytes(raw: bytes, max_side: int = 1400, quality: int = 82) -> str:
    """把截图像素压成 base64 data URL。"""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
    except Exception:
        data = raw
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def grab_primary_screen() -> bytes:
    """抓主屏一帧，返回 PNG 字节。需要 Qt（界面进程内调用）。"""
    from PySide6.QtCore import QBuffer, QByteArray
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is None:
        raise RuntimeError("抓屏需要 Qt 应用已启动（界面里用，不要在没有界面的地方调）")
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("没有可用的屏幕")
    pix = screen.grabWindow(0)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    pix.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def parse_vision_reply(raw: str) -> tuple[str, str]:
    """从模型回复里抠出 {text, kind}。模型经常夹带 ```json 或多余说明。"""
    from .llm import parse_json_loose

    s = (raw or "").strip()
    if not s:
        return "", "empty"
    obj = parse_json_loose(s)
    if isinstance(obj, dict):
        text = str(obj.get("text") or "").strip()
        kind = str(obj.get("kind") or "slide").strip().lower()
        return text, (kind or "slide")
    # 不是 JSON 就当成纯文本用，但要标成 unknown，让人知道这是兜底
    return s, "unknown"


class VisionExtractor:
    """用 OpenAI 兼容的视觉模型识别截图文字。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = 120.0, temperature: float = 0.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "未配置 base_url"
        if not self.model:
            return False, "未配置视觉模型名"
        from urllib.parse import urlparse
        host = (urlparse(self.base_url).hostname or "").lower()
        if not self.api_key and host not in ("127.0.0.1", "localhost", "::1"):
            return False, "缺 api_key（云端视觉模型需要鉴权）"
        return True, f"{host} · {self.model}"

    def extract(self, image_bytes: bytes, max_side: int = 1400) -> ScreenResult:
        t0 = time.time()
        ok, msg = self.available()
        if not ok:
            return ScreenResult(text="", model=self.model, error=msg)
        data_url = encode_jpeg_bytes(image_bytes, max_side=max_side)

        def run(hardened: bool) -> tuple[str, str]:
            # 视觉识别同样会撞上"推理把 token 吃光"：返回空或半截 JSON。
            # 走统一的降档重问：第二次在提示词里明确"别推理"，并放宽 max_tokens。
            prompt = PROMPT if not hardened else PROMPT + "\n" + HARDEN_HINT
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 1200 if not hardened else 1800,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
            }
            try:
                import httpx

                url = f"{self.base_url}/chat/completions"
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                with httpx.Client(timeout=self.timeout_s) as c:
                    r = c.post(url, json=payload, headers=headers)
                    if r.status_code >= 400:
                        body = r.text[:400]
                        hint = ""
                        if r.status_code == 404 or "model" in body.lower():
                            hint = f"（该端点可能没有 {self.model}，换个视觉模型名试试）"
                        return "", f"HTTP {r.status_code}: {body}{hint}"
                    data: Any = r.json()
                raw = data["choices"][0]["message"]["content"]
                if isinstance(raw, list):
                    raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
                return str(raw), ""
            except Exception as e:  # noqa: BLE001
                return "", f"{type(e).__name__}: {e}"

        def usable(raw: str) -> bool:
            # 空文字也是合法结果（纯视频画面），只要模型给了能解析的 JSON 就算成功
            from .llm import parse_json_loose

            if not (raw or "").strip():
                return False
            obj = parse_json_loose(raw)
            return isinstance(obj, dict) or bool(raw.strip())

        raw, err, attempts = call_with_retry(run, usable)
        text, kind = parse_vision_reply(raw)
        return ScreenResult(text=text, kind=kind, model=self.model,
                            latency_s=time.time() - t0, raw=raw,
                            error="" if not err else err,
                            retried=attempts > 1,
                            image_b64=data_url)


def build_extractor(settings) -> VisionExtractor:
    llm = settings.llm
    return VisionExtractor(llm.base_url, llm.api_key, llm.vision_model or llm.model,
                           timeout_s=llm.timeout_s)