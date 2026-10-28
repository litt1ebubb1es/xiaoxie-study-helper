"""附件：把用户拖进来／粘贴进来的东西变成能发给模型的内容。

为什么要单独一个模块：
1. 判断"这文件能不能发、按图片发还是按文本发"是有规则的，不能散在界面事件里；
2. 规则要能离线测（不启动 Qt 也能测分类、大小限制、错误文案）；
3. 图片要压（原图 4MB 直接塞进请求会超时），压缩逻辑已经在 screen.encode_jpeg_bytes，
   这里只负责调用和兜底。

三类东西：
- 图片（png/jpg/webp/bmp/gif 首帧）→ base64 data URL，走视觉模型看图；
- 文档（pdf/docx/pptx/txt/md/srt/vtt/csv…）→ 抽成纯文本当"资料"塞进提示词；
- 其它（exe/zip/mp4…）→ 明确拒绝并给出原因，不要静默丢掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import kb as kbmod
from .screen import encode_jpeg_bytes

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
DOC_SUFFIXES = ({".pdf", ".docx", ".pptx", ".doc", ".ppt"}
                | kbmod.TEXT_SUFFIXES | kbmod.CODE_SUFFIXES)

MAX_IMAGES = 4              # 一次最多几张图：再多模型也看不过来，还容易超时
MAX_FILE_CHARS = 20000      # 单个文档最多喂这么多字，超出截断并说明
MAX_IMAGE_SIDE = 1400       # 长边像素上限


@dataclass
class Attachment:
    kind: str                     # "image" | "file"
    name: str
    data_url: str = ""            # 图片：data:image/jpeg;base64,...
    text: str = ""                # 文档：抽出来的正文
    path: str = ""
    raw: bytes = b""              # 图片原字节，界面上做缩略图用
    note: str = ""                # 处理说明（截断、抽了几页…）
    error: str = ""               # 非空表示这个附件不可用

    @property
    def ok(self) -> bool:
        return not self.error


def _read_text(path: Path) -> tuple[str, str]:
    try:
        text, note = kbmod.read_document(path)
    except Exception as e:                        # noqa: BLE001
        return "", f"读不出来：{type(e).__name__}: {str(e)[:80]}"
    return text, note


def load_path(path: str | Path) -> Attachment:
    """把一个路径变成附件。不抛异常：失败也返回带 error 的附件，界面照实展示。"""
    p = Path(path)
    name = p.name
    suffix = p.suffix.lower()
    if not p.exists():
        return Attachment(kind="file", name=name, path=str(p), error="文件不存在")
    if p.is_dir():
        return Attachment(kind="file", name=name, path=str(p), error="这是一个文件夹")

    if suffix in IMAGE_SUFFIXES:
        try:
            raw = p.read_bytes()
        except Exception as e:                    # noqa: BLE001
            return Attachment(kind="image", name=name, path=str(p),
                              error=f"读不出来：{type(e).__name__}")
        return Attachment(kind="image", name=name, path=str(p), raw=raw,
                          data_url=encode_jpeg_bytes(raw, max_side=MAX_IMAGE_SIDE))

    if suffix in DOC_SUFFIXES:
        text, note = _read_text(p)
        if not text.strip():
            return Attachment(kind="file", name=name, path=str(p),
                              error=note or "没抽到文字（可能是扫描版 PDF）")
        clipped = text[:MAX_FILE_CHARS]
        if len(text) > MAX_FILE_CHARS:
            note = (note + "；" if note else "") + f"只取前 {MAX_FILE_CHARS} 字"
        return Attachment(kind="file", name=name, path=str(p), text=clipped, note=note)

    return Attachment(kind="file", name=name, path=str(p),
                      error=f"不支持这种文件（{suffix or '无后缀'}）")


def load_paths(paths: list[str] | tuple[str, ...]) -> list[Attachment]:
    return [load_path(p) for p in paths]


def image_attachment(raw: bytes, name: str = "剪贴板截图.png") -> Attachment:
    """剪贴板里的图：没有路径，只有字节。"""
    if not raw:
        return Attachment(kind="image", name=name, error="空图片")
    return Attachment(kind="image", name=name, raw=raw,
                      data_url=encode_jpeg_bytes(raw, max_side=MAX_IMAGE_SIDE))


def split(atts: list[Attachment]) -> tuple[list[Attachment], list[Attachment], list[Attachment]]:
    """返回 (可用图片, 可用文档, 有问题)。"""
    images = [a for a in atts if a.ok and a.kind == "image"]
    files = [a for a in atts if a.ok and a.kind == "file"]
    bad = [a for a in atts if not a.ok]
    return images, files, bad


def to_prompt(images: list[Attachment], files: list[Attachment]) -> str:
    """把附件正文拼成提示词里的一段（图片不在文本里，走 image_url）。"""
    parts: list[str] = []
    for a in files:
        head = f"【附件《{a.name}》"
        if a.note:
            head += f"（{a.note}）"
        parts.append(f"{head}】\n{a.text}")
    return "\n\n".join(parts)


def history_label(atts: list[Attachment]) -> str:
    """存进对话历史时用的一行说明，让人回看时知道当时发过什么。"""
    names = [a.name for a in atts if a.ok]
    if not names:
        return ""
    return "　[附件：" + "、".join(names) + "]"