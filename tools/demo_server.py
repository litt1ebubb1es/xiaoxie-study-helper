"""演示站的静态文件服务器：纯标准库，但补上了 http.server 缺的那件事。

为什么不用 `python -m http.server`：
    它忽略 Range 请求，整片一次性返回 200。视频因此**拖不动进度条**——
    每次 seek 浏览器都得从头下整个文件。这台机器上行只有 1.6 Mbps，
    一个 10 MB 的片子重下几次就是灾难。

这里实现的是生产静态站最小的那套：
    GET / HEAD、Range（含 suffix 区间）、Content-Type、条件请求（mtime）、
    目录穿越防护、访问日志。没有依赖、没有框架，内存占用和 http.server 一个量级。

用法：
    python demo_server.py [端口] [根目录]
"""

from __future__ import annotations

import datetime
import email.utils
import posixpath
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve()

TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".woff2": "font/woff2",
}

# 文本类不缓存（改完刷新就见效），视频/图片长缓存并支持 Range 拖动
NO_CACHE = {".html", ".json"}
LONG_CACHE = {".mp4", ".webm", ".m4a", ".mp3", ".jpg", ".jpeg", ".png", ".webp", ".woff2"}

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK = 256 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "xiaoxie-demo"
    protocol_version = "HTTP/1.1"          # 保持连接：视频分片请求效率高得多

    # ---------- 路径 ----------
    def _resolve(self) -> Path | None:
        """URL → ROOT 内的真实文件；越界或不存在返回 None。"""
        path = unquote(self.path.split("?", 1)[0].split("#", 1)[0])
        if not path.startswith("/"):
            return None
        clean = posixpath.normpath(path).lstrip("/")
        target = (ROOT / clean).resolve() if clean else ROOT
        if target != ROOT and ROOT not in target.parents:
            return None                                  # 目录穿越
        if target.is_dir():
            target = target / "index.html"
        return target if target.is_file() else None

    # ---------- 请求入口 ----------
    def do_HEAD(self) -> None:
        self._serve(head_only=True)

    def do_GET(self) -> None:
        self._serve(head_only=False)

    def _serve(self, head_only: bool) -> None:
        plain_path = self.path.split("?", 1)[0]
        if plain_path in ("/health", "/healthz"):
            self._body(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok", head_only)
            return

        target = self._resolve()
        if target is None:
            self._body(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8",
                       "404 没找到这个文件".encode("utf-8"), head_only)
            return

        st = target.stat()
        size, mtime = st.st_size, st.st_mtime
        if self._not_modified(mtime):
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Range 解析
        start, end, partial = 0, size - 1, False
        rng = self.headers.get("Range", "").strip()
        if rng:
            m = RANGE_RE.match(rng)
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1 == "" and g2 != "":                # bytes=-N → 最后 N 字节
                    n = int(g2)
                    if n <= 0:
                        self._unsatisfiable(size, head_only)
                        return
                    start, end, partial = max(0, size - n), size - 1, True
                else:
                    start = int(g1)
                    end = int(g2) if g2 else size - 1
                    if start >= size or end < start:
                        self._unsatisfiable(size, head_only)
                        return
                    end = min(end, size - 1)
                    partial = True

        # 状态行必须先发，再发头（顺序反了客户端解析不了）
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        ext = target.suffix.lower()
        self.send_header("Content-Type", TYPES.get(ext, "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", email.utils.formatdate(mtime, usegmt=True))
        if ext in NO_CACHE:
            self.send_header("Cache-Control", "no-cache")
        elif ext in LONG_CACHE:
            self.send_header("Cache-Control", "public, max-age=604800")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()

        if head_only:
            return

        # 分块送：大文件不整块读进内存（这台机器只有 2 GB）
        try:
            with open(target, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                                          # 用户拖进度条/关页面，正常

    # ---------- 小工具 ----------
    def _body(self, code: HTTPStatus, ctype: str, body: bytes, head_only: bool) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _unsatisfiable(self, size: int, head_only: bool) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _not_modified(self, mtime: float) -> bool:
        ims = self.headers.get("If-Modified-Since")
        if not ims:
            return False
        try:
            since = email.utils.parsedate_to_datetime(ims).timestamp()
        except (TypeError, ValueError):
            return False
        return int(mtime) <= int(since)

    def log_message(self, fmt: str, *args) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"[{stamp}] {self.address_string()} {fmt % args}\n")


def main() -> int:
    if not ROOT.is_dir():
        print(f"根目录不存在: {ROOT}")
        return 2
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"服务 {ROOT} 于 http://0.0.0.0:{PORT}  (Range 已启用)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())