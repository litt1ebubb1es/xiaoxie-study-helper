"""在 GitHub 上建 v0.2.0 Release 并上传 zip 资产。

用 git 凭据里已有的 token（不打印明文）。先建 release，再上传资产，
最后回读一遍确认资产大小和下载 URL。
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "litt1ebubb1es/xiaoxie-study-helper"
TAG = "v0.2.0"
ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "dist" / "xiaoxie-study-helper-v0.2.0-win64.zip"
NOTES = ROOT / "docs" / "发布说明-v0.2.0.md"


def token() -> str:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("拿不到 GitHub 凭据")


def api(url: str, tok: str, data: bytes | None = None, method: str = "GET",
        ctype: str = "application/json", timeout: int = 120):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        # 404 是"还没有这个 release"，属于正常分支，不能当异常炸掉
        body = e.read()
        try:
            info = json.loads(body) if body else {}
        except Exception:
            info = {"raw": body[:300].decode("utf-8", "replace")}
        return e.code, info


def main() -> int:
    tok = token()
    if not ZIP.exists():
        raise SystemExit(f"zip 不存在：{ZIP}")
    size = ZIP.stat().st_size
    print(f"资产：{ZIP.name}  {size:,} 字节")
    print(f"说明：{NOTES.name}  {NOTES.stat().st_size:,} 字节")

    payload = json.dumps({
        "tag_name": TAG,
        "name": TAG,
        "body": NOTES.read_text(encoding="utf-8"),
        "draft": False,
        "prerelease": False,
    }).encode()

    # 已经建过就先删掉重来，避免同名 release 冲突
    status, rel = api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", tok)
    if status == 200:
        print(f"已存在 {TAG}（id={rel['id']}），先删除")
        api(f"https://api.github.com/repos/{REPO}/releases/{rel['id']}", tok, method="DELETE")

    status, rel = api(f"https://api.github.com/repos/{REPO}/releases", tok,
                      data=payload, method="POST")
    print(f"建 release：{status}  id={rel.get('id')}  {rel.get('html_url')}")
    rid = rel["id"]

    up = (f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets"
          f"?name={ZIP.name}")
    print(f"上传中（{size / 1048576:.0f} MB，慢是正常的）…")
    status, asset = api(up, tok, data=ZIP.read_bytes(), method="POST",
                        ctype="application/zip", timeout=1800)
    print(f"上传：{status}  state={asset.get('state')}  "
          f"{asset.get('size'):,} 字节")
    print(f"下载地址：{asset.get('browser_download_url')}")

    status, again = api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", tok)
    for a in again.get("assets", []):
        ok = a["size"] == size
        print(f"回读：{a['name']}  {a['size']:,} 字节  与本地一致={ok}  "
              f"下载数={a['download_count']}")
    print(f"页面：{again.get('html_url')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())