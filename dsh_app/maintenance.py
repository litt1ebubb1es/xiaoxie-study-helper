"""课后维护：音频占用、过期清理、数据库备份与恢复。

为什么值得单独一层：这三件事都会动用户的学习资料，必须是"可测、可回滚"的纯函数，
不能散在界面回调里。界面只负责按按钮和报结果。

三条底线：

1. 清理音频只删**文件**，不动转写/特征/回看点——课还能看，只是不能再重新转写。
2. 备份用 sqlite 自己的 `backup()`，分析线程正在写库时拿到的是完整快照，
   不会拷到半截事务。
3. 恢复之前先给当前库再备份一份（`pre-restore`），手滑也有退路。
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

from .config import AUDIO_DIR, DATA_DIR, Settings, ensure_dirs
from .db import DB

BACKUP_DIR = DATA_DIR / "backups"
BACKUP_KEEP = 7


def _human_mb(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{max(0.0, n) / 1024:.0f} KB"


def audio_usage(audio_dir: Path | None = None) -> dict:
    """音频目录的家底：文件数、总字节、最老一条的时间。"""
    d = Path(audio_dir or AUDIO_DIR)
    files = [p for p in d.glob("*") if p.is_file()] if d.exists() else []
    total = sum(p.stat().st_size for p in files)
    oldest = min((p.stat().st_mtime for p in files), default=0.0)
    return {"files": len(files), "bytes": total, "oldest": oldest,
            "text": _human_mb(total)}


def sweep_audio(db: DB, cfg: Settings | None = None, *,
                now: float | None = None, audio_dir: Path | None = None) -> dict:
    """删掉超过保留天数的音频文件，并把 sessions.audio_path 清空。

    只删文件、不动记录：转写、回看点、题目、掌握度全都留着。`keep_audio_days<=0`
    表示用户选择"永不自动清理"，直接返回。
    """
    cfg = cfg or Settings.load()
    days = int(getattr(cfg.capture, "keep_audio_days", 0) or 0)
    if days <= 0:
        return {"deleted": 0, "freed": 0, "skipped": True}
    t = time.time() if now is None else float(now)
    cutoff = t - days * 86400.0
    deleted, freed = 0, 0
    for row in db.query("SELECT id, audio_path, created_at FROM sessions"):
        path = Path(row["audio_path"] or "")
        created = float(row["created_at"] or 0)
        if not row["audio_path"] or created >= cutoff:
            continue
        try:
            if path.exists():
                freed += path.stat().st_size
                path.unlink()
                deleted += 1
        except Exception:                  # noqa: BLE001 —— 文件被占用就下次再删
            continue
        db.update_session(row["id"], audio_path="")
    # 顺手收掉"没有人认领"的音频：压缩时原件偶尔没删掉（文件被占用），
    # 那之后就再没有任何记录指向它了。只删够老的文件，正在录的那份绝不碰。
    strays, stray_bytes = 0, 0
    try:
        known = {str(Path(r["audio_path"])) for r in
                 db.query("SELECT audio_path FROM sessions WHERE audio_path<>''")}
        adir = Path(audio_dir) if audio_dir else AUDIO_DIR
        if adir.exists():
            for f in adir.iterdir():
                if not f.is_file() or str(f) in known:
                    continue
                try:
                    if t - f.stat().st_mtime < max(1.0, days) * 86400.0:
                        continue
                    stray_bytes += f.stat().st_size
                    f.unlink()
                    strays += 1
                except Exception:          # noqa: BLE001
                    continue
    except Exception:                      # noqa: BLE001
        pass
    return {"deleted": deleted, "freed": freed, "skipped": False,
            "strays": strays, "stray_bytes": stray_bytes,
            "text": _human_mb(freed + stray_bytes)}


# --------------------------------------------------------------------------
# 备份 / 恢复
# --------------------------------------------------------------------------
def backup_dir() -> Path:
    ensure_dirs()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def backup_db(db: DB, *, keep: int = BACKUP_KEEP, tag: str = "auto",
              now: float | None = None) -> Path:
    """整库快照。文件名带时间戳，按时间轮转只留 `keep` 份。"""
    t = time.time() if now is None else float(now)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    dest = backup_dir() / f"app-{tag}-{stamp}.db"
    n = 2
    while dest.exists():                  # 同一秒内连按两次，别互相覆盖
        dest = backup_dir() / f"app-{tag}-{stamp}-{n}.db"
        n += 1
    with db._lock:                        # noqa: SLF001 —— 与写库线程共用同一把锁
        target = sqlite3.connect(str(dest))
        try:
            db.conn.backup(target)
        finally:
            target.close()
    _rotate(keep)
    return dest


def _rotate(keep: int) -> None:
    """只轮转自动备份。

    手动备份（用户点「立即备份」）和恢复前的保险备份（pre-restore）不参与轮转——
    "我刚点的那份备份被下一次自动备份删了"是最容易让人不敢用备份的行为。
    这两类各自单独封顶 20 份，防无限增长。
    """
    keep = max(1, int(keep))
    auto = sorted(backup_dir().glob("app-auto-*.db"), key=lambda p: p.stat().st_mtime)
    for old in auto[:-keep]:
        try:
            old.unlink()
        except Exception:                  # noqa: BLE001
            pass
    for tag in ("manual", "pre-restore"):
        items = sorted(backup_dir().glob(f"app-{tag}-*.db"), key=lambda p: p.stat().st_mtime)
        for old in items[:-20]:
            try:
                old.unlink()
            except Exception:              # noqa: BLE001
                pass


def list_backups() -> list[dict]:
    out = []
    for p in sorted(backup_dir().glob("app-*.db"), key=lambda x: x.stat().st_mtime,
                    reverse=True):
        st = p.stat()
        out.append({"path": p, "name": p.name, "bytes": st.st_size,
                    "text": _human_mb(st.st_size), "mtime": st.st_mtime,
                    "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
    return out


def last_backup_time() -> float:
    items = list_backups()
    return items[0]["mtime"] if items else 0.0


def restore_backup(db: DB, src: str | Path) -> dict:
    """用备份覆盖当前库。恢复前先给当前库留一份 `pre-restore`。"""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"备份文件不存在：{src}")
    safety = backup_db(db, tag="pre-restore")
    db.close()
    # sqlite 的 -wal/-shm 必须一起清掉，否则旧 WAL 会把新文件覆盖回旧内容
    for suffix in ("-wal", "-shm"):
        side = Path(str(db.path) + suffix)
        if side.exists():
            try:
                side.unlink()
            except Exception:              # noqa: BLE001
                pass
    shutil.copy2(src, db.path)
    db.reopen()
    return {"restored_from": src.name, "safety": safety.name,
            "sessions": len(db.list_sessions())}