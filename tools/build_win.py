"""把应用打成"解压即用"的 Windows 文件夹（PyInstaller onedir）+ 一个 zip。

用法（在项目根目录）：

    .venv\\Scripts\\python.exe tools\\build_win.py            # 打包 + 打 zip
    .venv\\Scripts\\python.exe tools\\build_win.py --console  # 带黑框版本（排查启动问题用）
    .venv\\Scripts\\python.exe tools\\build_win.py --no-zip   # 只要文件夹

**为什么 onedir 不做 onefile**

- onefile 每次启动都要把 PySide6 那几百 MB 解到 %TEMP%，冷启动十几秒；
- 杀软会反复扫这个"自解压"行为，第一次运行还可能被拦；
- onedir 就是普通进程启动，双击即开，代价是发一个文件夹——
  对"把 zip 发给别人"这个场景完全够用。

**发出去之前要检查什么**

1. `dist\\小蟹学习助手\\小蟹学习助手.exe --selftest` 能跑通（结果写在 data\\logs\\app.log）；
2. 双击 exe 能开界面，设置页填上自己的 key 能拉到模型列表；
3. 目标机器没装 Python / 没装 VC 运行库也能开（PySide6 自带依赖）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "小蟹学习助手"
ICON = ROOT / "docs" / "icon" / "xiaoxie.ico"
DIST = ROOT / "dist"
BUILD = ROOT / "build"

# 这些是运行时才 import 的东西（或在 try/except 里 import），静态分析容易漏
HIDDEN = [
    "_portaudiowpatch",          # pyaudiowpatch 的 C 扩展（WASAPI 环回录音）
    "pyaudiowpatch",
    "pycaw",                     # 判断"浏览器是不是在放视频"
    "pycaw.pycaw",
    "comtypes",
    "comtypes.client",
    "comtypes.gen",
    "sounddevice",
    "soundfile",
    "imageio_ffmpeg",
    "psutil",
    "openai",
    "httpx",
    "dsh_app.netfix",            # 启动即用的兼容补丁
]

# 明确不收：体积大、跟"云端 API 路线"无关的本地转写栈
# （faster-whisper / ctranslate2 / onnxruntime 加起来约 180 MB；
#   云端与侧车字幕路线用不到它们，`dsh_app/asr` 里是 try/except 降级，
#   所以精简版照常能用，只是"本地离线转写"这一档不可选。要全功能加 --full。）
SLIM = [
    "faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers", "hf_xet",
    "cv2", "lxml", "torch", "torchaudio", "transformers",
]

# 无论如何都不收：测试与没用到的 GUI 模块
EXCLUDE = [
    "pytest", "_pytest", "matplotlib", "pandas", "IPython", "notebook",
    "tkinter", "test", "tests", "unittest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
    "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia", "PySide6.QtQml",
]

USAGE_TXT = """小蟹学习助手 —— 免安装版
================================

**这个版本不需要装任何依赖**：Python、Qt、ffmpeg、音频库都在 _internal\\ 里了，
拷到没装过 Python 的电脑上也是双击就能开。（想验证：双击「一键自检.bat」）

1. 双击「小蟹学习助手.exe」即可打开。不需要装 Python。
2. 第一次打开后去「设置与模型部署」：
   - 「服务商」选一家（阿里云百炼 / DeepSeek / 智谱…），地址会自动填好；
   - 点「去哪拿 Key？」去官网申请，粘到「API Key」里；
   - 点「刷新列表」→ 模型下拉会出现真实可用的模型，选一个 → 「保存设置」。
   不知道选哪家、怎么开 key：点设置页里的「怎么导入模型？（图文说明）」。

打不开 / 想确认有没有装坏
--------------------------
双击「一键自检.bat」：它会跑一遍完整自检（建库→分析→诊断），
把结果打在黑框里，并把 data\\logs\\app.log 打开给你看。
自检过了就说明程序本身没问题，问题多半在 Key / 网络 / 录音设备上。

数据存在哪
----------
这个文件夹里的 data\\ 子目录：
  config.json   你的设置和 API Key（只在本机，不上传）
  app.db        课程、会话、知识点、待答队列
  audio\\       录音
  logs\\        日志（出问题时把这个文件夹发回来）
整个文件夹拷走/换电脑，直接带着这些数据。删掉 data\\ 就是"恢复出厂"。

提示
----
- 第一次运行 Windows 可能弹「已保护你的电脑」：点「更多信息」→「仍要运行」。
  这是因为 exe 没买代码签名证书，不是病毒。
- 首次运行杀软扫描会慢几秒，之后正常。
- 不需要管理员权限；装到 C:\\Program Files 里也能跑（数据会自动存到
  %LOCALAPPDATA%\\小蟹学习助手\\data）。
- Windows 运行库已随包自带（_internal\\VCRUNTIME140.dll、PySide6\\MSVCP140.dll）；
  万一系统提示缺 MSVCP140.dll，装一次「Microsoft Visual C++ 2015-2022 运行库 x64」。
"""

# 包里的一键自检：不打开界面，直接跑核心链路并把日志弹出来。
# 发给不懂技术的人时，这是"出问题了先点这个"的入口——比让他去命令行敲参数现实。
SELFTEST_BAT = r"""@echo off
chcp 936 >nul
cd /d "%~dp0"
title 小蟹学习助手 - 一键自检

echo ============================================
echo   小蟹学习助手 - 一键自检（不开界面，约 10 秒）
echo ============================================
echo.
echo 正在检查：数据库 / 分析 / 回看点 / 诊断 / 生成讲解 ...
echo.

"%~dp0小蟹学习助手.exe" --selftest
set "RC=%errorlevel%"

echo.
if "%RC%"=="0" (
  echo [OK] 自检通过，程序本身没问题。
  echo      如果界面里某功能不好用，多半是 API Key、网络或录音设备的问题。
) else (
  echo [x] 自检失败，退出码 %RC%
  echo     把 data\logs\app.log 发给我就能定位。
)
echo.
echo 现在打开 data 文件夹，日志在 logs\app.log
if exist "%~dp0data" start "" "%~dp0data"
echo.
pause
"""


def _check_env() -> None:
    if not ICON.exists():
        sys.exit(f"缺图标文件：{ICON}")
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit("没装 PyInstaller：.venv\\Scripts\\python.exe -m pip install \"pyinstaller>=6.6\"")


def _cmd(console: bool, full: bool) -> list[str]:
    icon = str(ICON)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir",
        "--console" if console else "--windowed",
        "--name", APP_NAME,
        "--icon", icon,
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        "--paths", str(ROOT),
        # exe 资源里的图标 + 文档里的图标都给一份，界面上画标不用外部文件但留着有用
        "--add-data", f"{icon}{os.pathsep}docs/icon",
        "--add-data", f"{ROOT / 'LICENSE'}{os.pathsep}.",
    ]
    for h in HIDDEN:
        cmd += ["--hidden-import", h]
    for e in EXCLUDE + ([] if full else SLIM):
        cmd += ["--exclude-module", e]
    cmd.append(str(ROOT / "run_app.py"))
    return cmd


def _post(dist_dir: Path) -> None:
    """把用户真正需要的东西放进文件夹：使用说明、许可、README、一键自检；清掉我这台机器的运行痕迹。"""
    (dist_dir / "使用说明.txt").write_text(USAGE_TXT, encoding="utf-8")
    # .bat 必须是 GBK：cmd 按系统 ANSI 读脚本，UTF-8 的中文会变成乱码指令
    (dist_dir / "一键自检.bat").write_bytes(
        SELFTEST_BAT.replace("\n", "\r\n").encode("gbk"))
    for name in ("LICENSE", "README.md"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dist_dir / name)
    # data 目录先建好，用户一眼能看到"东西存这儿"；
    # 但绝不能把我自检时的库/日志顺便发出去（里面可能有我试过的配置）。
    data = dist_dir / "data"
    data.mkdir(exist_ok=True)
    for junk in ("selftest.db", "app.db", "config.json", "bili_session.json"):
        (data / junk).unlink(missing_ok=True)
    logs = data / "logs"
    if logs.is_dir():
        for f in logs.glob("*"):
            if f.is_file():
                f.unlink()


def _zip(dist_dir: Path) -> Path:
    from dsh_app import __version__          # 版本号跟着包走，不用手改

    out = DIST / f"{APP_NAME}-v{__version__}-win64.zip"
    if out.exists():
        out.unlink()
    skip_dirs = {("data", "audio"), ("data", "logs"), ("data", "export"),
                 ("data", "backups"), ("data", "models"), ("data", "deploy")}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(dist_dir.rglob("*")):
            rel = p.relative_to(dist_dir)
            if rel.parts[:2] in skip_dirs:            # 空目录留着，内容不发
                continue
            if p.is_file() and rel.parts[:1] == ("data",):
                continue                              # 库/配置/日志一律不发
            z.write(p, p.relative_to(dist_dir.parent))
    return out


def _size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1024 / 1024
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="打包成 Windows 免安装文件夹")
    ap.add_argument("--console", action="store_true", help="保留控制台窗口（排查用）")
    ap.add_argument("--no-zip", action="store_true", help="只出文件夹，不打 zip")
    ap.add_argument("--full", action="store_true",
                    help="连本地转写栈一起打（faster-whisper/ctranslate2/onnxruntime，+180MB）")
    args = ap.parse_args(argv)

    _check_env()
    print(f"入口：{ROOT / 'run_app.py'}")
    print(f"输出：{DIST}")

    print("模式：" + ("全功能（含本地离线转写）" if args.full else "精简（云端 API + 侧车字幕）"))
    r = subprocess.run(_cmd(args.console, args.full))
    if r.returncode != 0:
        print(f"打包失败（PyInstaller 退出码 {r.returncode}）")
        return r.returncode

    dist_dir = DIST / APP_NAME
    exe = dist_dir / f"{APP_NAME}.exe"
    if not exe.exists():
        print(f"没找到产物：{exe}")
        return 1
    _post(dist_dir)

    print(f"\néxe  ：{exe}")
    print(f"文件夹：{_size_mb(dist_dir):.0f} MB")
    if not args.no_zip:
        z = _zip(dist_dir)
        print(f"zip  ：{z}（{_size_mb(z):.0f} MB）")

    print("\n下一步自检（在目标机器上同样适用）：")
    print(f'  "{exe}" --selftest        # 结果写到 data\\logs\\app.log')
    print(f'  双击 "{exe}" 开界面')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))