@echo off
chcp 936 >nul
cd /d "%~dp0"
title 小蟹学习助手 - 首次安装

echo ============================================
echo   小蟹学习助手  -  首次安装（约 1-3 分钟）
echo ============================================
echo.
echo 只装一次。装完双击「启动助手.bat」就能用。
echo.

rem ---------- 1. 找 Python ----------
set "PY="
py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"

if not defined PY (
  echo [x] 没找到 Python。
  echo.
  echo     去 https://www.python.org/downloads/windows/ 装 Python 3.11 或 3.12，
  echo     安装时**勾上** "Add python.exe to PATH"，然后重新双击本文件。
  echo.
  pause
  exit /b 1
)

%PY% -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
  echo [x] Python 版本太旧，需要 3.10 以上。
  %PY% -V
  echo.
  pause
  exit /b 1
)

echo [1/4] Python 就绪：
%PY% -V

rem ---------- 2. 建虚拟环境 ----------
if exist ".venv\Scripts\python.exe" (
  echo [2/4] 已有 .venv，跳过创建。
) else (
  echo [2/4] 创建 .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [x] 创建虚拟环境失败。
    pause
    exit /b 1
  )
)

rem ---------- 3. 装依赖 ----------
echo [3/4] 安装依赖（先试国内镜像，失败再走官方源）...
".venv\Scripts\python.exe" -m pip install -q -U pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if errorlevel 1 (
  echo     镜像没成功，改用官方源再试一次...
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)
if errorlevel 1 (
  echo [x] 依赖没装成功。检查网络后重试，或手动执行：
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

rem ---------- 4. 生成带图标的启动器 ----------
echo [4/4] 生成启动器图标 ...
".venv\Scripts\python.exe" tools_patch_exe_icon.py
if errorlevel 1 (
  echo     图标没打成（不影响使用，只是任务栏显示 Python 图标）。
)

echo.
echo 自检一下（不用打开界面）...
".venv\Scripts\python.exe" run_app.py --selftest
if errorlevel 1 (
  echo [x] 自检失败，把上面的报错发我。
  pause
  exit /b 1
)

echo.
echo ============================================
echo   装好了。双击「启动助手.bat」开始用。
echo   第一次打开请去「设置与模型部署」填 API Key：
echo   点「去哪拿 Key？」申请，粘进去，点「刷新列表」选模型、保存。
echo ============================================
echo.
pause
