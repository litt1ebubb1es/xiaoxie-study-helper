@echo off
cd /d "%~dp0"

rem 任务栏图标取自"真正在跑的那个 exe"，而 venv 里的 pythonw.exe 只是转发器，
rem 所以带图标的副本放在 base 解释器目录里（见 tools_patch_exe_icon.py）。
rem venv 的依赖由 run_app.py 自己补 sys.path，这里不需要设环境变量。
for /f "delims=" %%i in ('.venv\Scripts\python.exe -c "import sys;print(sys.base_prefix)"') do set "BASE=%%i"

if not exist "%BASE%\小蟹学习助手.exe" (
  echo [first run] patching the launcher icon ...
  ".venv\Scripts\python.exe" tools_patch_exe_icon.py
)
if not exist "%BASE%\小蟹学习助手.exe" (
  echo [error] launcher missing, install deps first per README.
  pause
  exit /b 1
)
start "" "%BASE%\小蟹学习助手.exe" "run_app.py"
exit /b 0
