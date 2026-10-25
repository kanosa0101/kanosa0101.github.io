@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  start "" wscript.exe "%~dp0tools\launch.vbs"
) else (
  start "" wscript.exe "%~dp0tools\launch.vbs" "%~1"
)
exit /b 0
