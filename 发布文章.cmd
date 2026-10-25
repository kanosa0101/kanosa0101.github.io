@echo off
setlocal
cd /d "%~dp0"
start "" wscript.exe "%~dp0tools\launch.vbs" "%~1"
exit /b 0
