@echo off
setlocal
where node >nul 2>nul
if errorlevel 1 (
  echo Please install Node.js 22 or newer first.
  pause
  exit /b 1
)
if not exist "%~dp0node_modules\hexo\bin\hexo" (
  echo Please run npm ci in the blog folder once before first use.
  pause
  exit /b 1
)
powershell -NoProfile -STA -File "%~dp0tools\publish-ui.ps1" -ArticlePath "%~1"
if errorlevel 1 pause
endlocal
