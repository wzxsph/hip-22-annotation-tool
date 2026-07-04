@echo off
setlocal

set "EXE=%~dp0Hip22AnnotationTool.exe"
set "LOGDIR=%LOCALAPPDATA%\Hip22AnnotationTool\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set "STAMP=%%I"
set "LOGFILE=%LOGDIR%\run-%STAMP%.log"

set "HIP22_LOG_LEVEL=info"

echo Hip 22 Annotation Tool
echo ======================
echo Log file: %LOGFILE%
echo Close this window to stop the tool.
echo.

"%EXE%" 1>"%LOGFILE%" 2>&1

echo.
echo Exited. Log file: %LOGFILE%
pause
