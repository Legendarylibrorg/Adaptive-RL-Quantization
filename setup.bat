@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (py -3.11 scripts\setup_from_clone.py %* & exit /b %ERRORLEVEL%)
python scripts\setup_from_clone.py %*
exit /b %ERRORLEVEL%
