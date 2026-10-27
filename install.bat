@echo off
rem One-time setup: creates .venv with Python 3.13 and installs the core app.
cd /d "%~dp0"
if not exist .venv (
    py -3.13 -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo.
echo Done. Start the app with run.bat. For stems, also run install_stems.bat.
pause
exit /b 0

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
