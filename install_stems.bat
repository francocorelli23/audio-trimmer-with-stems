@echo off
rem Optional: installs PyTorch (GPU build) + Demucs for stem separation (~3 GB download).
cd /d "%~dp0"
if not exist .venv (
    echo Run install.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-stems.txt || goto :error
echo.
echo Stems installed. Restart Audio Trimmer and click "Separate stems".
pause
exit /b 0

:error
echo.
echo Install failed. See the messages above.
pause
exit /b 1
