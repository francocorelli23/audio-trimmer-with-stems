@echo off
rem Starts Audio Trimmer with the project's own Python (.venv), without a console window.
rem To see error output instead, run:  .venv\Scripts\python.exe main.py
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py" %*
