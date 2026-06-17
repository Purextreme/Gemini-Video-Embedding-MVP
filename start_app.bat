@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] .venv not found. Creating virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto error
)

"%PYTHON_EXE%" -m pip show streamlit >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies from requirements.txt...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto error
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffmpeg was not found on PATH. Video thumbnails and previews may fail.
)

where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffprobe was not found on PATH. Video metadata and previews may fail.
)

echo [INFO] Starting Streamlit app...
"%PYTHON_EXE%" -m streamlit run streamlit_app.py --server.address 127.0.0.1
if errorlevel 1 goto error

exit /b 0

:error
echo.
echo [ERROR] Failed to start the app. Please check the messages above.
pause
exit /b 1
