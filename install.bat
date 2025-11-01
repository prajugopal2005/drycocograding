@echo off
echo ====================================
echo Coconut Purity Grading System
echo Installation Script
echo ====================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python 3.8+ is installed
    pause
    exit /b 1
)
echo ✓ Virtual environment created

echo.
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated

echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo To run the application:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run app: python app.py
echo   3. Open browser: http://127.0.0.1:5000
echo.
pause
