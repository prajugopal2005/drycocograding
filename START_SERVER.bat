@echo off
echo ============================================================
echo   Automated Purity Grading System for Dry Coconuts
echo ============================================================
echo.
echo Starting Flask server...
echo Access the application at: http://127.0.0.1:5000
echo.
echo To stop the server, press Ctrl+C
echo ============================================================
echo.

cd /d %~dp0
.\venv\Scripts\python.exe app.py

pause

