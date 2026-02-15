@echo off
REM ==============================================================================
REM setup_reflex_dashboard.bat
REM ==============================================================================
REM Setup script for the Reflex (React in Python) Dashboard
REM ==============================================================================

echo.
echo ============================================================
echo   Trading Research Dashboard - Reflex Setup
echo ============================================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo        ERROR: Python not found!
    pause
    exit /b 1
)
echo        Python found

REM Install Reflex
echo [2/4] Installing Reflex (this may take a minute)...
pip install reflex --quiet
if %ERRORLEVEL% neq 0 (
    echo        ERROR: Failed to install Reflex
    pause
    exit /b 1
)
echo        Reflex installed

REM Create dashboard directory
echo [3/4] Setting up dashboard...
if not exist "dashboard_app" mkdir dashboard_app
cd dashboard_app

REM Initialize Reflex project if needed
if not exist "rxconfig.py" (
    echo        Initializing Reflex project...
    reflex init --template blank
)

REM Copy dashboard file
echo [4/4] Copying dashboard...
copy /Y "..\dashboard_reflex.py" "dashboard_app\dashboard_app.py" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo        Note: Please manually copy dashboard_reflex.py to dashboard_app\dashboard_app\dashboard_app.py
)

cd ..

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To run the dashboard:
echo.
echo     cd dashboard_app
echo     reflex run
echo.
echo   Then open: http://localhost:3000
echo.

set /p RUNNOW="Run the dashboard now? (Y/N): "
if /i "%RUNNOW%"=="Y" (
    cd dashboard_app
    reflex run
)
