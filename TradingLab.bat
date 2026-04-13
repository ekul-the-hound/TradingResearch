@echo off
:: ==============================================================================
:: TradingLab.bat — Production Launcher
:: ==============================================================================
setlocal

set "PROJECT_DIR=D:\Luke Files\Coding\Developer\TradingResearch"
set "VENV_DIR=D:\Luke Files\Coding\Developer\TradingResearch\.venv"
set "PORT=8080"
set "LOGFILE=%PROJECT_DIR%\TradingLab_log.txt"
set "DISC_LOG=%PROJECT_DIR%\discovery_log.txt"
set "PYTHON_EXE=python"

title TradingLab
color 0A

echo.
echo  ============================================================
echo   TradingLab Launcher
echo  ============================================================
echo   %date% %time%
echo  ============================================================
echo.

echo [%date% %time%] === Launcher started === > "%LOGFILE%"

:: ==================================================================
:: CHECK 1: Project directory
:: ==================================================================
if not exist "%PROJECT_DIR%\" (
    echo  [FAIL] Directory not found: %PROJECT_DIR%
    goto :fatal
)
cd /d "%PROJECT_DIR%"
echo  [OK]   Project: %CD%

:: ==================================================================
:: CHECK 2: Python (prefer venv, fallback to system)
:: ==================================================================
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
    call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
    echo  [OK]   Using venv Python
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo  [FAIL] Python not found.
        goto :fatal
    )
    echo  [WARN] No venv. Using system Python.
)
for /f "tokens=*" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do echo         %%v

:: ==================================================================
:: CHECK 3: Required files
:: ==================================================================
if not exist "react_dashboard2.py" (
    echo  [FAIL] react_dashboard2.py not found
    goto :fatal
)
echo  [OK]   react_dashboard2.py

set "DISC_OK=0"
if exist "run_discovery.py" ( echo  [OK]   run_discovery.py & set "DISC_OK=1" ) else ( echo  [WARN] run_discovery.py missing )
if exist "strategy_inbox.py" ( echo  [OK]   strategy_inbox.py ) else ( echo  [WARN] strategy_inbox.py missing )

:: ==================================================================
:: CHECK 4: Dependencies
:: ==================================================================
echo.
echo  --- Dependencies ---
"%PYTHON_EXE%" -c "import reactpy; print('  [OK]   reactpy ' + getattr(reactpy, '__version__', ''))" 2>nul
if errorlevel 1 ( echo  [FAIL] reactpy — run: pip install "reactpy[fastapi]" & goto :fatal )
"%PYTHON_EXE%" -c "import uvicorn; print('  [OK]   uvicorn')" 2>nul
if errorlevel 1 ( echo  [FAIL] uvicorn — run: pip install uvicorn & goto :fatal )
"%PYTHON_EXE%" -c "import plotly; print('  [OK]   plotly ' + plotly.__version__)" 2>nul
if errorlevel 1 echo  [WARN] plotly missing
"%PYTHON_EXE%" -c "import numpy; print('  [OK]   numpy ' + numpy.__version__)" 2>nul
if errorlevel 1 echo  [WARN] numpy missing
"%PYTHON_EXE%" -c "import pandas" 2>nul
if not errorlevel 1 ( echo  [OK]   pandas ) else ( echo  [WARN] pandas missing )

:: ==================================================================
:: CHECK 5: Port
:: ==================================================================
echo.
echo  --- Network ---
netstat -ano 2>nul | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo  [WARN] Port %PORT% already in use
) else (
    echo  [OK]   Port %PORT% available
)

:: ==================================================================
:: CHECK 6: Optional services
:: ==================================================================
docker ps >nul 2>&1
if errorlevel 1 ( echo  [INFO] Docker not running ) else ( echo  [OK]   Docker running )
"%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" 2>nul
if not errorlevel 1 ( echo  [OK]   Ollama responding ) else ( echo  [INFO] Ollama not responding )

:: ==================================================================
:: LAUNCH: Discovery (background, own log file)
:: ==================================================================
echo.
echo  ============================================================
echo   Launching...
echo  ============================================================
echo.

if "%DISC_OK%"=="1" (
    echo  Starting discovery runner...
    start "TradingLab_Discovery" /min cmd /c ""%PYTHON_EXE%" run_discovery.py --continuous --interval 3600 >"%DISC_LOG%" 2>&1"
    echo  [OK]   Discovery running — logging to discovery_log.txt
)

:: ==================================================================
:: LAUNCH: Browser (one tab, after short delay for server startup)
:: ==================================================================
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://127.0.0.1:%PORT%"

:: ==================================================================
:: LAUNCH: Dashboard (FOREGROUND — you see all output and errors)
:: ==================================================================
echo  Starting dashboard on http://127.0.0.1:%PORT%
echo  Browser will open automatically when server is ready.
echo.
echo  Close this window or press Ctrl+C to stop everything.
echo  ============================================================
echo.

:: This runs in the foreground — you see everything
"%PYTHON_EXE%" react_dashboard2.py

:: ==================================================================
:: CLEANUP: Dashboard exited, kill discovery
:: ==================================================================
echo.
echo  Shutting down...
taskkill /fi "WINDOWTITLE eq TradingLab_Discovery" /f >nul 2>&1
echo  [OK]   Stopped.
echo.
echo [%date% %time%] TradingLab stopped >> "%LOGFILE%"
timeout /t 3 /nobreak >nul
endlocal
exit /b 0

:fatal
echo.
echo  ============================================================
echo   LAUNCH ABORTED
echo  ============================================================
pause
endlocal
exit /b 1