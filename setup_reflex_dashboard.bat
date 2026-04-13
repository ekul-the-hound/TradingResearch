@echo off
:: ==============================================================================
:: TradingLab.bat — Hardened Production Launcher
:: ==============================================================================
:: One-click: validates everything, starts discovery + dashboard, opens browser.
:: Logs all output. Kills background processes cleanly via PID on exit.
:: ==============================================================================

setlocal

set "PROJECT_DIR=D:\Luke Files\Coding\Developer\TradingResearch"
set "VENV_DIR=D:\Luke Files\Coding\Developer\TradingResearch\.venv"
set "PORT=8080"
set "LOGFILE=%PROJECT_DIR%\TradingLab_log.txt"
set "DISC_LOG=%PROJECT_DIR%\discovery_log.txt"
set "DISC_PID_FILE=%PROJECT_DIR%\discovery.pid"
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

echo [%date% %time%] === Launcher started === > "%LOGFILE%" 2>nul

:: ==================================================================
:: CHECK 1: Project directory
:: ==================================================================
if not exist "%PROJECT_DIR%\" (
    echo  [FAIL] Directory not found: %PROJECT_DIR%
    goto :fatal
)
cd /d "%PROJECT_DIR%"
echo  [OK]   Project: %CD%
echo [%date% %time%] Project dir: %CD% >> "%LOGFILE%"

:: ==================================================================
:: CHECK 2: Python
:: ==================================================================

:: If venv exists, use its python directly (no ambiguity)
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
    echo  [OK]   Using venv Python
    echo [%date% %time%] Using venv: %VENV_DIR%\Scripts\python.exe >> "%LOGFILE%"

    :: Also activate so pip/scripts work
    call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
) else (
    :: No venv — check system python
    where python >nul 2>&1
    if errorlevel 1 (
        echo  [FAIL] Python not found. No venv at %VENV_DIR% and python not in PATH.
        goto :fatal
    )
    echo  [WARN] No venv found. Using system Python.
    echo [%date% %time%] WARNING: No venv, using system python >> "%LOGFILE%"
)

:: Print version
for /f "tokens=*" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PYVER=%%v"
echo         %PYVER%
echo [%date% %time%] %PYVER% >> "%LOGFILE%"

:: ==================================================================
:: CHECK 3: Required files
:: ==================================================================
if not exist "%PROJECT_DIR%\react_dashboard2.py" (
    echo  [FAIL] react_dashboard2.py not found
    goto :fatal
)
echo  [OK]   react_dashboard2.py

set "DISC_OK=0"
if exist "%PROJECT_DIR%\run_discovery.py" (
    echo  [OK]   run_discovery.py
    set "DISC_OK=1"
) else (
    echo  [WARN] run_discovery.py missing — discovery skipped
)

if exist "%PROJECT_DIR%\strategy_inbox.py" (
    echo  [OK]   strategy_inbox.py
) else (
    echo  [WARN] strategy_inbox.py missing — manual entry disabled
)

:: ==================================================================
:: CHECK 4: Key Python packages
:: ==================================================================
echo.
echo  --- Dependencies ---

"%PYTHON_EXE%" -c "import reactpy; print('  [OK]   reactpy ' + getattr(reactpy, '__version__', ''))" 2>>"%LOGFILE%"
if errorlevel 1 (
    echo  [FAIL] reactpy not installed
    echo         Run: pip install "reactpy[fastapi]"
    echo [%date% %time%] FAIL: reactpy missing >> "%LOGFILE%"
    goto :fatal
)

"%PYTHON_EXE%" -c "import uvicorn; print('  [OK]   uvicorn')" 2>>"%LOGFILE%"
if errorlevel 1 (
    echo  [FAIL] uvicorn not installed
    echo         Run: pip install uvicorn
    goto :fatal
)

"%PYTHON_EXE%" -c "import plotly; print('  [OK]   plotly ' + plotly.__version__)" 2>>"%LOGFILE%"
if errorlevel 1 echo  [WARN] plotly not installed — charts disabled

"%PYTHON_EXE%" -c "import numpy; print('  [OK]   numpy ' + numpy.__version__)" 2>>"%LOGFILE%"
if errorlevel 1 echo  [WARN] numpy not installed

"%PYTHON_EXE%" -c "import pandas" 2>nul
if errorlevel 1 ( echo  [WARN] pandas not installed ) else ( echo  [OK]   pandas )

:: ==================================================================
:: CHECK 5: Port
:: ==================================================================
echo.
echo  --- Network ---

set "PORT_BLOCKED=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    set "BLOCKING_PID=%%a"
    set "PORT_BLOCKED=1"
)

if "%PORT_BLOCKED%"=="1" (
    echo  [WARN] Port %PORT% in use by PID %BLOCKING_PID%
    echo         To free it: taskkill /f /pid %BLOCKING_PID%
    echo         Or change PORT at top of this file.
    echo [%date% %time%] WARNING: Port %PORT% blocked by PID %BLOCKING_PID% >> "%LOGFILE%"
) else (
    echo  [OK]   Port %PORT% available
)

:: ==================================================================
:: CHECK 6: Optional services
:: ==================================================================

docker ps >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Docker not running
) else (
    echo  [OK]   Docker running
    docker ps --format "{{.Names}}" 2>nul | findstr /i "searxng" >nul 2>&1
    if errorlevel 1 (
        echo  [INFO] SearXNG container not running
    ) else (
        echo  [OK]   SearXNG container up
    )
)

:: Ollama check (portable — no curl dependency)
"%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2); print('  [OK]   Ollama responding')" 2>nul
if errorlevel 1 echo  [INFO] Ollama not responding

:: ==================================================================
:: LAUNCH: Discovery (background with PID capture + logging)
:: ==================================================================
echo.
echo  ============================================================
echo   Launching...
echo  ============================================================
echo.

:: Clean old PID file
del "%DISC_PID_FILE%" >nul 2>&1

if "%DISC_OK%"=="1" (
    echo  Starting discovery runner...

    :: Start in background, capture its PID via a helper
    start "TradingLab_Discovery" /min cmd /c ""%PYTHON_EXE%" "%PROJECT_DIR%\run_discovery.py" --continuous --interval 3600 >> "%DISC_LOG%" 2>&1"

    :: Give it a moment to start, then grab the PID
    timeout /t 1 /nobreak >nul
    for /f "tokens=2" %%p in ('tasklist /fi "WINDOWTITLE eq TradingLab_Discovery" /fo list 2^>nul ^| findstr "PID:"') do (
        echo %%p > "%DISC_PID_FILE%"
        echo  [OK]   Discovery running ^(PID %%p^), logging to discovery_log.txt
        echo [%date% %time%] Discovery PID: %%p >> "%LOGFILE%"
    )
    if not exist "%DISC_PID_FILE%" (
        echo  [WARN] Discovery started but could not capture PID
    )
) else (
    echo  [SKIP] Discovery runner
)

:: ==================================================================
:: LAUNCH: Dashboard (foreground)
:: ==================================================================
echo.
echo  Starting dashboard on http://127.0.0.1:%PORT%
echo  Press Ctrl+C to stop everything.
echo  ============================================================
echo.
echo [%date% %time%] Starting dashboard >> "%LOGFILE%"

:: Start dashboard in background briefly so we can poll the port
start "TradingLab_Dashboard" /min cmd /c ""%PYTHON_EXE%" "%PROJECT_DIR%\react_dashboard2.py" >> "%LOGFILE%" 2>&1"

:: ==================================================================
:: WAIT: Poll until server is live (max 20 seconds)
:: ==================================================================
set "READY=0"
for /l %%i in (1,1,20) do (
    if "!READY!"=="0" (
        timeout /t 1 /nobreak >nul
        netstat -ano 2>nul | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
        if not errorlevel 1 (
            set "READY=1"
        )
    )
)

:: Need delayed expansion just for this check
setlocal enabledelayedexpansion
if "!READY!"=="1" (
    echo  [OK]   Server is live on port %PORT%
    start "" http://127.0.0.1:%PORT%
    echo  [OK]   Browser opened
    echo [%date% %time%] Server live, browser opened >> "%LOGFILE%"
) else (
    echo  [WARN] Server did not respond within 20 seconds.
    echo         Opening browser anyway...
    start "" http://127.0.0.1:%PORT%
    echo [%date% %time%] WARNING: Server not confirmed live >> "%LOGFILE%"
)
endlocal

:: ==================================================================
:: HOLD: Keep this window open while dashboard runs
:: ==================================================================
echo.
echo  ============================================================
echo   TradingLab is running.
echo   Dashboard: http://127.0.0.1:%PORT%
echo   Discovery log: discovery_log.txt
echo   System log: TradingLab_log.txt
echo  ============================================================
echo.
echo  Press any key to STOP TradingLab...
pause >nul

:: ==================================================================
:: CLEANUP: Kill everything by PID
:: ==================================================================
echo.
echo  Shutting down...

:: Kill dashboard
taskkill /fi "WINDOWTITLE eq TradingLab_Dashboard" /f >nul 2>&1
echo  [OK]   Dashboard stopped

:: Kill discovery by saved PID
if exist "%DISC_PID_FILE%" (
    set /p DISC_PID=<"%DISC_PID_FILE%"
    taskkill /pid %DISC_PID% /f >nul 2>&1
    taskkill /fi "WINDOWTITLE eq TradingLab_Discovery" /f >nul 2>&1
    del "%DISC_PID_FILE%" >nul 2>&1
    echo  [OK]   Discovery stopped
) else (
    taskkill /fi "WINDOWTITLE eq TradingLab_Discovery" /f >nul 2>&1
    echo  [OK]   Discovery cleanup done
)

:: Kill any remaining python processes with our scripts (safety net)
taskkill /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq TradingLab*" /f >nul 2>&1

echo.
echo [%date% %time%] === TradingLab stopped === >> "%LOGFILE%"
echo  TradingLab stopped cleanly.
echo  Logs: TradingLab_log.txt, discovery_log.txt
echo.
timeout /t 3 /nobreak >nul
endlocal
exit /b 0

:: ==================================================================
:: FATAL
:: ==================================================================
:fatal
echo.
echo  ============================================================
echo   LAUNCH ABORTED — fix the errors above and try again.
echo  ============================================================
echo [%date% %time%] FATAL: Aborted >> "%LOGFILE%" 2>nul
pause
endlocal
exit /b 1