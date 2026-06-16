@echo off
setlocal

pushd "%~dp0" >nul

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python was not found. Please install Python 3 and try again.
    pause
    popd >nul
    exit /b 1
)

%PYTHON_CMD% -c "import PySide6, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        popd >nul
        exit /b 1
    )
)

if /I "%~1"=="--check" (
    echo launcher check ok
    popd >nul
    exit /b 0
)

set "APP_CMD=%PYTHON_CMD%"
if /I "%PYTHON_CMD%"=="py -3" (
    where pyw >nul 2>&1
    if not errorlevel 1 set "APP_CMD=pyw -3"
) else (
    where pythonw >nul 2>&1
    if not errorlevel 1 set "APP_CMD=pythonw"
)

start "" %APP_CMD% "%CD%\main.py"
popd >nul
exit /b 0
