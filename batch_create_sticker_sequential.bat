@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage:
    echo   Drag and drop files here.
    pause
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found.
    pause
    exit /b 1
)

echo Processing sequential batch...

set "ARGS="
:collect
if "%~1"=="" goto run
set "ARGS=!ARGS! "%~1""
shift
goto collect

:run
python batch_create_sticker.py !ARGS!

echo.
echo Done.
pause

