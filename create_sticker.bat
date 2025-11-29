@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Check if file is passed as argument (drag and drop on .bat file)
if "%~1"=="" (
    echo Usage:
    echo   1. Drag and drop PNG file onto this .bat file
    echo   2. Or specify file path: create_sticker.bat "path\to\file.png"
    echo.
    echo Will create file with _sticker.png suffix
    pause
    exit /b 1
)

set "INPUT_FILE=%~1"

REM Check if file exists
if not exist "!INPUT_FILE!" (
    echo Error: File not found: !INPUT_FILE!
    pause
    exit /b 1
)

REM Extract directory, filename without extension, and extension
for %%F in ("!INPUT_FILE!") do (
    set "DIR_PATH=%%~dpF"
    set "NAME_NO_EXT=%%~nF"
    set "EXT=%%~xF"
)

REM Create output filename
set "OUTPUT_FILE=!DIR_PATH!!NAME_NO_EXT!_sticker!EXT!"

echo Creating sticker...
echo Input file: !INPUT_FILE!
echo Output file: !OUTPUT_FILE!
echo.

python create_sticker.py "!INPUT_FILE!" "!OUTPUT_FILE!" --width 10

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Done! Sticker created: !OUTPUT_FILE!
) else (
    echo.
    echo ✗ Error creating sticker
)

echo.
pause

