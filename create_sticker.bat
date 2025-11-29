@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Check if file is passed as argument
if "%~1"=="" (
    echo Usage:
    echo   1. Drag and drop one or more PNG files onto this .bat file
    echo   2. Or specify file path: create_sticker.bat "path\to\file.png"
    echo.
    echo Will create files with _sticker.png suffix
    echo.
    pause
    exit /b 1
)

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ========================================
    echo Error: Python is not found in PATH
    echo ========================================
    echo Please install Python or add it to your PATH
    echo.
    pause
    exit /b 1
)

set "FILE_COUNT=0"
set "SUCCESS_COUNT=0"
set "ERROR_COUNT=0"

:process_loop
if "%~1"=="" goto :summary

set /a FILE_COUNT+=1
set "CURRENT_FILE=%~1"
echo.
echo ========================================
echo Processing file !FILE_COUNT!: !CURRENT_FILE!
echo ========================================

REM Get the argument - handle truncation issues
set "INPUT_FILE=%~f1"
set "EXISTS=0"

if exist "!INPUT_FILE!" (
    set "EXISTS=1"
)

if !EXISTS! EQU 0 (
    REM Try with .png extension
    if exist "!INPUT_FILE!.png" (
        set "INPUT_FILE=!INPUT_FILE!.png"
        set "EXISTS=1"
    )
)

if !EXISTS! EQU 0 (
    REM Try wildcards for truncated filenames with ==
    for %%F in ("!INPUT_FILE!*.png") do (
        if !EXISTS! EQU 0 (
            set "INPUT_FILE=%%~fF"
            set "EXISTS=1"
        )
    )
)

if !EXISTS! EQU 0 (
    REM Try jpg/jpeg/webp
    for %%F in ("!INPUT_FILE!*.jpg") do (
        if !EXISTS! EQU 0 (
            set "INPUT_FILE=%%~fF"
            set "EXISTS=1"
        )
    )
    for %%F in ("!INPUT_FILE!*.jpeg") do (
        if !EXISTS! EQU 0 (
            set "INPUT_FILE=%%~fF"
            set "EXISTS=1"
        )
    )
    for %%F in ("!INPUT_FILE!*.webp") do (
        if !EXISTS! EQU 0 (
            set "INPUT_FILE=%%~fF"
            set "EXISTS=1"
        )
    )
)

if !EXISTS! EQU 0 (
    echo Error: File not found: !CURRENT_FILE!
    echo Windows might have truncated the path because it contains == characters.
    echo.
    echo Solutions:
    echo   1. Rename the file to remove == characters, OR
    echo   2. Run from command line with full quoted path
    set /a ERROR_COUNT+=1
    shift
    goto :process_loop
)

REM Extract directory, filename without extension, and extension
for %%F in ("!INPUT_FILE!") do (
    set "DIR_PATH=%%~dpF"
    set "NAME_NO_EXT=%%~nF"
    set "EXT=%%~xF"
)

REM Create output filename
set "OUTPUT_FILE=!DIR_PATH!!NAME_NO_EXT!_sticker!EXT!"

echo Input file: !INPUT_FILE!
echo Output file: !OUTPUT_FILE!

REM Run Python script
python create_sticker.py "!INPUT_FILE!" "!OUTPUT_FILE!" --width 10
set "PYTHON_EXIT=!ERRORLEVEL!"

if !PYTHON_EXIT! EQU 0 (
    echo ✓ Success!
    set /a SUCCESS_COUNT+=1
) else (
    echo ✗ Error creating sticker. Exit code: !PYTHON_EXIT!
    set /a ERROR_COUNT+=1
)

shift
goto :process_loop

:summary
echo.
echo ========================================
echo SUMMARY
echo ========================================
echo Total files processed: !FILE_COUNT!
echo Successful: !SUCCESS_COUNT!
echo Errors: !ERROR_COUNT!
echo ========================================
echo.
echo Press any key to close this window...
pause
