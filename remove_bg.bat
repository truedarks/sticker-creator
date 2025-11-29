@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Check if file is passed as argument (drag and drop on .bat file)
if "%~1"=="" (
    echo Usage:
    echo   1. Drag and drop one or more PNG files onto this .bat file
    echo   2. Or specify file path: remove_bg.bat "path\to\file.png"
    echo.
    echo Will create files with _nobg.png suffix
    echo.
    pause
    exit /b 1
)

REM Check if Python is available first
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

REM Count files and process each one
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

REM Get the argument - Windows truncates paths with ==, so we need special handling
set "RAW_ARG=!CURRENT_FILE!"
set "INPUT_FILE=%~f1"

REM Check if file exists, if not try to find it by partial name match
if not exist "!INPUT_FILE!" (
    REM Try with .png extension first
    if exist "!INPUT_FILE!.png" (
        set "INPUT_FILE=!INPUT_FILE!.png"
    ) else (
        REM Try to find file by searching for files starting with the partial name
        REM This handles the case where == was truncated
        set "FOUND_FILE="
        for %%F in ("!INPUT_FILE!*.png") do (
            set "INPUT_FILE=%%~fF"
            set "FOUND_FILE=1"
            goto :file_found
        )
        for %%F in ("!INPUT_FILE!*.jpg") do (
            set "INPUT_FILE=%%~fF"
            set "FOUND_FILE=1"
            goto :file_found
        )
        for %%F in ("!INPUT_FILE!*.jpeg") do (
            set "INPUT_FILE=%%~fF"
            set "FOUND_FILE=1"
            goto :file_found
        )
        for %%F in ("!INPUT_FILE!*.webp") do (
            set "INPUT_FILE=%%~fF"
            set "FOUND_FILE=1"
            goto :file_found
        )
        
        :file_found
        if "!FOUND_FILE!"=="" (
            echo Error: File not found: !RAW_ARG!
            echo Windows truncated the path because it contains == characters.
            echo Solutions:
            echo   1. Rename the file to remove == characters, OR
            echo   2. Run from command line with full quoted path
            set /a ERROR_COUNT+=1
            shift
            goto :process_loop
        )
    )
)

REM Extract directory, filename without extension, and extension
for %%F in ("!INPUT_FILE!") do (
    set "DIR_PATH=%%~dpF"
    set "NAME_NO_EXT=%%~nF"
    set "EXT=%%~xF"
)

REM Create output filename
set "OUTPUT_FILE=!DIR_PATH!!NAME_NO_EXT!_nobg!EXT!"

echo Input file: !INPUT_FILE!
echo Output file: !OUTPUT_FILE!

REM Run Python script
python remove_bg.py "!INPUT_FILE!" "!OUTPUT_FILE!"
set "PYTHON_EXIT=!ERRORLEVEL!"

if !PYTHON_EXIT! EQU 0 (
    echo ✓ Success!
    set /a SUCCESS_COUNT+=1
) else (
    echo ✗ Error removing background. Exit code: !PYTHON_EXIT!
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

