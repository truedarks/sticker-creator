@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Check if any files are passed as arguments (drag and drop on .bat file)
if "%~1"=="" (
    echo Usage:
    echo   1. Drag and drop one or more image files onto this .bat file
    echo   2. Or specify file paths: batch_remove_bg_parallel.bat "file1.png" "file2.png" ...
    echo.
    echo This will process images in PARALLEL using all CPU cores (except one).
    echo Output files will have _nobg suffix.
    pause
    exit /b 1
)

echo ========================================
echo Batch Background Removal - Parallel
echo ========================================
echo.
echo Processing images in parallel using multiple CPU cores...
echo.

REM Collect all file arguments
set "FILE_COUNT=0"
set "ARGS="

:collect_files
if "%~1"=="" goto process_files
set /a FILE_COUNT+=1
set "ARGS=!ARGS! "%~1""
shift
goto collect_files

:process_files
if !FILE_COUNT! EQU 0 (
    echo Error: No files specified
    pause
    exit /b 1
)

echo Found !FILE_COUNT! file(s) to process
echo.

REM Call Python script with all files and --parallel flag
python batch_remove_bg.py --parallel !ARGS!

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✓ Batch processing completed!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ✗ Some errors occurred during processing
    echo ========================================
)

echo.
pause

