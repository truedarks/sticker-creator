@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Use PowerShell for file selection dialog
echo Select PNG file to create sticker...
echo.

for /f "delims=" %%i in ('powershell -Command "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; $dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Filter = 'PNG files (*.png)|*.png|All files (*.*)|*.*'; $dialog.Title = 'Select PNG file'; if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }"') do set "INPUT_FILE=%%i"

if "!INPUT_FILE!"=="" (
    echo File not selected. Cancelled.
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

echo.
echo Input file: !INPUT_FILE!
echo Output file: !OUTPUT_FILE!
echo.

python create_sticker.py "!INPUT_FILE!" "!OUTPUT_FILE!" --width 20

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Done! Sticker created: !OUTPUT_FILE!
    echo.
    REM Open folder with result
    explorer /select,"!OUTPUT_FILE!"
) else (
    echo.
    echo ✗ Error creating sticker
)

echo.
pause

