@echo off
chcp 65001 >nul
echo Installing dependencies for Sticker Creator...
echo.
echo This will install:
echo   - Pillow (image processing)
echo   - NumPy, SciPy (mathematical operations)
echo   - rembg (background removal)
echo   - onnxruntime (required by rembg)
echo   - requests (for optional Ollama integration)
echo.
echo Choose installation type:
echo   1. System-wide (default Python installation) - Recommended
echo   2. User-only (current user only)
echo.

set /p choice="Enter choice [1/2] (default: 1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    echo.
    echo Installing system-wide...
    python -m pip install -r requirements.txt
) else if "%choice%"=="2" (
    echo.
    echo Installing for current user only...
    python -m pip install --user -r requirements.txt
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Installation completed successfully!
    echo.
    echo You can now use the sticker creator scripts.
) else (
    echo.
    echo ✗ Installation failed. Please check the error messages above.
)

echo.
pause

