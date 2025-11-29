@echo off
chcp 65001 >nul
echo ========================================
echo GitHub Repository Setup
echo ========================================
echo.
echo This script will help you connect your local repository to GitHub.
echo.
echo STEP 1: Create a new repository on GitHub
echo   1. Go to: https://github.com/new
echo   2. Choose a repository name (e.g., "sticker-creator")
echo   3. Choose Public or Private
echo   4. DO NOT initialize with README, .gitignore, or license
echo   5. Click "Create repository"
echo.
echo STEP 2: After creating, come back here and press any key
echo.
pause
echo.
echo ========================================
echo STEP 3: Enter your GitHub repository URL
echo ========================================
echo.
echo Example: https://github.com/YOUR_USERNAME/sticker-creator.git
echo Or SSH: git@github.com:YOUR_USERNAME/sticker-creator.git
echo.
set /p repo_url="Enter repository URL: "

if "%repo_url%"=="" (
    echo Error: Repository URL is required
    pause
    exit /b 1
)

echo.
echo Adding remote repository...
git remote add origin "%repo_url%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Warning: Remote might already exist. Trying to update...
    git remote set-url origin "%repo_url%"
)

echo.
echo Renaming branch to 'main'...
git branch -M main

echo.
echo Pushing to GitHub...
echo (You may be prompted for GitHub credentials)
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✓ Success! Repository uploaded to GitHub
    echo ========================================
    echo.
    echo Your repository is now available at:
    echo %repo_url%
) else (
    echo.
    echo ========================================
    echo ✗ Error pushing to GitHub
    echo ========================================
    echo.
    echo Possible issues:
    echo   - Authentication required (use GitHub token or SSH)
    echo   - Repository URL is incorrect
    echo   - Network connection problem
    echo.
    echo For authentication, you may need to:
    echo   1. Use GitHub Personal Access Token
    echo   2. Or set up SSH keys
    echo   3. Or use GitHub Desktop
)

echo.
pause

