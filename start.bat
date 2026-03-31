@echo off
chcp 65001 >nul
title Image Tampering Detection System
echo ============================================
echo    Four-Branch Image Tampering Detection
echo ============================================
echo.

:: Change to script directory
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [INFO] Starting detection system...
echo.

:: Run detection
python run_detection.py

if errorlevel 1 (
    echo.
    echo [ERROR] Detection failed. Check error messages above.
    pause
)

exit /b 0
