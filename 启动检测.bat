@echo off
chcp 65001 >nul
title 四分支图像篡改检测系统
echo ============================================
echo        四分支图像篡改检测系统
echo ============================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python
    pause
    exit /b 1
)

:: 运行交互式检测脚本
python run_detection.py

exit /b 0
