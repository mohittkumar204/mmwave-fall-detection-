@echo off
title TI mmWave Fall Detection Visualizer
cd /d "%~dp0"
echo =======================================================
echo Starting TI mmWave People Counting & Fall Detection GUI
echo =======================================================
python run_gui.py
if %errorlevel% neq 0 (
    echo.
    echo GUI exited with error code %errorlevel%.
    pause
)
