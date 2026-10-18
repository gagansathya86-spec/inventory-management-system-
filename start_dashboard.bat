@echo off
title Smart Inventory Management System - Web Dashboard
cd /d "%~dp0\product recognition"
echo ======================================================================
echo    Smart Inventory Management System Web Dashboard
echo ======================================================================
echo Tab 1: Camera Settings (Laptop Webcam ^& Phone Camera Stream/USB)
echo Tab 2: Product Recognition (17 Categories CNN ^& Barcode Scanner Mode)
echo Tab 3: Shelf ^& Layout Inspection (Real CNN Counting, Empty Shelf ^& Misplaced Detection)
echo Tab 4: Trend Analysis (LSTM Forecasting, 6 Written Insights ^& ROP Tables)
echo Utility: Supabase Cloud Database Inspector (Live Table Explorer)
echo ======================================================================
echo Dashboard URL: http://localhost:5000
echo ======================================================================
python run.py --mode web
pause
