@echo off
title CERIP Web Dashboard
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONUTF8=1
echo ====================================================
echo Starting CERIP Streamlit Web Dashboard...
echo URL: http://localhost:8501
echo ====================================================
python -m streamlit run dashboard/app.py
pause
