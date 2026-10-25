@echo off
title CERIP FastAPI Backend
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONUTF8=1
echo ====================================================
echo Starting CERIP FastAPI Backend...
echo Interactive Swagger Docs: http://localhost:8000/docs
echo ====================================================
python -m uvicorn src.api.main:app --reload --port 8000
pause
