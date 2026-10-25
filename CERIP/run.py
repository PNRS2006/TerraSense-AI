"""
CERIP — 1-Click Platform Launcher
Run: python run.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Fix Windows threading & encoding
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("  CERIP — Environmental Risk Intelligence Platform")
    print("=" * 60)
    print("  Select what you want to launch:")
    print("    [1] Interactive Web Dashboard (Streamlit - Recommended)")
    print("    [2] REST API Backend (FastAPI / Swagger Docs)")
    print("    [3] Run Both (API on :8000 + Dashboard on :8501)")
    print("    [4] Run Automated Unit Tests (pytest)")
    print("    [0] Exit")
    print("=" * 60)
    
    choice = input("Enter choice [1-4, default=1]: ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 Starting Streamlit Dashboard on http://localhost:8501 ...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")]
        subprocess.run(cmd, cwd=str(ROOT))
        
    elif choice == "2":
        print("\n🚀 Starting FastAPI Backend on http://localhost:8000 ...")
        print("📖 Swagger Docs: http://localhost:8000/docs")
        cmd = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"]
        subprocess.run(cmd, cwd=str(ROOT))
        
    elif choice == "3":
        print("\n🚀 Starting both API and Dashboard in parallel windows...")
        # Start API in a new terminal window
        os.system(f'start "CERIP FastAPI (:8000)" cmd /k "{sys.executable} -m uvicorn src.api.main:app --port 8000"')
        # Start Streamlit in current window
        print("📖 API running at http://localhost:8000/docs")
        print("🚀 Starting Streamlit Dashboard at http://localhost:8501 ...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")]
        subprocess.run(cmd, cwd=str(ROOT))
        
    elif choice == "4":
        print("\n🧪 Running full test suite...")
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        subprocess.run(cmd, cwd=str(ROOT))
        
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
