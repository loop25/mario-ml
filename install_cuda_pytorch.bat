@echo off
echo ============================================================
echo  Installing PyTorch with CUDA 12.8 (RTX 5070 GPU support)
echo  This may take several minutes (downloading ~2.5GB)...
echo ============================================================
echo.

echo Step 1: Installing CUDA-enabled PyTorch (nightly with cu128)...
echo The RTX 5070 (Blackwell/sm_120) needs CUDA 12.8+ for full support.
echo.
C:\Projects\mario-ml\venv\Scripts\pip.exe install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Nightly cu128 install failed. Trying stable cu124 as fallback...
    C:\Projects\mario-ml\venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo Step 2: Reinstalling stable-baselines3 (may have been removed)...
C:\Projects\mario-ml\venv\Scripts\pip.exe install stable-baselines3

echo.
echo Step 3: Verifying GPU support...
C:\Projects\mario-ml\venv\Scripts\python.exe C:\Projects\mario-ml\check_gpu.py

echo.
echo ============================================================
echo  Installation complete! You can close this window.
echo ============================================================
pause
