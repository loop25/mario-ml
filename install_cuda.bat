@echo off
echo Installing PyTorch with CUDA 12.4 support...
C:\Projects\mario-ml\venv\Scripts\pip.exe install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.
echo Done! Running GPU check...
C:\Projects\mario-ml\venv\Scripts\python.exe C:\Projects\mario-ml\check_gpu.py
pause
