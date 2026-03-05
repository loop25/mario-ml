"""Quick GPU check script."""
import sys
import warnings
warnings.filterwarnings("ignore")  # Suppress the sm_120 warning for cleaner output

print(f"Python: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Test if CUDA operations actually work
        try:
            x = torch.randn(3, 3).cuda()
            y = x @ x.T
            print(f"CUDA operations: WORKING (GPU will be used for training)")
        except RuntimeError as e:
            print(f"CUDA operations: FAILED ({e})")
            print(f"  Your GPU may need a newer PyTorch nightly build.")
            print(f"  Run: install_cuda_pytorch.bat to try cu128 nightly")
            print(f"  Training will fall back to CPU until this is resolved.")
    else:
        print("NO CUDA - PyTorch was installed without GPU support")
        print("To fix, run: install_cuda_pytorch.bat")
except ImportError:
    print("PyTorch not installed")

try:
    import stable_baselines3
    print(f"stable-baselines3 version: {stable_baselines3.__version__}")
except ImportError:
    print("stable-baselines3 not installed")
