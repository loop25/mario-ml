"""
GPU Setup Helper for Super Mario Bros ML.

This script checks your GPU compatibility and installs the correct
PyTorch version for CUDA support.

RTX 50-series (Blackwell/sm_120) GPUs need PyTorch nightly with
CUDA 12.8+ (cu128). The standard PyTorch release only supports up
to sm_90 (RTX 40-series).

Usage:
    python setup_gpu.py          # Check GPU status
    python setup_gpu.py --install # Install PyTorch nightly cu128
"""

import subprocess
import sys
import os


def check_nvidia_driver():
    """Check if NVIDIA driver is installed and get version."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,name,compute_cap',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    driver, name, compute = parts[0], parts[1], parts[2]
                    print(f'  GPU: {name}')
                    print(f'  Driver: {driver}')
                    print(f'  Compute Capability: {compute}')

                    # Check driver version for Blackwell
                    major = int(driver.split('.')[0])
                    if major < 572:
                        print(f'  WARNING: Driver {driver} may be too old.')
                        print(f'  RTX 50-series needs driver 572.xx or newer.')
                    return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_pytorch_cuda():
    """Check if current PyTorch installation supports CUDA."""
    try:
        import torch
        print(f'  PyTorch version: {torch.__version__}')
        print(f'  CUDA available: {torch.cuda.is_available()}')

        if torch.cuda.is_available():
            print(f'  CUDA version: {torch.version.cuda}')
            gpu_name = torch.cuda.get_device_name(0)
            print(f'  GPU detected: {gpu_name}')

            # Try actual computation
            try:
                a = torch.randn(4, 4, device='cuda')
                _ = a @ a.T
                del a
                torch.cuda.empty_cache()
                print(f'  GPU computation: WORKING!')
                return True
            except RuntimeError as e:
                print(f'  GPU computation: FAILED')
                print(f'  Error: {e}')
                if 'sm_120' in str(e) or 'no kernel image' in str(e):
                    print(f'\n  Your GPU uses sm_120 (Blackwell architecture).')
                    print(f'  Current PyTorch does not include sm_120 kernels.')
                    print(f'  Solution: Install PyTorch nightly with cu128.')
                return False
        else:
            cuda_ver = getattr(torch.version, 'cuda', 'None')
            print(f'  Built with CUDA: {cuda_ver}')
            if cuda_ver and cuda_ver != 'None':
                print(f'  CUDA is built-in but not detecting your GPU.')
                print(f'  Check NVIDIA driver installation.')
            else:
                print(f'  PyTorch was installed without CUDA support.')
            return False
    except ImportError:
        print('  PyTorch not installed!')
        return False


def install_pytorch_cu128():
    """Install PyTorch nightly with CUDA 12.8 support."""
    print('\nInstalling PyTorch nightly with CUDA 12.8 (cu128)...')
    print('This will replace your current PyTorch installation.\n')

    cmd = [
        sys.executable, '-m', 'pip', 'install', '--pre',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/nightly/cu128',
    ]

    print(f'Running: {" ".join(cmd)}\n')
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print('\nPyTorch nightly cu128 installed successfully!')
        print('Run this script again to verify GPU support.')
    else:
        print('\nInstallation failed. Try manually:')
        print('  pip install --pre torch torchvision torchaudio '
              '--index-url https://download.pytorch.org/whl/nightly/cu128')


def main():
    print('=' * 60)
    print('  Super Mario Bros ML - GPU Setup')
    print('=' * 60)

    # Check NVIDIA driver
    print('\n[1/2] Checking NVIDIA GPU driver...')
    has_driver = check_nvidia_driver()
    if not has_driver:
        print('  No NVIDIA GPU detected or nvidia-smi not found.')
        print('  Training will use CPU (which is fine for this project).')
        print('  Note: NEAT always runs on CPU regardless.')

    # Check PyTorch CUDA
    print('\n[2/2] Checking PyTorch CUDA support...')
    cuda_works = check_pytorch_cuda()

    # Summary
    print('\n' + '=' * 60)
    if cuda_works:
        print('  STATUS: GPU READY!')
        print('  PPO and DQN will automatically use your GPU.')
        print('  NEAT always uses CPU (tiny networks, no GPU benefit).')
    elif has_driver:
        print('  STATUS: GPU detected but CUDA not working.')
        print('\n  To enable GPU training, run:')
        print('    python setup_gpu.py --install')
        print('\n  Or manually:')
        print('    pip install --pre torch torchvision torchaudio '
              '--index-url https://download.pytorch.org/whl/nightly/cu128')
    else:
        print('  STATUS: No GPU — training will use CPU.')
        print('  This is fine! CPU training works well for all algorithms.')
    print('=' * 60)

    # Handle --install flag
    if '--install' in sys.argv:
        install_pytorch_cu128()


if __name__ == '__main__':
    main()
