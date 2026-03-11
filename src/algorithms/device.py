"""Centralized GPU/device auto-detection for all trainers.

Provides a single `select_device()` function that all trainers share,
eliminating code duplication and ensuring consistent behavior.

Device priority order:
    1. CUDA (NVIDIA GPUs) — most common for ML training
    2. MPS (Apple Silicon) — Metal Performance Shaders on M1/M2/M3/M4 Macs
    3. CPU — universal fallback

A smoke test is run for CUDA to catch GPUs that report availability
but fail on actual kernel execution (e.g., RTX 50-series Blackwell
GPUs with older CUDA toolkit versions).

Usage:
    from src.algorithms.device import select_device

    # Auto-detect best available device
    device = select_device()

    # Force a specific device
    device = select_device(preference='cpu')

    # Get info without creating a device
    info = get_device_info()
"""

from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _cuda_smoke_test() -> bool:
    """Verify CUDA kernels actually execute on this GPU.

    Some GPUs (e.g. RTX 5070 Blackwell/sm_120) report
    torch.cuda.is_available() == True but crash on kernel
    execution with older CUDA toolkit versions.

    Returns True if CUDA kernels work, False otherwise.
    """
    try:
        a = torch.randn(4, 4, device='cuda')
        _ = a @ a.T
        del a
        torch.cuda.empty_cache()
        return True
    except (RuntimeError, AssertionError, Exception):
        return False


def _mps_smoke_test() -> bool:
    """Verify MPS (Apple Metal) backend works.

    Returns True if MPS is functional, False otherwise.
    """
    try:
        a = torch.randn(4, 4, device='mps')
        _ = a @ a.T
        del a
        return True
    except (RuntimeError, Exception):
        return False


def get_device_info() -> dict:
    """Get information about available compute devices.

    Returns a dict with:
        - 'selected': str — the device that would be selected ('cuda', 'mps', 'cpu')
        - 'cuda_available': bool — whether CUDA is available and functional
        - 'mps_available': bool — whether MPS is available and functional
        - 'gpu_name': str or None — GPU name if CUDA is available
        - 'vram_gb': float or None — VRAM in GB if CUDA is available
        - 'torch_version': str — PyTorch version
        - 'cuda_version': str or None — CUDA toolkit version
    """
    if not HAS_TORCH:
        return {
            'selected': 'cpu',
            'cuda_available': False,
            'mps_available': False,
            'gpu_name': None,
            'vram_gb': None,
            'torch_version': None,
            'cuda_version': None,
        }

    info = {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        'gpu_name': None,
        'vram_gb': None,
        'cuda_available': False,
        'mps_available': False,
        'selected': 'cpu',
    }

    # Check CUDA
    if torch.cuda.is_available() and _cuda_smoke_test():
        info['cuda_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        info['vram_gb'] = round(vram / 1024**3, 1)
        info['selected'] = 'cuda'
    # Check MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if _mps_smoke_test():
            info['mps_available'] = True
            info['selected'] = 'mps'

    return info


def select_device(
    preference: Optional[str] = None,
    algo_name: str = '',
    verbose: bool = True,
) -> 'torch.device':
    """Select the best available compute device.

    Args:
        preference: Force a specific device. One of:
            - 'auto' or None: auto-detect best device (default)
            - 'cuda': force CUDA (error if unavailable)
            - 'mps': force MPS (error if unavailable)
            - 'cpu': force CPU
        algo_name: Algorithm name for log messages (e.g., 'DQN', 'PPO').
        verbose: Whether to print device selection info.

    Returns:
        torch.device: The selected device.

    Raises:
        RuntimeError: If a forced device is not available.
        ImportError: If PyTorch is not installed.
    """
    if not HAS_TORCH:
        raise ImportError(
            'PyTorch is required for GPU training. '
            'Install: pip install torch torchvision'
        )

    prefix = f'{algo_name} using' if algo_name else 'Using'

    # Handle explicit preference
    if preference and preference != 'auto':
        preference = preference.lower()
        if preference == 'cpu':
            if verbose:
                print(f'{prefix} device: cpu (forced)')
            return torch.device('cpu')

        if preference == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    'CUDA requested but not available. '
                    'Install CUDA-enabled PyTorch: '
                    'pip install torch torchvision --index-url '
                    'https://download.pytorch.org/whl/cu128'
                )
            if not _cuda_smoke_test():
                raise RuntimeError(
                    'CUDA is available but kernels fail to execute. '
                    'Your GPU may need a newer CUDA toolkit. '
                    'Install: pip install torch torchvision '
                    '--index-url https://download.pytorch.org/whl/cu128'
                )
            gpu = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_mem / 1024**3, 1
            )
            if verbose:
                print(f'{prefix} device: cuda ({gpu}, {vram_gb}GB VRAM)')
            return torch.device('cuda')

        if preference == 'mps':
            if not (hasattr(torch.backends, 'mps')
                    and torch.backends.mps.is_available()):
                raise RuntimeError(
                    'MPS requested but not available. '
                    'MPS requires macOS 12.3+ and Apple Silicon.'
                )
            if verbose:
                print(f'{prefix} device: mps (Apple Silicon)')
            return torch.device('mps')

        raise ValueError(
            f"Unknown device preference '{preference}'. "
            f"Use 'auto', 'cuda', 'mps', or 'cpu'."
        )

    # Auto-detect: CUDA → MPS → CPU
    info = get_device_info()

    if info['cuda_available']:
        if verbose:
            print(
                f"{prefix} device: cuda "
                f"({info['gpu_name']}, {info['vram_gb']}GB VRAM)"
            )
        return torch.device('cuda')

    if info['mps_available']:
        if verbose:
            print(f'{prefix} device: mps (Apple Silicon)')
        return torch.device('mps')

    # CPU fallback
    if verbose:
        print(f'{prefix} device: cpu')
        if torch.cuda.is_available():
            # CUDA reports available but smoke test failed
            print('  Note: CUDA detected but kernels failed. '
                  'Your GPU may need a newer CUDA toolkit.')
        print('  Tip: Install CUDA-enabled PyTorch for GPU acceleration:')
        print('  pip install torch torchvision '
              '--index-url https://download.pytorch.org/whl/cu128')
    return torch.device('cpu')
