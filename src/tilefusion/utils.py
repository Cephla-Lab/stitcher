"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.
"""

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift as cp_shift
    from cucim.skimage.exposure import match_histograms
    from cucim.skimage.measure import block_reduce
    from cucim.skimage.registration import phase_cross_correlation
    from opm_processing.imageprocessing.ssim_cuda import (
        structural_similarity_cupy_sep_shared as ssim_cuda,
    )

    xp = cp
    USING_GPU = True
except Exception:
    cp = None
    cp_shift = None
    from skimage.exposure import match_histograms
    from skimage.measure import block_reduce
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as _shift_cpu
    from skimage.metrics import structural_similarity as _ssim_cpu

    xp = np
    USING_GPU = False


def shift_array(arr, shift_vec):
    """Shift array using GPU if available, else CPU fallback."""
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """SSIM wrapper that routes to GPU kernel or CPU skimage."""
    if USING_GPU and "ssim_cuda" in globals():
        return float(ssim_cuda(arr1, arr2, win_size=win_size))
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def make_1d_profile(length: int, blend: int, bias: float = 0.5) -> np.ndarray:
    """
    Create a ramp profile over `blend` pixels at each end.

    Parameters
    ----------
    length : int
        Number of pixels.
    blend : int
        Ramp width.
    bias : float
        Blend bias from 0.0 to 1.0 (default 0.5).
        - 0.5: symmetric linear blend (50/50 at overlap center)
        - > 0.5: favor left/top tiles (e.g., 0.7 gives ~70% from left tile)
        - < 0.5: favor right/bottom tiles (e.g., 0.3 gives ~70% from right tile)

    Returns
    -------
    prof : (length,) float32
        Blend profile.
    """
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        # Linear position from 0 to 1 across the blend region
        t = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)

        # Use power-law ramps for asymmetric blending
        # bias=0.5 gives linear ramps (power=1), symmetric profile
        # bias>0.5 makes left ramp rise faster, right ramp fall slower (favor left tiles)
        # bias<0.5 makes left ramp rise slower, right ramp fall faster (favor right tiles)
        p_rise = max(0.1, bias * 2)  # Power for rising ramp (left edge)
        p_fall = max(0.1, (1 - bias) * 2)  # Power for falling ramp (right edge)

        # Left edge: rising ramp with power p_rise (0 -> ~1)
        prof[:blend] = t**p_rise
        # Right edge: falling ramp with power p_fall (~1 -> 0), using reversed t
        prof[-blend:] = t[::-1] ** p_fall
    return prof


def to_numpy(arr):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if USING_GPU and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Move array to current device (GPU if available, else CPU)."""
    return xp.asarray(arr)
