"""Tensor-contraction backend: prefers ncon_torch (GPU/autodiff via PyTorch) when
installed, falls back to the numpy-only ncon package otherwise."""

try:
    from ncon_torch import ncon, permute
except ImportError:
    import numpy as np
    from ncon import ncon

    def permute(tensor: np.ndarray, perm: list[int]) -> np.ndarray:
        return np.transpose(tensor, perm) if tensor.ndim > 0 else tensor


__all__ = ["ncon", "permute"]
