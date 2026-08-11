"""Grayscale image from the PPI score vector — outer absolute-difference matrix.

M_ij = |v_i - v_j|, min-max normalized to [0,255]. NO dilate, NO enhance.
"""
from __future__ import annotations

import numpy as np
import torch


def outer_diff_image(vec: np.ndarray) -> np.ndarray:
    """Return an (L x L) uint8 grayscale image, L = len(vec). Symmetric, zero diagonal."""
    v = vec.astype(np.float32, copy=False)
    M = np.abs(np.subtract.outer(v, v))
    mn, mx = float(M.min()), float(M.max())
    if mx - mn < 1e-8:
        Mz = np.zeros_like(M, dtype=np.float32)
    else:
        Mz = (M - mn) / (mx - mn + 1e-8)
    return (Mz * 255.0).astype(np.uint8)


def image_tensor_from_vec(vec: np.ndarray) -> torch.Tensor:
    """(L,) score vector -> (3, L, L) float tensor, Normalize(0.5, 0.5), 3 channels.

    The image side length equals len(vec); call make_vector with out_size=image_size
    so the outer-diff matrix already has the desired spatial size (no resize needed).
    """
    img = outer_diff_image(vec)                      # (L, L) uint8
    t = torch.from_numpy(img).float() / 255.0        # [0,1]
    t = (t - 0.5) / 0.5                              # Normalize(0.5, 0.5)
    t = t.unsqueeze(0).repeat(3, 1, 1).contiguous()  # (3, L, L)
    return t
