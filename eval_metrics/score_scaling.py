"""Confidence scaling helpers (full min–max and p1–p99)."""
from typing import Optional, Tuple

import numpy as np

PERCENTILE_CLIP_PCT = 1.0
_UNIT_HI = 0.9999
_UNIT_MID = 0.49995


def scale_to_unit_interval(
    s: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Clip to ``[lo, hi]`` and map to ``[0, 0.9999]``. Degenerate range → midpoint."""
    arr = np.asarray(s, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return arr
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo + 1e-15:
        return np.full_like(arr, _UNIT_MID)
    clipped = np.clip(flat, lo, hi)
    return np.interp(clipped, (lo, hi), (0.0, _UNIT_HI)).reshape(arr.shape)


def percentile_bounds(
    s: np.ndarray, clip_pct: float = PERCENTILE_CLIP_PCT
) -> Tuple[float, float]:
    """Two-sided percentile range on finite values (default p1–p99)."""
    flat = np.asarray(s, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, clip_pct))
    hi = float(np.percentile(finite, 100.0 - clip_pct))
    return lo, hi


def percentile_minmax_to_unit_interval(
    s: np.ndarray,
    *,
    clip_pct: float = PERCENTILE_CLIP_PCT,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    """Clip to p1–p99 (or given ``lo``/``hi``) and map to ``[0, 0.9999]``."""
    arr = np.asarray(s, dtype=np.float64)
    if lo is None or hi is None:
        lo, hi = percentile_bounds(arr, clip_pct)
    return scale_to_unit_interval(arr, lo, hi)


def minmax_confidence_to_unit_interval(s: np.ndarray) -> np.ndarray:
    """Map raw confidence (higher = more certain) to [0, 0.9999]; constant vector → midpoint."""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return s
    lo, hi = float(np.min(s)), float(np.max(s))
    return scale_to_unit_interval(s, lo, hi).reshape(-1)
