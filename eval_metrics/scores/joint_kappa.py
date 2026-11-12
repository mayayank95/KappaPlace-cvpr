"""Joint vMF concentration utilities."""
import math
from typing import Optional

import numpy as np


def per_example_mean(values: np.ndarray) -> np.ndarray:
    """Mean over the feature axis. A ``(N,)`` vector is already one value per example."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    return np.mean(arr, axis=-1)


def _pair_kappa_and_cos(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return query κ, retrieved reference κ, and descriptor cosine."""
    kappa_q = per_example_mean(q_var)
    mean_db_var = per_example_mean(db_var)
    q_normed = q_desc / np.maximum(np.linalg.norm(q_desc, axis=1, keepdims=True), 1e-8)
    db_normed = db_desc / np.maximum(np.linalg.norm(db_desc, axis=1, keepdims=True), 1e-8)
    kappa_ref = mean_db_var[predictions]
    cos_theta = np.sum(q_normed[:, None, :] * db_normed[predictions], axis=2)
    return kappa_q[:, None], kappa_ref, cos_theta


def compute_joint_kappa(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
) -> np.ndarray:
    """Return top-k joint kappa scores for retrieved query/reference pairs."""
    kappa_q, kappa_ref, cos_theta = _pair_kappa_and_cos(
        predictions, q_var, db_var, q_desc, db_desc
    )
    return np.sqrt(
        np.maximum(
            kappa_q**2 + kappa_ref**2 + 2 * kappa_q * kappa_ref * cos_theta,
            0.0,
        )
    )


def compute_l2_joint_kappa(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
) -> np.ndarray:
    """Return S(Q,R) = κ_Q κ_R cos(θ) for each retrieved pair."""
    kappa_q, kappa_ref, cos_theta = _pair_kappa_and_cos(
        predictions, q_var, db_var, q_desc, db_desc
    )
    return kappa_q * kappa_ref * cos_theta


def log_partition_vmf(kappa: np.ndarray, d: int, eps: float = 1e-7) -> np.ndarray:
    """Amos approximation of log Z_d(κ), matching ``losses.vmf_loss.VMFLikelihood``.

    Z_d is the vMF partition function, so the density is exp(κ μ·z) / Z_d(κ).
    """
    dim = int(d)
    if dim < 2:
        raise ValueError(f"vMF dimension must be >= 2, got {dim}.")
    v = (dim - 1) / 2.0
    kappa_eps = np.asarray(kappa, dtype=np.float64) + eps
    y = np.sqrt(kappa_eps**2 + v**2)
    integral = y - v * np.log(v + y)
    log_z0 = math.log(2.0) + (dim / 2.0) * math.log(math.pi) - math.lgamma(dim / 2.0)
    integral_0 = v - v * math.log(2.0 * v)
    return integral + (log_z0 - integral_0)


def compute_mls(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
    *,
    d: Optional[int] = None,
) -> np.ndarray:
    """Mutual likelihood score of two r-vMF features (SCF, CVPR 2021).

    S_MLS = log Z_d(R) − log Z_d(κ_q) − log Z_d(κ_r), with
    R = ||κ_q μ_q + κ_r μ_r||. Descriptors are L2-normalized, so the sphere
    radius is 1 and the paper's −d log r term is zero. Higher means the two
    features are more likely to be the same point on the sphere.
    """
    kappa_q, kappa_ref, cos_theta = _pair_kappa_and_cos(
        predictions, q_var, db_var, q_desc, db_desc
    )
    resultant = np.sqrt(
        np.maximum(
            kappa_q**2 + kappa_ref**2 + 2.0 * kappa_q * kappa_ref * cos_theta,
            0.0,
        )
    )
    dim = int(q_desc.shape[-1] if d is None else d)
    return (
        log_partition_vmf(resultant, dim)
        - log_partition_vmf(kappa_q, dim)
        - log_partition_vmf(kappa_ref, dim)
    )
