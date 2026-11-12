"""Uncertainty / confidence score definitions used by eval.

Learned pair scores and baseline extractors live here. Evaluation code
iterates the registries below so new scores are picked up consistently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from eval_metrics.scores.baselines import (
    SUE_NUM_NN,
    compute_l2_scores,
    compute_pa_scores,
    compute_sue_scores,
    skip_sue_dataset,
)
from eval_metrics.scores.joint_kappa import (
    compute_joint_kappa,
    compute_l2_joint_kappa,
    compute_mls,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learned pair scores (vMF): [N_q, pool], higher = more confident
# ---------------------------------------------------------------------------

LEARNED_PAIR_SCORE_FNS: Dict[str, Callable[..., np.ndarray]] = {
    "joint_kappa": compute_joint_kappa,
    "l2_joint_kappa": compute_l2_joint_kappa,
    "mls": compute_mls,
}

# Query-level ECE and rejection use top-1 of each learned pair score.
LEARNED_QUERY_SCORE_NAMES: Tuple[str, ...] = tuple(LEARNED_PAIR_SCORE_FNS)

# Match-level ECE / rejection use the full pair matrix (prefixed pairwise_*).
LEARNED_PAIRWISE_SCORE_NAMES: Tuple[str, ...] = tuple(
    f"pairwise_{name}" for name in LEARNED_PAIR_SCORE_FNS
)

# ---------------------------------------------------------------------------
# Baselines: query-level (higher = uncertain) + optional pairwise
# ---------------------------------------------------------------------------

BASELINE_QUERY_SCORE_NAMES: Tuple[str, ...] = ("l2", "pa", "sue", "sue_log")
BASELINE_PAIRWISE_SCORE_NAMES: Tuple[str, ...] = ("pairwise_l2",)

# SUE-log and κ, z-scored onto a shared uncertainty axis (higher = more uncertain).
HYBRID_QUERY_SCORE_NAMES: Tuple[str, ...] = ("sue_log_kappa",)

# Query-level eval order: kappa, learned pair top-1, baselines, then hybrids.
QUERY_EVAL_METHOD_ORDER: Tuple[str, ...] = (
    "kappa",
    *LEARNED_QUERY_SCORE_NAMES,
    *BASELINE_QUERY_SCORE_NAMES,
    *HYBRID_QUERY_SCORE_NAMES,
)

METHOD_LOG_NAMES: Dict[str, str] = {
    "kappa": "kappa",
    "mls": "MLS",
    "l2": "L2 distance",
    "pa": "PA-score",
    "sue": "SUE",
    "sue_log": "SUE-log",
    "sue_log_kappa": "SUE-log+κ",
}


@dataclass
class MethodScore:
    """One scoring method with query-level scores and optional pair matrix."""

    name: str
    log_name: str
    query_scores: np.ndarray  # per-query scalar ``[N]`` (mean-κ, L2, …)
    higher_is_confidence: bool
    uncertainty_loss: str
    pair_scores: Optional[np.ndarray] = None
    pairwise_name: Optional[str] = None
    pairwise_plot_filename: Optional[str] = None
    use_vmf_kappa_floor: bool = False
    percentile_two_sided: bool = False

    def for_queries(self, mask: np.ndarray) -> "MethodScore":
        """Return a copy with per-query arrays sliced by ``mask``."""
        if mask is None or bool(np.all(mask)):
            return self

        def _slice(values):
            if values is None:
                return None
            arr = np.asarray(values)
            if arr.ndim < 1 or arr.shape[0] != mask.shape[0]:
                return values
            return arr[mask]

        return replace(
            self,
            query_scores=_slice(self.query_scores),
            pair_scores=_slice(self.pair_scores),
        )


def method_score(
    name: str,
    query_scores: np.ndarray,
    *,
    higher_is_confidence: bool,
    uncertainty_loss: str,
    pair_scores: Optional[np.ndarray] = None,
    pairwise: bool = False,
    use_vmf_kappa_floor: bool = False,
    percentile_two_sided: bool = False,
    log_name: Optional[str] = None,
) -> MethodScore:
    """Build a ``MethodScore`` from per-query scores and an optional pair matrix."""
    s = np.asarray(query_scores, dtype=np.float64)
    if s.ndim > 1:
        s = np.mean(s, axis=-1)
    pw_name = f"pairwise_{name}" if pair_scores is not None or pairwise else None
    return MethodScore(
        name=name,
        log_name=log_name or METHOD_LOG_NAMES.get(name, name),
        query_scores=s.reshape(-1),
        higher_is_confidence=higher_is_confidence,
        uncertainty_loss=uncertainty_loss,
        pair_scores=pair_scores,
        pairwise_name=pw_name,
        pairwise_plot_filename=(f"ece_pairwise_{name}.png" if pw_name else None),
        use_vmf_kappa_floor=use_vmf_kappa_floor,
        percentile_two_sided=percentile_two_sided,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    """Per-dataset z-score. Non-finite entries become 0; a flat vector stays 0."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    mu = float(np.mean(x[finite]))
    sd = float(np.std(x[finite]))
    if sd < 1e-12:
        return out
    out[finite] = (x[finite] - mu) / sd
    return out


def _query_kappa(q_var: np.ndarray) -> np.ndarray:
    """Mean κ (or variance) per query, matching the ``kappa`` eval score."""
    arr = np.asarray(q_var, dtype=np.float64)
    if arr.ndim > 1:
        return np.mean(arr, axis=-1).reshape(-1)
    return arr.reshape(-1)


def combine_sue_log_kappa(
    sue_log: np.ndarray,
    q_var: np.ndarray,
    *,
    is_vmf: bool,
) -> np.ndarray:
    """Z-scored sum of SUE-log and κ, both as uncertainty (higher = less sure).

    SUE-log is already uncertainty. vMF κ is confidence; the calibrated map is
    ``1/max(κ, 1)`` (κ below 1 is floored, same as ``--ece_vmf_kappa_floor``),
    not ``−κ`` and not ``1/κ`` on the raw tail. Gaussian variance is already
    uncertainty and is left as-is. Each term is standardized on this dataset
    so neither raw scale dominates.
    """
    sue_u = np.asarray(sue_log, dtype=np.float64).reshape(-1)
    kappa = _query_kappa(q_var)
    if is_vmf:
        kappa_u = 1.0 / np.maximum(kappa, 1.0)
    else:
        kappa_u = kappa
    return _zscore(sue_u) + _zscore(kappa_u)


def _log_sue_kappa_scales(
    dataset_name: Optional[str],
    sue_log: np.ndarray,
    kappa: np.ndarray,
    *,
    is_vmf: bool,
) -> None:
    def _brief(name: str, values: np.ndarray) -> str:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return f"{name}: empty"
        return (
            f"{name}: min={np.min(v):.4g} median={np.median(v):.4g} "
            f"max={np.max(v):.4g} mean={np.mean(v):.4g} std={np.std(v):.4g}"
        )

    if is_vmf:
        n_floor = int(np.sum(np.asarray(kappa, dtype=np.float64).reshape(-1) < 1.0))
        n = int(np.asarray(kappa).reshape(-1).size)
        kappa_note = f"vMF uncertainty is 1/max(κ, 1); floored {n_floor}/{n}"
        kappa_term = "1/max(κ, 1)"
    else:
        kappa_note = "variance kept"
        kappa_term = "variance"
    logger.info(
        "SUE-log+κ on %s (%s). %s. %s. score = z(SUE-log) + z(%s).",
        dataset_name or "dataset",
        kappa_note,
        _brief("SUE-log", sue_log),
        _brief("κ", kappa),
        kappa_term,
    )


def compute_learned_pair_scores(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute every registered learned pair score on the retrieval pool."""
    return {
        name: fn(predictions, q_var, db_var, q_desc, db_desc)
        for name, fn in LEARNED_PAIR_SCORE_FNS.items()
    }


def collect_eval_scores(
    predictions: np.ndarray,
    distances: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
    *,
    uncertainty_loss: str,
    database_utms: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
    skip_baselines: bool = False,
    ece_two_sided: bool = True,
    ece_vmf_kappa_floor: bool = False,
) -> Tuple[List[MethodScore], Dict[str, np.ndarray]]:
    """Compute scores once, in QUERY_EVAL_METHOD_ORDER.

    Returns ``(methods, pair_matrices)``. Pair matrices (learned + L2) are
    reused for the combined pool. Missing baselines (no UTMs,
    Nordland/AmsterTime/SPED, K<2 for PA, K<10 for SUE, or ``skip_baselines``)
    are omitted.
    """
    methods: List[MethodScore] = []
    is_vmf = str(uncertainty_loss).lower() == "vmf"
    pair_matrices: Dict[str, np.ndarray] = {}
    floor = bool(ece_vmf_kappa_floor) if is_vmf else False
    two_sided = bool(ece_two_sided)

    methods.append(
        method_score(
            "kappa",
            q_var,
            higher_is_confidence=is_vmf,
            uncertainty_loss=uncertainty_loss,
            use_vmf_kappa_floor=floor,
            percentile_two_sided=two_sided,
        )
    )

    if is_vmf:
        pair_matrices = compute_learned_pair_scores(
            predictions, q_var, db_var, q_desc, db_desc
        )
        for name, scores_2d in pair_matrices.items():
            methods.append(
                method_score(
                    name,
                    scores_2d[:, 0],
                    higher_is_confidence=True,
                    uncertainty_loss="vmf",
                    pair_scores=scores_2d,
                    use_vmf_kappa_floor=floor,
                    percentile_two_sided=two_sided,
                )
            )

    if not skip_baselines:
        l2 = compute_l2_scores(distances)
        pair_matrices["l2"] = l2["pair"]
        methods.append(
            method_score(
                "l2",
                l2["per_query"],
                higher_is_confidence=False,
                uncertainty_loss="gaussian_nll",
                pair_scores=l2["pair"],
                percentile_two_sided=False,
            )
        )
        pa = compute_pa_scores(distances)
        if pa is not None:
            methods.append(
                method_score(
                    "pa",
                    pa,
                    higher_is_confidence=False,
                    uncertainty_loss="gaussian_nll",
                    percentile_two_sided=False,
                )
            )
        else:
            logger.warning("Fewer than 2 retrievals; skipping PA-score baseline.")

        if skip_sue_dataset(dataset_name):
            logger.info("Skipping SUE baseline on %s (mock 1-D UTM).", dataset_name)
        elif database_utms is None:
            logger.warning("No UTM coordinates found; skipping SUE baseline.")
        else:
            sue = compute_sue_scores(distances, predictions, database_utms)
            if sue is None:
                logger.warning(
                    "Fewer than %d retrievals; skipping SUE baseline.", SUE_NUM_NN
                )
            else:
                methods.append(
                    method_score(
                        "sue",
                        sue,
                        higher_is_confidence=False,
                        uncertainty_loss="gaussian_nll",
                        percentile_two_sided=False,
                    )
                )
                sue_log = np.log(sue + 1e-10)
                methods.append(
                    method_score(
                        "sue_log",
                        sue_log,
                        higher_is_confidence=False,
                        uncertainty_loss="gaussian_nll",
                        percentile_two_sided=False,
                    )
                )
                kappa_q = _query_kappa(q_var)
                _log_sue_kappa_scales(dataset_name, sue_log, kappa_q, is_vmf=is_vmf)
                methods.append(
                    method_score(
                        "sue_log_kappa",
                        combine_sue_log_kappa(sue_log, q_var, is_vmf=is_vmf),
                        higher_is_confidence=False,
                        uncertainty_loss="gaussian_nll",
                        percentile_two_sided=True,
                        log_name=METHOD_LOG_NAMES["sue_log_kappa"],
                    )
                )

    return methods, pair_matrices


__all__ = [
    "BASELINE_PAIRWISE_SCORE_NAMES",
    "BASELINE_QUERY_SCORE_NAMES",
    "HYBRID_QUERY_SCORE_NAMES",
    "LEARNED_PAIRWISE_SCORE_NAMES",
    "LEARNED_PAIR_SCORE_FNS",
    "LEARNED_QUERY_SCORE_NAMES",
    "METHOD_LOG_NAMES",
    "MethodScore",
    "QUERY_EVAL_METHOD_ORDER",
    "collect_eval_scores",
    "combine_sue_log_kappa",
    "method_score",
    "compute_joint_kappa",
    "compute_l2_joint_kappa",
    "compute_mls",
    "SUE_NUM_NN",
    "compute_l2_scores",
    "compute_learned_pair_scores",
    "compute_pa_scores",
    "compute_sue_scores",
]
