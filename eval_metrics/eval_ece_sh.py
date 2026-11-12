"""Expected Calibration Error (ECE) for uncertainty-aware VPR.

Bins queries by predicted uncertainty (mean variance), computes recall@K and
mAP@K per bin, and measures how well uncertainty tracks retrieval performance.

A well-calibrated model should have high recall in low-uncertainty bins and low
recall in high-uncertainty bins, with a smooth monotonic relationship.

ECE = Σ (bin_weight * |bin_metric - expected_metric|)  over all bins.

Adapted from: https://github.com/ramdrop/stun
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from eval_metrics.recall import _cal_recall

logger = logging.getLogger(__name__)


def vmf_like_uncertainty_loss(uncertainty_loss: str) -> bool:
    """vMF and kappa_ms both use softplus κ where higher = more confident."""
    return str(uncertainty_loss).lower() in ("vmf", "kappa_ms")


def per_query_scalars(scores: np.ndarray) -> np.ndarray:
    """One scalar per query: mean over the last axis if ``scores`` is ``[N, D]``."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 0:
        return s.reshape(1)
    if s.ndim > 1:
        return np.mean(s, axis=-1)
    return s.reshape(-1)


def scores_to_uncertainty(
    scores: np.ndarray,
    uncertainty_loss: str,
    vmf_kappa_floor: bool = False,
    invert: str = "reciprocal",
) -> np.ndarray:
    """Map scores to uncertainty (higher = more uncertain). Shape is preserved.

    For vMF, ``invert='reciprocal'`` uses ``1/κ`` and ``invert='neg'`` uses ``−κ``.
    Non-vMF scores are returned as-is (already uncertainty).
    """
    s = np.asarray(scores, dtype=np.float64)
    if not vmf_like_uncertainty_loss(uncertainty_loss):
        return s
    if vmf_kappa_floor:
        s = np.maximum(s, 1.0)
    if invert == "neg":
        return -s
    return 1.0 / (s + 1e-6)


def mean_query_uncertainty(
    query_scores: np.ndarray,
    uncertainty_loss: str,
    vmf_kappa_floor: bool = False,
    invert: str = "reciprocal",
) -> np.ndarray:
    """Per-query scalar uncertainty (higher = more uncertain).

    ``query_scores`` is already one value per query (mean-κ / L2 / …). Collapses
    extra axes first, then applies the same invert as ``scores_to_uncertainty``.
    """
    return scores_to_uncertainty(
        per_query_scalars(query_scores),
        uncertainty_loss,
        vmf_kappa_floor,
        invert=invert,
    )


# KappaPlace primary: two-sided p1–p99 for learned κ / joint κ / L2·κ; one-sided
# [min, p99] for L2 / PA / SUE. Nested equal-mass / opposite-clip / −κ are --ece_extras.
PERCENTILE_CLIP_PCT = 1.0  # clip 1% tail(s): two-sided → [p1, p99]; one-sided → hi at p99


def _get_zoomed_bins(variances: np.ndarray, num_bins: int, zoom_threshold: float = 0.001):
    """Equal-width bins with adaptive zoom to handle long-tailed distributions.

    Iteratively narrows the upper range until the last bin has at least zoom_threshold%
    of the data (default 0.1%), preventing empty high-uncertainty bins from outliers.

    Returns:
        indices: list of arrays, each containing query indices for that bin.
        k: number of zoom steps applied.
    """
    s_min, s_max = np.min(variances), np.max(variances)
    bins_parent = np.linspace(s_min, s_max, num=num_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_bins)
        for i in range(num_bins - 1):
            left = np.where(variances >= bins_child[i])
            if i != num_bins - 2:
                right = np.where(variances < bins_child[i + 1])
            else:
                right = np.where(variances <= bins_child[i + 1])
            indices.append(np.intersect1d(left[0], right[0]))
        if len(indices[-1]) > int(variances.shape[0] * zoom_threshold) or k == num_bins - 2:
            break
        k += 1
    
    bin_sizes = [len(b) for b in indices]
    logger.debug(f"ECE Binning: zoom_k={k}, bin_sizes={bin_sizes} (threshold={int(variances.shape[0]*zoom_threshold)})")
    
    return indices, k


def _get_percentile_bins(variances: np.ndarray, num_bins: int,
                          two_sided: bool = False,
                          clip_pct: float = PERCENTILE_CLIP_PCT):
    """Percentile bounds + clip, then uniform bins (every sample is assigned).

    Percentiles define bounds ``[lo, hi]`` on the **raw** scores. Each score is then
    ``np.clip``ped into ``[lo, hi]``, and equal-width bins are built on ``[lo, hi]``.
    No sample is omitted: outliers are folded into the lowest/highest bins.

    Args:
        variances: 1D array of scores.
        num_bins: number of bin edges (so num_bins - 1 actual bins).
        two_sided: if True, ``lo``/``hi`` are the ``clip_pct`` and ``100 - clip_pct``
            percentiles (``PERCENTILE_CLIP_PCT`` default 1.0 → 1st and 99th).
        clip_pct: Tail fraction defining the clip window, in percent.
            Default ``PERCENTILE_CLIP_PCT`` (1.0): one-sided uses min–(100−clip_pct);
            two-sided uses clip_pct–(100−clip_pct).

    Returns:
        indices: list of length num_bins - 1, each entry is the array of indices in that bin.
        k: 0 (kept for return-shape compatibility with _get_zoomed_bins).
    """
    if two_sided:
        lo, hi = np.percentile(variances, [clip_pct, 100.0 - clip_pct])
    else:
        lo = float(np.min(variances))
        hi = float(np.percentile(variances, 100.0 - clip_pct))
    if hi <= lo:
        hi = float(np.max(variances))
    v = np.clip(np.asarray(variances, dtype=np.float64), lo, hi)
    clip_stats = {
        "before_min": float(np.min(variances)),
        "before_max": float(np.max(variances)),
        "before_mean": float(np.mean(variances)),
        "before_median": float(np.median(variances)),
        "after_min": float(np.min(v)),
        "after_max": float(np.max(v)),
        "after_mean": float(np.mean(v)),
        "after_median": float(np.median(v)),
        "clip_lo": float(lo),
        "clip_hi": float(hi),
    }
    edges = np.linspace(lo, hi, num=num_bins)

    # Assign every value to a bin in one pass. Subtract 1 so bin indices start at 0.
    bin_assignments = np.digitize(v, edges) - 1
    # np.digitize puts values exactly equal to hi into an out-of-bounds bin (num_bins - 1).
    # The original loop folded those into the final bin; mirror that here.
    bin_assignments[bin_assignments == num_bins - 1] = num_bins - 2
    indices = [np.where(bin_assignments == i)[0] for i in range(num_bins - 1)]
    bin_sizes = [len(b) for b in indices]
    n = int(variances.shape[0])
    assigned = int(sum(bin_sizes))
    if assigned != n:
        logger.warning(
            "ECE percentile bins: assigned %d/%d samples (unexpected; check for NaNs in uncertainty)",
            assigned, n,
        )
    logger.debug(f"ECE Binning (percentile, two_sided={two_sided}, clip-to-bounds): bin_sizes={bin_sizes} "
                 f"[lo={lo:.4g}, hi={hi:.4g}, clip_pct={clip_pct}%]")
    return indices, 0, clip_stats


def _get_equal_mass_bins(variances: np.ndarray, num_bins: int):
    """
    True equal-mass percentile binning (AdaECE).

    Generates bin edges based on data quantiles so each bin contains an equal
    number of samples. Clipping is removed as it is mathematically unnecessary
    for equal-mass binning.
    """
    variances = np.asarray(variances, dtype=np.float64)

    # 1. Define edges using percentiles so every bin gets ~equal samples
    # np.linspace(0, 100, num_bins) creates percentages like [0, 10, 20, ... 100]
    edges = np.percentile(variances, np.linspace(0, 100, num_bins))

    # Ensure the absolute min and max are strictly enforced at the boundaries
    edges[0] = np.min(variances)
    edges[-1] = np.max(variances)

    # 2. Vectorized bin assignment using the optimized logic
    bin_assignments = np.digitize(variances, edges) - 1

    # np.digitize puts values exactly equal to the max edge into an out-of-bounds bin.
    # We fold them into the final bin (num_bins - 2).
    bin_assignments[bin_assignments == num_bins - 1] = num_bins - 2

    # 3. Group the original array indices by their assigned bin
    indices = [np.where(bin_assignments == i)[0] for i in range(num_bins - 1)]

    # 4. Logging and validation
    bin_sizes = [len(b) for b in indices]
    n = int(variances.shape[0])
    assigned = int(sum(bin_sizes))

    if assigned != n:
        logger.warning(
            "ECE percentile bins: assigned %d/%d samples (unexpected; check for NaNs in uncertainty)",
            assigned, n,
        )

    logger.debug(f"ECE Binning (Equal-Mass): bin_sizes={bin_sizes}")

    # Return basic stats to maintain pipeline compatibility
    stats = {
        "min": float(edges[0]),
        "max": float(edges[-1]),
        "mean": float(np.mean(variances)),
        "median": float(np.median(variances))
    }

    return indices, 0, stats


def _select_bins(variances: np.ndarray, num_bins: int, bin_mode: str,
                 zoom_threshold: float = 0.001, percentile_two_sided: bool = False):
    """Dispatch helper: pick zoom, equal-width percentile, or equal-mass binning."""
    mode = str(bin_mode).lower()
    if mode == "percentile":
        return _get_percentile_bins(variances, num_bins, two_sided=percentile_two_sided)
    if mode == "equal_mass":
        return _get_equal_mass_bins(variances, num_bins)
    indices, zoom_k = _get_zoomed_bins(variances, num_bins, zoom_threshold=zoom_threshold)
    return indices, zoom_k, None


def _score_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def _clamp_scores_for_logging(scores: np.ndarray, clip_stats: Optional[Dict[str, float]]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not clip_stats or "clip_lo" not in clip_stats:
        return scores
    return np.clip(scores, clip_stats["clip_lo"], clip_stats["clip_hi"])


def _build_score_stats(
    variances_before_clamp: np.ndarray,
    variances_after_clamp: np.ndarray,
    confidence_before_clamp: np.ndarray,
    confidence_after_clamp: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    return {
        "variances_before_clamp": _score_summary(variances_before_clamp),
        "variances_after_clamp": _score_summary(variances_after_clamp),
        "confidence_before_clamp": _score_summary(confidence_before_clamp),
        "confidence_after_clamp": _score_summary(confidence_after_clamp),
    }


def _cal_apk(positives, prediction, k):
    """Average precision at K for a single query."""
    if len(positives) == 0 or k == 0:
        return 0.0
    if len(prediction) > k:
        prediction = prediction[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(prediction):
        if p in positives and p not in prediction[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(positives), k) * 100.0


def _cal_mapk(predictions, positives_per_query, k, drop_empty_gt=False):
    """Mean average precision at K. Empty-GT queries score 0 unless dropped."""
    vals = [
        _cal_apk(pos, pred, k)
        for pos, pred in zip(positives_per_query, predictions)
        if not drop_empty_gt or len(pos) > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def _bin_pr(predictions: np.ndarray, distances: np.ndarray, positives_per_query: List):
    """Precision-recall curve by sweeping distance threshold. Returns (recalls, precisions) for AP integration."""
    dists_u = np.linspace(np.min(distances[:, 0]), np.max(distances[:, 0]), num=100)
    recalls, precisions = [], []
    for th in dists_u:
        tp = fp = fn = tn = 0
        for q in range(distances.shape[0]):
            if distances[q, 0] < th:
                if np.any(np.isin(predictions[q, 0], positives_per_query[q])):
                    tp += 1
                else:
                    fp += 1
            else:
                if np.any(np.isin(predictions[q, 0], positives_per_query[q])):
                    fn += 1
                else:
                    tn += 1
        if (tp + fn) == 0 or (tp + fp) == 0:
            continue
        recalls.append(tp / (tp + fn))
        precisions.append(tp / (tp + fp))
    return recalls, precisions


def compute_ece(
    predictions: np.ndarray,
    positives_per_query: List,
    query_variances: np.ndarray,
    n_values: List[int] = [1, 5, 10],
    num_bins: int = 11,
    output_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    distances: Optional[np.ndarray] = None,
    uncertainty_loss: str = "gaussian_nll",
    plot_filename: str = "ece_plot.png",
    zoom_threshold: float = 0.001,
    bin_mode: str = "percentile",
    vmf_kappa_floor: bool = False,
    percentile_two_sided: bool = False,
    invert: str = "reciprocal",
) -> Dict:
    """Compute Expected Calibration Error for uncertainty-aware retrieval.

    Args:
        predictions: [num_queries, max_k] predicted DB indices per query.
        positives_per_query: list of arrays, ground-truth positive DB indices per query.
        query_variances: [num_queries] per-query score (mean-κ, L2, …). 2D ``[N, D]``
            is still accepted and reduced with a mean on the last axis.
        n_values: recall@K values to evaluate.
        num_bins: number of bin edges (actual bins = num_bins - 1).
        output_dir: if provided, save ECE plot here.
        metrics: list of metrics to compute: 'recall', 'map', 'ap'. Default ['recall'].
        distances: [num_queries, max_k] L2 distances per prediction. Required for 'ap'.
        uncertainty_loss: type of uncertainty loss used (e.g., 'gaussian_nll', 'vmf').
        plot_filename: Base filename for calibration curves of the **first** metric in
            ``metrics`` (usually recall → ``ece_plot.png``). Also writes ``{stem}_bins.png``,
            ``{stem}_weights.png``, and extra ``{stem}_map.png`` / ``{stem}_ap.png`` when
            those metrics are included.
        zoom_threshold: Fraction of data required in the last bin for adaptive zoom (default 0.001).
        bin_mode: ``percentile`` (KappaPlace default: equal-width after percentile clip),
            ``zoom`` (legacy, adaptive), or ``equal_mass`` (quantile edges, ~same sample
            count per bin).
        vmf_kappa_floor: when True, clip kappa < 1 to 1 before inversion (vMF only) to
            avoid huge 1/kappa outliers from low-confidence queries.
        invert: vMF confidence→uncertainty map: ``reciprocal`` (1/κ) or ``neg`` (−κ).
        percentile_two_sided: when bin_mode='percentile', clip both the low and high tails
            (p_clip and p_{100-clip}) before forming equal-width bins. Default at this
            layer is one-sided (high tail only). The eval protocol passes True for the
            κ family and False for L2 / PA / SUE.

    Returns:
        dict with ece_recall, ece_map, ece_ap (when included), bin_*.
    """
    metrics = metrics or ["recall"]
    logger.debug(f"compute_ece: Saving {plot_filename} to {output_dir}")
    variances_before_clamp = per_query_scalars(query_variances)
    variances_after_clamp = variances_before_clamp
    mean_var = mean_query_uncertainty(
        variances_before_clamp,
        uncertainty_loss,
        vmf_kappa_floor=vmf_kappa_floor,
        invert=invert,
    )
    if vmf_like_uncertainty_loss(uncertainty_loss):
        variances_after_clamp = np.maximum(variances_before_clamp, 1.0) if vmf_kappa_floor else variances_before_clamp
        
    bin_indices, zoom_k, clip_stats = _select_bins(
        mean_var, num_bins, bin_mode,
        zoom_threshold=zoom_threshold,
        percentile_two_sided=percentile_two_sided,
    )
    mean_var_after_clamp = _clamp_scores_for_logging(mean_var, clip_stats)
    num_actual_bins = num_bins - 1
    num_queries = len(mean_var)

    bin_recalls = np.zeros((num_actual_bins, len(n_values)))
    bin_map = np.zeros((num_actual_bins, len(n_values)))
    bin_ap = np.zeros(num_actual_bins) if "ap" in metrics else None
    bin_weights = np.zeros(num_actual_bins)

    ece_recall = np.zeros(len(n_values))
    ece_map = np.zeros(len(n_values))
    ece_ap = 0.0 if "ap" in metrics else None

    for b, q_in_bin in enumerate(bin_indices):
        if len(q_in_bin) == 0:
            continue
        bin_weights[b] = len(q_in_bin) / num_queries

        bin_preds = predictions[q_in_bin]
        bin_positives = [positives_per_query[i] for i in q_in_bin]

        # Expected performance: linearly decreasing from 1.0 (low uncertainty) to 0.0 (high uncertainty)
        expected = (num_actual_bins - 1 - b) / (num_actual_bins - 1)

        if "recall" in metrics:
            recall_at_n = _cal_recall(bin_preds, bin_positives, n_values)
            bin_recalls[b] = recall_at_n
            for i in range(len(n_values)):
                ece_recall[i] += bin_weights[b] * abs(recall_at_n[i] / 100.0 - expected)

        if "map" in metrics:
            map_at_n = [_cal_mapk(bin_preds, bin_positives, n) for n in n_values]
            bin_map[b] = map_at_n
            for i in range(len(n_values)):
                ece_map[i] += bin_weights[b] * abs(map_at_n[i] / 100.0 - expected)

        if "ap" in metrics and distances is not None:
            bin_dists = distances[q_in_bin]
            recalls_pr, precisions_pr = _bin_pr(bin_preds, bin_dists, bin_positives)
            ap = 0.0
            for j in range(len(recalls_pr) - 1):
                ap += precisions_pr[j] * (recalls_pr[j + 1] - recalls_pr[j])
            bin_ap[b] = ap
            ece_ap += bin_weights[b] * abs(ap - expected)

    if output_dir:
        _plot_ece(bin_recalls, bin_map, bin_ap, bin_weights, bin_indices, n_values, num_actual_bins,
                 output_dir, metrics, plot_filename)

    result = {
        "bin_recalls": bin_recalls,
        "bin_map": bin_map,
        "bin_weights": bin_weights,
        "score_stats": _build_score_stats(
            variances_before_clamp,
            variances_after_clamp,
            mean_var,
            mean_var_after_clamp,
        ),
    }
    if clip_stats:
        result["clip_stats"] = clip_stats
    if "recall" in metrics:
        result["ece_recall"] = {n: e for n, e in zip(n_values, ece_recall)}
    if "map" in metrics:
        result["ece_map"] = {n: e for n, e in zip(n_values, ece_map)}
    if "ap" in metrics:
        result["ece_ap"] = ece_ap
        result["bin_ap"] = bin_ap
    return result


def _get_paper_rcparams():
    """Return matplotlib rcParams for publication-quality fonts matching reference figure."""
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "cm",
        "font.size": 40,
        "axes.titlesize": 40,
        "axes.labelsize": 48,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 30,
        "legend.frameon": True,
    }


# Calibration curves: thick lines + large markers (readable in slides / downsized figures).
_ECE_DATA_LINEWIDTH = 3.0
_ECE_REF_LINEWIDTH = 2.8
_ECE_MARKERSIZE = 11


def _plot_ece(bin_recalls, bin_map, bin_ap, bin_weights, bin_indices, n_values, num_bins, output_dir, metrics, plot_filename):
    """Save ECE figures as separate PNGs (bin histogram, weights, one file per metric curve)."""
    try:
        import matplotlib.pyplot as plt
        with plt.rc_context(_get_paper_rcparams()):
            out_dir = Path(output_dir)
            stem = Path(plot_filename).stem
            bins_png = f"{stem}_bins.png"
            weights_png = f"{stem}_weights.png"
            x = np.arange(1, num_bins + 1)
            perfect_x = np.array([1, num_bins])
            perfect_y_pct = np.array([100.0, 0.0])
            even_ticks = [t for t in x if t % 2 == 0]

            saved: List[str] = []

            # --- 1. Bin distribution ---
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.bar(x, [len(b) for b in bin_indices])
            ax.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
            ax.set_ylabel("Number of samples")
            ax.set_xticks(even_ticks)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / bins_png, dpi=300)
            plt.close(fig)
            saved.append(bins_png)

            # --- 2. Bin weights ---
            fig_w, ax_w = plt.subplots(figsize=(12, 10))
            ax_w.bar(x, bin_weights)
            ax_w.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
            ax_w.set_ylabel("Bin weight (fraction of queries)")
            ax_w.set_xticks(even_ticks)
            ax_w.grid(True, alpha=0.3)
            fig_w.tight_layout()
            fig_w.savefig(out_dir / weights_png, dpi=300)
            plt.close(fig_w)
            saved.append(weights_png)

            # --- 3. Calibration curves: first metric uses ``plot_filename``; others use ``{stem}_<metric>.png``
            primary_used = False

            if "recall" in metrics:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_recall.png"
                fig_r, ax_r = plt.subplots(figsize=(12, 10))
                ax_r.plot(
                    perfect_x,
                    perfect_y_pct,
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                for i, n in enumerate(n_values):
                    ax_r.plot(
                        x,
                        bin_recalls[:, i],
                        marker="o",
                        markersize=_ECE_MARKERSIZE,
                        linewidth=_ECE_DATA_LINEWIDTH,
                        label=f"R@{n}",
                    )
                ax_r.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_r.set_ylabel("Recall@K (%)")
                ax_r.set_xticks(even_ticks)
                ax_r.legend(loc="lower left")
                ax_r.grid(True, alpha=0.3)
                fig_r.tight_layout()
                fig_r.savefig(outp, dpi=300)
                plt.close(fig_r)
                saved.append(outp.name)
                primary_used = True

            if "map" in metrics:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_map.png"
                fig_m, ax_m = plt.subplots(figsize=(12, 10))
                ax_m.plot(
                    perfect_x,
                    perfect_y_pct,
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                for i, n in enumerate(n_values):
                    ax_m.plot(
                        x,
                        bin_map[:, i],
                        marker="o",
                        markersize=_ECE_MARKERSIZE,
                        linewidth=_ECE_DATA_LINEWIDTH,
                        label=f"mAP@{n}",
                    )
                ax_m.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_m.set_ylabel("mAP@K (%)")
                ax_m.set_xticks(even_ticks)
                ax_m.legend(loc="lower left", prop={"weight": "bold"})
                ax_m.grid(True, alpha=0.3)
                fig_m.tight_layout()
                fig_m.savefig(outp, dpi=300)
                plt.close(fig_m)
                saved.append(outp.name)
                primary_used = True

            if "ap" in metrics and bin_ap is not None:
                outp = out_dir / plot_filename if not primary_used else out_dir / f"{stem}_ap.png"
                fig_a, ax_a = plt.subplots(figsize=(12, 10))
                ax_a.plot(
                    perfect_x,
                    np.array([1.0, 0.0]),
                    color="grey",
                    linestyle="--",
                    linewidth=_ECE_REF_LINEWIDTH,
                    label="Perfect calibration",
                    zorder=0,
                )
                ax_a.plot(
                    x,
                    bin_ap,
                    marker="o",
                    markersize=_ECE_MARKERSIZE,
                    linewidth=_ECE_DATA_LINEWIDTH,
                    label="AP",
                )
                ax_a.set_xlabel("Uncertainty bin (low $\\rightarrow$ high)")
                ax_a.set_ylabel("AP")
                ax_a.set_xticks(even_ticks)
                ax_a.legend(loc="lower left", prop={"weight": "bold"})
                ax_a.grid(True, alpha=0.3)
                fig_a.tight_layout()
                fig_a.savefig(outp, dpi=300)
                plt.close(fig_a)
                saved.append(outp.name)

            logger.debug("ECE plots saved to %s: %s", out_dir, saved)
    except ImportError:
        logger.warning("matplotlib not installed, skipping ECE plot.")
