"""Build W&B metric dicts and collect plot paths from per-dataset eval outputs."""
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_metrics.rejection_curves import rejection_delta_pcts_from_entry, rejection_delta_pcts_from_stats
from eval_metrics.uncertainty import compute_score_statistics

logger = logging.getLogger(__name__)

# Uncertainty ECE variants used for calibration tables and bins-distribution panels.
UNCERTAINTY_ECE_VARIANTS = (
    "kappa",
    "joint_kappa",
    "l2_joint_kappa",
    "mls",
    "pairwise_joint_kappa",
    "pairwise_l2_joint_kappa",
    "pairwise_mls",
)

# Baseline ECE variants (plot stem ``ece_{name}``).
BASELINE_ECE_VARIANTS = (
    "l2",
    "pairwise_l2",
    "pa",
    "sue",
    "sue_log",
    "sue_log_kappa",
)

# Rows for ``calibration_ece`` summary (learned + baseline ECE_R@k, plus −κ / one-two-sided).
ECE_NEG_KAPPA_VARIANTS = tuple(f"{v}_neg" for v in UNCERTAINTY_ECE_VARIANTS)
_ECE_BIN_BASE = UNCERTAINTY_ECE_VARIANTS + BASELINE_ECE_VARIANTS
ECE_TWO_SIDED_VARIANTS = tuple(f"{v}_two_sided" for v in _ECE_BIN_BASE)
ECE_ONE_SIDED_VARIANTS = tuple(f"{v}_one_sided" for v in _ECE_BIN_BASE)
ECE_EQUAL_MASS_VARIANTS = tuple(f"{v}_equal_mass" for v in _ECE_BIN_BASE)
CALIBRATION_ECE_VARIANTS = (
    UNCERTAINTY_ECE_VARIANTS
    + BASELINE_ECE_VARIANTS
    + ECE_NEG_KAPPA_VARIANTS
    + ECE_TWO_SIDED_VARIANTS
    + ECE_ONE_SIDED_VARIANTS
    + ECE_EQUAL_MASS_VARIANTS
)

CALIBRATION_ECE_AP_PANEL_KEY = "calibration_ece_ap"
ECE_CURVES_AP_PANEL_PREFIX = "ece_curves_ap"
REJECTION_CURVES_PANEL_PREFIX = "rejection_curves"
REJECTION_CURVES_TABLE_KEY = "rejection_curves"
# Query-level: one overlay PNG per K. Pair-level: one all-method overlay.
REJECTION_CURVES_PANEL_VARIANTS = (
    "query_r1",
    "query_r5",
    "query_r10",
    "query_r20",
    "pairwise",
)
REJECTION_CURVES_PLOT_SUFFIXES = (
    ("query_r1", ""),
    ("query_r5", "_r5"),
    ("query_r10", "_r10"),
    ("query_r20", "_r20"),
    ("pairwise", "_pairwise"),
)
# Table row order (SUE-log omitted: same rejection ranking as SUE).
REJECTION_CURVES_METHOD_ORDER = (
    "kappa",
    "joint_kappa",
    "l2_joint_kappa",
    "mls",
    "pairwise_joint_kappa",
    "pairwise_l2_joint_kappa",
    "pairwise_mls",
    "l2",
    "pairwise_l2",
    "pa",
    "sue",
    "sue_log_kappa",
)
UNCERTAINTY_STATS_CURVES_PANEL_PREFIX = "uncertainty_stats_curves"

# Fixed ECE curves panel per dataset (same order as the calibration table).
ECE_CURVES_PANEL_VARIANTS = CALIBRATION_ECE_VARIANTS

# Score tensors summarized inside ``uncertainty_stats`` (after descriptor row).
SCORE_STAT_VARIANTS = UNCERTAINTY_ECE_VARIANTS + BASELINE_ECE_VARIANTS

PAIRWISE_SCORE_VARIANTS = (
    "pairwise_l2",
    "pairwise_joint_kappa",
    "pairwise_l2_joint_kappa",
    "pairwise_mls",
)


def _ece_plot_stem(variant: str) -> str:
    return f"ece_{variant}"


def _recall_value_dict(arr, rv: List[int]) -> Dict[int, float]:
    if arr is None:
        return {}
    return {int(k): float(arr[rv.index(k)]) for k in rv if rv.index(k) < len(arr)}


def _uncertainty_variant_keys(panel: Dict[str, Any]) -> set:
    return {k for k in UNCERTAINTY_ECE_VARIANTS if k in (panel.get("ece") or {})}


def _ece_field(result: Any, field: str):
    if not isinstance(result, dict):
        return None
    return result.get(field)


def _build_panel_data(
    dataset_name: str,
    args: Dict[str, Any],
    recalls,
    map_at_k,
    uncertainty_ece_results: Optional[Dict[str, Any]],
    uncertainty_raw_scores: Optional[Dict[str, Any]],
    baseline_results: Optional[Dict[str, Any]],
    uncertainty_stats: Optional[Dict[str, float]],
    base_dataset_name: Optional[str] = None,
    query_folder_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build structured eval data for W&B panels and summary tables."""
    panel: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "base_dataset_name": base_dataset_name or dataset_name,
        "query_folder_name": query_folder_name,
        "recalls": _recall_value_dict(recalls, args.get("recall_values", [1, 5, 10, 20])),
        "map_at_k": _recall_value_dict(map_at_k, args.get("recall_values", [1, 5, 10, 20])),
        "ece": {},
        "variance_stat": dict(uncertainty_stats or {}),
        "raw_scores": dict(uncertainty_raw_scores or {}),
    }

    for variant, ece_result in (uncertainty_ece_results or {}).items():
        if variant == "raw_scores" or not isinstance(ece_result, dict):
            continue
        panel["ece"][variant] = ece_result

    if baseline_results:
        baseline_raw = baseline_results.get("raw_scores") or {}
        for variant in ("l2", "pa", "sue", "sue_log"):
            ece = baseline_results.get(f"ece_{variant}")
            if isinstance(ece, dict):
                panel["ece"][variant] = {
                    "ece_recall": {int(k): float(val) for k, val in ece.items()}
                }
                if baseline_raw.get(variant) is not None:
                    panel["raw_scores"][variant] = baseline_raw[variant]
            ap = baseline_results.get(f"ece_{variant}_ap")
            if ap is not None:
                panel.setdefault("ece", {}).setdefault(variant, {})["ece_ap"] = float(ap)
        pair = baseline_results.get("ece_pairwise_l2")
        if isinstance(pair, dict) and "ece_recall" in pair:
            panel["ece"]["pairwise_l2"] = pair
            if baseline_raw.get("pairwise_l2") is not None:
                panel["raw_scores"]["pairwise_l2"] = baseline_raw["pairwise_l2"]

    logger.debug(
        "%s W&B ece variants: %s",
        dataset_name,
        sorted((panel.get("ece") or {}).keys()),
    )
    return panel


def _rejection_delta_wandb_metrics(
    prefix: str,
    variant: str,
    stats: Any,
    *,
    key_suffix: str = "",
) -> Dict[str, float]:
    """``rejection_{joint_kappa}_delta_20_recall_01`` vs ``rejection_{pairwise_joint_kappa}_...``."""
    if not isinstance(stats, dict):
        return {}
    method = str(stats.get("method") or variant)
    n_rank = stats.get("n_rank")
    rank_part = f"_recall_{int(n_rank):02d}" if n_rank is not None else ""
    out: Dict[str, float] = {}
    for pct in rejection_delta_pcts_from_entry(stats):
        val = stats.get(f"delta_at_{pct}")
        if val is None:
            continue
        out[f"{prefix}rejection_{method}_delta_{pct}{rank_part}{key_suffix}"] = float(val)
    return out


def panel_to_wandb_metrics(
    panel: Dict[str, Any],
    wandb_step: Optional[int] = None,
) -> Dict[str, float]:
    """Flatten panel data into W&B scalar keys."""
    dataset_name = panel["dataset_name"]
    prefix = f"Eval_{dataset_name}/"
    wandb_metrics: Dict[str, float] = {}

    for k, v in panel.get("recalls", {}).items():
        wandb_metrics[f"{prefix}recall_{k:02d}"] = float(v)
    for k, v in panel.get("map_at_k", {}).items():
        wandb_metrics[f"{prefix}map_{k:02d}"] = float(v)

    uncertainty_keys = _uncertainty_variant_keys(panel)
    for variant, result in (panel.get("ece") or {}).items():
        rec = _ece_field(result, "ece_recall") or {}
        for k, v in rec.items():
            key = f"{prefix}ece_{variant}_recall_{int(k):02d}"
            wandb_metrics[key] = float(v)
            if wandb_step is not None and variant in uncertainty_keys:
                wandb_metrics[f"uncertainty/ece_{variant}/recall_{int(k):02d}"] = float(v)
        if variant in uncertainty_keys:
            ece_map = _ece_field(result, "ece_map") or {}
            for k, v in ece_map.items():
                key = f"{prefix}ece_{variant}_map_{int(k):02d}"
                wandb_metrics[key] = float(v)
                if wandb_step is not None:
                    wandb_metrics[f"uncertainty/ece_{variant}/map_{int(k):02d}"] = float(v)
            ap = _ece_field(result, "ece_ap")
            if ap is not None:
                key = f"{prefix}ece_{variant}_ap"
                wandb_metrics[key] = float(ap)
                if wandb_step is not None:
                    wandb_metrics[f"uncertainty/ece_{variant}/ap"] = float(ap)
    for variant, result in (panel.get("ece_norm_per_dataset") or {}).items():
        rec = _ece_field(result, "ece_recall") or {}
        for k, v in rec.items():
            wandb_metrics[f"{prefix}ece_{variant}_recall_{int(k):02d}_norm_per_dataset"] = float(v)
        ap = _ece_field(result, "ece_ap")
        if ap is not None:
            wandb_metrics[f"{prefix}ece_{variant}_ap_norm_per_dataset"] = float(ap)

    for name, v in panel.get("variance_stat", {}).items():
        wandb_metrics[f"{prefix}uncertainty_{name}"] = float(v)
        if wandb_step is not None:
            wandb_metrics[f"uncertainty/stats_{name}"] = float(v)

    for variant in SCORE_STAT_VARIANTS:
        raw = panel.get("raw_scores", {}).get(variant)
        if raw is None:
            continue
        for name, v in compute_score_statistics(raw).items():
            wandb_metrics[f"{prefix}{variant}_{name}"] = float(v)

    for variant, stats in (panel.get("rejection_curves") or {}).items():
        method = str((stats or {}).get("method") or variant) if isinstance(stats, dict) else str(variant)
        if method == "sue_log" or str(variant) == "sue_log":
            continue
        wandb_metrics.update(
            _rejection_delta_wandb_metrics(prefix, variant, stats)
        )
    for variant, stats in (panel.get("rejection_curves_norm_per_dataset") or {}).items():
        method = str((stats or {}).get("method") or variant) if isinstance(stats, dict) else str(variant)
        if method == "sue_log" or str(variant) == "sue_log":
            continue
        wandb_metrics.update(
            _rejection_delta_wandb_metrics(
                prefix, variant, stats, key_suffix="_norm_per_dataset"
            )
        )

    return wandb_metrics


def collect_wandb_images(
    dataset_name: str,
    dataset_output_dir: Path,
    args: Dict[str, Any],
    save_plots: bool,
) -> Optional[Dict[str, Any]]:
    """Collect variance-distribution and prediction plot paths for W&B logging."""
    if not args.get("use_wandb") or not dataset_output_dir.exists():
        return None

    prefix = f"Eval_{dataset_name}/"
    wandb_images: Dict[str, Any] = {}

    var_path = dataset_output_dir / "variance_distribution.png"
    if var_path.exists():
        wandb_images[f"{prefix}variance_distribution"] = var_path.resolve()

    preds_dir = dataset_output_dir / "preds"
    if save_plots and preds_dir.exists():
        image_list = sorted(preds_dir.glob("*.jpg"))
        if image_list:
            wandb_images[f"{prefix}predictions"] = image_list

    return wandb_images or None


def _bins_png_path(output_dir: Path, variant_key: str) -> Path:
    """Resolve ECE bin-count plot path."""
    return output_dir / f"{_ece_plot_stem(variant_key)}_bins.png"


def _curve_png_path(output_dir: Path, variant_key: str) -> Path:
    """Resolve ECE curve plot path (``plot_ece`` outputs, not bins)."""
    return output_dir / f"ece_{variant_key}.png"


def _curve_ap_png_path(output_dir: Path, variant_key: str) -> Path:
    """Resolve AP calibration curve path (``{stem}_ap.png`` from ``_plot_ece``)."""
    base = _curve_png_path(output_dir, variant_key)
    return base.parent / f"{base.stem}_ap.png"


def collect_ece_curve_images(output_dir: Optional[Path]) -> Dict[str, Path]:
    """Collect ``plot_ece`` PNGs for the fixed ECE-curves panel variants."""
    if output_dir is None or not output_dir.exists():
        return {}

    images: Dict[str, Path] = {}
    for variant in ECE_CURVES_PANEL_VARIANTS:
        path = _curve_png_path(output_dir, variant)
        if path.exists():
            images[variant] = path.resolve()
    return images


def collect_ece_curve_ap_images(output_dir: Optional[Path]) -> Dict[str, Path]:
    """Collect AP calibration curve PNGs (``*_ap.png``) for the ECE-curves-ap panel."""
    if output_dir is None or not output_dir.exists():
        return {}

    images: Dict[str, Path] = {}
    for variant in ECE_CURVES_PANEL_VARIANTS:
        path = _curve_ap_png_path(output_dir, variant)
        if path.exists():
            images[variant] = path.resolve()
    return images


def collect_bins_distribution_images(output_dir: Optional[Path]) -> Dict[str, Path]:
    """Collect ``*_bins.png`` for uncertainty ECE and baseline ECE variants."""
    if output_dir is None or not output_dir.exists():
        return {}

    images: Dict[str, Path] = {}
    for variant in CALIBRATION_ECE_VARIANTS:
        path = _bins_png_path(output_dir, variant)
        if path.exists():
            images[variant] = path.resolve()
    return images


def collect_combined_bins_distribution_images(
    output_dir: Path,
    output_subdir: str,
) -> Dict[str, Path]:
    """Collect combined ECE bin plots and optional kappa distribution histogram."""
    images = collect_bins_distribution_images(output_dir)
    dist_path = output_dir / f"{output_subdir}_kappa_distribution.png"
    if dist_path.exists():
        images["query_uncertainty_distribution"] = dist_path.resolve()
    return images


def collect_rejection_curve_images(
    output_dir: Path,
    output_subdir: Optional[str] = None,
    *,
    name_suffix: str = "",
) -> Dict[str, Path]:
    """Collect query-per-K and pairwise overlay rejection-curve PNGs."""
    extra = name_suffix or ""
    if output_subdir:
        stem = f"{output_subdir}_rejection_curves{extra}"
    else:
        stem = f"rejection_curves{extra}"
    images: Dict[str, Path] = {}
    for variant, suffix in REJECTION_CURVES_PLOT_SUFFIXES:
        path = output_dir / f"{stem}{suffix}.png"
        if path.exists():
            images[variant] = path.resolve()
    return images


def collect_rejection_curve_image(
    output_dir: Path,
    output_subdir: str,
) -> Optional[Path]:
    """Resolve main query-level rejection-curve PNG (combined views)."""
    path = output_dir / f"{output_subdir}_rejection_curves.png"
    return path.resolve() if path.exists() else None


def _hist_values_for_variant(variant: str, raw: Any) -> np.ndarray:
    """Return 1D values for histogram plotting for one metric variant."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if variant in PAIRWISE_SCORE_VARIANTS:
        return arr.reshape(-1)
    if arr.ndim >= 2 and arr.shape[0] > 0:
        return np.mean(arr, axis=tuple(range(1, arr.ndim))).reshape(-1)
    return arr.reshape(-1)


def _matched_top1_from_pool(
    panel: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(hits_with_gt, gt_mask)`` for top-1 recall labels.

    ``hits_with_gt`` excludes queries without ground-truth positives (same as R@K).
    ``gt_mask`` is the full-length boolean mask used to align per-query scores.
    """
    pool = panel.get("pool") or {}
    predictions = pool.get("predictions")
    positives_per_query = pool.get("positives_per_query")
    if predictions is None or positives_per_query is None:
        return None

    preds = np.asarray(predictions)
    if preds.ndim < 2:
        return None

    nq = preds.shape[0]
    if len(positives_per_query) < nq:
        return None

    from eval_metrics.recall import has_ground_truth_mask, recall_hits_at_k

    hits = recall_hits_at_k(preds, positives_per_query, n_rank=1)
    gt_mask = has_ground_truth_mask(positives_per_query, nq)
    return hits[gt_mask], gt_mask


def collect_uncertainty_stats_curve_images(
    output_dir: Path,
    panel: Dict[str, Any],
) -> Dict[str, Path]:
    """Create and collect per-metric uncertainty histogram PNGs.

    When ``panel['pool']`` is present (labels available), the histogram is split into
    top-1 *correct* vs *incorrect* queries.
    """
    raw_scores = panel.get("raw_scores") or {}
    if not raw_scores:
        return {}
    out_dir = output_dir / "uncertainty_stats_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    images: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return images

    matched_info = _matched_top1_from_pool(panel)
    matched_top1 = matched_info[0] if matched_info is not None else None
    gt_mask = matched_info[1] if matched_info is not None else None
    for variant in SCORE_STAT_VARIANTS:
        raw = raw_scores.get(variant)
        if raw is None:
            continue
        values = _hist_values_for_variant(variant, raw)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        label = variant

        split_correct_incorrect = matched_top1 is not None and gt_mask is not None
        if split_correct_incorrect:
            if variant in PAIRWISE_SCORE_VARIANTS:
                arr = np.asarray(raw, dtype=np.float64)
                count_per_query = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
                if gt_mask.size * count_per_query != values.size:
                    split_correct_incorrect = False
                else:
                    values = values[np.repeat(gt_mask, count_per_query)]
            else:
                if gt_mask.size != values.size:
                    split_correct_incorrect = False
                else:
                    values = values[gt_mask]
            if split_correct_incorrect and matched_top1.size != values.size:
                split_correct_incorrect = False

        fig, ax = plt.subplots(figsize=(8, 5))

        if not split_correct_incorrect:
            ax.hist(
                values,
                bins=60,
                alpha=0.85,
                color="steelblue",
                edgecolor="white",
                linewidth=0.3,
            )
            ax.set_title(f"{panel.get('dataset_name', 'dataset')} - {label}")
        else:
            correct_mask = matched_top1.astype(bool)
            if variant in PAIRWISE_SCORE_VARIANTS:
                arr = np.asarray(raw, dtype=np.float64)
                count_per_query = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
                # values already GT-filtered; matched is per-query — expand for pairs
                pair_mask = np.repeat(correct_mask, count_per_query)
                correct_vals = values[pair_mask]
                incorrect_vals = values[~pair_mask]
            else:
                correct_vals = values[correct_mask]
                incorrect_vals = values[~correct_mask]

            ax.hist(
                correct_vals,
                bins=60,
                alpha=0.7,
                color="seagreen",
                edgecolor="white",
                linewidth=0.3,
                label=f"correct (n={correct_vals.size})",
            )
            ax.hist(
                incorrect_vals,
                bins=60,
                alpha=0.7,
                color="indianred",
                edgecolor="white",
                linewidth=0.3,
                label=f"incorrect (n={incorrect_vals.size})",
            )
            ax.set_title(f"{panel.get('dataset_name', 'dataset')} - {label} (top-1 hit vs miss)")
            ax.legend(fontsize=8)

        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_path = out_dir / f"{variant}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        images[label] = out_path.resolve()
    return images


def _bins_distribution_log_keys(name: str, bins_images: Dict[str, Any]) -> Dict[str, Any]:
    """Single tiled key under ``bins_distribution/<dataset>``."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    prefix = "bins_distribution"
    ordered = [caption for caption, img in sorted(bins_images.items()) if img and Path(img).exists()]
    if ordered:
        payload[f"{prefix}/{name}_gallery"] = [
            _wandb.Image(str(Path(bins_images[caption])), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=bins_images,
        ordered_captions=ordered,
        output_path=(Path("/tmp") / f"bins_distribution_{name}_all_metrics.png"),
        title=f"bins_distribution/{name}",
        show_missing=False,
    )
    if tile and tile.exists():
        payload[f"{prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    return payload


def _ece_curves_panel_log_keys(
    name: str,
    curve_images: Dict[str, Any],
    *,
    panel_prefix: str,
    empty_message: str,
) -> Dict[str, Any]:
    """Tiled W&B image panel for one dataset/combined view."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    ordered = [
        caption
        for caption in ECE_CURVES_PANEL_VARIANTS
        if curve_images.get(caption) and Path(curve_images[caption]).exists()
    ]
    if ordered:
        payload[f"{panel_prefix}/{name}_gallery"] = [
            _wandb.Image(str(Path(curve_images[caption])), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=curve_images,
        ordered_captions=list(ECE_CURVES_PANEL_VARIANTS),
        output_path=(Path("/tmp") / f"{panel_prefix}_{name}_all_metrics.png"),
        title=f"{panel_prefix}/{name}",
        show_missing=True,
    )
    if tile and tile.exists():
        payload[f"{panel_prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    else:
        payload[f"{panel_prefix}/{name}"] = _wandb.Html(
            f"<div style='padding:12px;font-family:monospace;'>{empty_message}</div>"
        )
    return payload


def _ece_curves_log_keys(name: str, curve_images: Dict[str, Any]) -> Dict[str, Any]:
    """Single tiled key under ``ece_curves/<dataset>``."""
    return _ece_curves_panel_log_keys(
        name,
        curve_images,
        panel_prefix="ece_curves",
        empty_message="No ECE curve plots available",
    )


def _ece_curves_ap_log_keys(name: str, curve_images: Dict[str, Any]) -> Dict[str, Any]:
    """Single tiled key under ``ece_curves_ap/<dataset>``."""
    return _ece_curves_panel_log_keys(
        name,
        curve_images,
        panel_prefix=ECE_CURVES_AP_PANEL_PREFIX,
        empty_message="No ECE-AP curve plots available",
    )


def _rejection_curves_log_keys(
    name: str,
    image_paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Tiled W&B panel under ``rejection_curves/<dataset>`` (like ``ece_curves``)."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    prefix = REJECTION_CURVES_PANEL_PREFIX
    ordered = [
        caption
        for caption in REJECTION_CURVES_PANEL_VARIANTS
        if image_paths.get(caption) and Path(image_paths[caption]).exists()
    ]
    if ordered:
        payload[f"{prefix}/{name}_gallery"] = [
            _wandb.Image(str(image_paths[caption]), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=image_paths,
        ordered_captions=list(REJECTION_CURVES_PANEL_VARIANTS),
        output_path=(Path("/tmp") / f"{prefix}_{name}_all_metrics.png"),
        title=f"{prefix}/{name}",
        show_missing=True,
    )
    if tile and tile.exists():
        payload[f"{prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    else:
        payload[f"{prefix}/{name}"] = _wandb.Html(
            "<div style='padding:12px;font-family:monospace;'>"
            "No rejection curve plots available</div>"
        )
    return payload


def _uncertainty_stats_curves_log_keys(
    name: str,
    image_paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Tiled W&B panel under ``uncertainty_stats_curves/<dataset>``."""
    import wandb as _wandb

    payload: Dict[str, Any] = {}
    prefix = UNCERTAINTY_STATS_CURVES_PANEL_PREFIX
    ordered = [
        v
        for v in SCORE_STAT_VARIANTS
        if image_paths.get(v) and Path(image_paths[v]).exists()
    ]
    if ordered:
        payload[f"{prefix}/{name}_gallery"] = [
            _wandb.Image(str(image_paths[caption]), caption=caption) for caption in ordered
        ]
    tile = _render_image_tile(
        image_map=image_paths,
        ordered_captions=list(SCORE_STAT_VARIANTS),
        output_path=(Path("/tmp") / f"{prefix}_{name}_all_metrics.png"),
        title=f"{prefix}/{name}",
        show_missing=True,
    )
    if tile and tile.exists():
        payload[f"{prefix}/{name}"] = _wandb.Image(str(tile), caption="all_metrics")
    else:
        payload[f"{prefix}/{name}"] = _wandb.Html(
            "<div style='padding:12px;font-family:monospace;'>"
            "No uncertainty-stats histograms available</div>"
        )
    return payload


def _render_image_tile(
    *,
    image_map: Dict[str, Any],
    ordered_captions: Sequence[str],
    output_path: Path,
    title: str,
    show_missing: bool,
) -> Optional[Path]:
    """Render a multi-plot image tile from metric images."""
    if not ordered_captions:
        return None

    existing = [c for c in ordered_captions if image_map.get(c) and Path(image_map[c]).exists()]
    if not existing and not show_missing:
        return None

    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cols = min(3, max(1, math.ceil(math.sqrt(len(ordered_captions)))))
    rows = math.ceil(len(ordered_captions) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows))
    axes_arr = np.array(axes).reshape(-1)

    for i, caption in enumerate(ordered_captions):
        ax = axes_arr[i]
        img = image_map.get(caption)
        if img and Path(img).exists():
            ax.imshow(mpimg.imread(str(img)))
            ax.set_title(caption, fontsize=10)
            ax.axis("off")
        else:
            ax.set_title(caption, fontsize=10)
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(len(ordered_captions), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.resolve()


def _html_table(
    headers: List[str],
    rows: List[List[Any]],
    *,
    label_columns: int = 1,
) -> Any:
    """Compact HTML table (all rows visible; avoids W&B Table row pagination)."""
    import wandb as _wandb

    def _cell_text(cell: Any) -> str:
        if cell is None:
            return "—"
        if isinstance(cell, (float, np.floating)):
            return f"{float(cell):.3f}"
        return str(cell)

    style = "style='border-collapse:collapse;width:100%;font-family:monospace;font-size:14px;'"
    th_style = "style='border:1px solid #555;padding:6px;background:#2a2a2a;color:#eee;'"
    td_style = "style='border:1px solid #555;padding:6px;text-align:center;'"
    td_label = "style='border:1px solid #555;padding:6px;text-align:left;font-weight:bold;'"
    html = [f"<table {style}><tr>"]
    for h in headers:
        html.append(f"<th {th_style}>{h}</th>")
    html.append("</tr>")
    for row in rows:
        html.append("<tr>")
        for i, cell in enumerate(row):
            s = td_label if i < label_columns else td_style
            html.append(f"<td {s}>{_cell_text(cell)}</td>")
        html.append("</tr>")
    html.append("</table>")
    return _wandb.Html("".join(html))


# label_columns: leading columns rendered left-aligned (dataset / variant names).
_SUMMARY_TABLE_LABEL_COLS = {
    "retrieval": 1,
    "calibration_ece": 2,
    CALIBRATION_ECE_AP_PANEL_KEY: 2,
    REJECTION_CURVES_TABLE_KEY: 3,
    "descriptor_stats": 1,
    "uncertainty_stats": 2,
}





def _build_descriptor_stats_table(
    all_panel_data: List[Dict[str, Any]],
    combined_panels: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """DB+query learned uncertainty (κ / variance), one row per base dataset.

    When multiple query folders are evaluated for the same dataset, aggregate the
    descriptor stats by mean so the table stays one-row-per-dataset.
    """
    stat_keys = sorted({k for d in all_panel_data for k in d.get("variance_stat", {})})
    if not stat_keys:
        return None
    cols = ["dataset"] + stat_keys
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for d in all_panel_data:
        vs = d.get("variance_stat")
        if not vs:
            continue
        base_name = d.get("base_dataset_name") or d.get("dataset_name")
        grouped.setdefault(str(base_name), []).append(vs)

    rows: List[List[Any]] = []
    for dataset in sorted(grouped):
        stats_list = grouped[dataset]
        row_vals: List[Any] = []
        for k in stat_keys:
            vals = [float(s[k]) for s in stats_list if k in s and s[k] is not None]
            row_vals.append(float(np.mean(vals)) if vals else None)
        rows.append([dataset] + row_vals)

    # Keep descriptor_stats as per-dataset only (no combined_* rows).
    if not rows:
        return None
    return cols, rows


def _build_uncertainty_stats_table(
    all_panel_data: List[Dict[str, Any]],
    combined_panels: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """Per-metric score stats used for ECE (kappa, baselines, …)."""
    entries: List[Tuple[str, str, Dict[str, float]]] = []

    for d in all_panel_data:
        dataset = d["dataset_name"]
        for variant in SCORE_STAT_VARIANTS:
            raw = d.get("raw_scores", {}).get(variant)
            if raw is None:
                continue
            stats = compute_score_statistics(raw)
            if stats:
                entries.append((dataset, variant, stats))

    for d in combined_panels or []:
        dataset = d.get("dataset_name")
        for variant in SCORE_STAT_VARIANTS:
            raw = d.get("raw_scores", {}).get(variant)
            if raw is None:
                continue
            stats = compute_score_statistics(raw)
            if stats:
                entries.append((dataset, variant, stats))

    if not entries:
        return None

    stat_keys = sorted({k for _, _, stats in entries for k in stats})
    cols = ["dataset", "metric"] + stat_keys
    rows = [
        [dataset, metric] + [stats.get(k) for k in stat_keys]
        for dataset, metric, stats in entries
    ]
    return cols, rows


def _metric_summary_rows(
    panel: Dict[str, Any],
    metric: str,
    columns: Sequence[str],
) -> List[List[Any]]:
    """One row per dataset, or two labeled rows when combined dual-norm is present."""
    name = panel.get("dataset_name", "")
    values = panel.get(metric) or {}
    per_dataset = panel.get(f"{metric}_norm_per_dataset") or {}
    if per_dataset:
        rows: List[List[Any]] = []
        if values:
            rows.append([f"{name} (none)"] + [values.get(k) for k in columns])
        rows.append(
            [f"{name} (per_dataset)"] + [per_dataset.get(k) for k in columns]
        )
        return rows
    return [[name] + [values.get(k) for k in columns]]


def _dual_norm_groups(
    panel: Dict[str, Any],
    base_key: str,
) -> List[tuple]:
    """``[(dataset_label, values_dict), ...]`` for none / per-dataset combined keys."""
    name = panel.get("dataset_name", "")
    base_vals = panel.get(base_key) or {}
    pd_vals = panel.get(f"{base_key}_norm_per_dataset")
    if pd_vals:
        groups = []
        if base_vals:
            groups.append((f"{name} (none)", base_vals))
        groups.append((f"{name} (per_dataset)", pd_vals))
        return groups
    return [(name, base_vals)]


def _build_eval_summary_tables(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[List[str], List[List[Any]]]]:
    """Shared columns/rows for wandb.Table and HTML summary panels."""
    rv_sorted = sorted(cfg.get("recall_values", [1, 5, 10, 20]))
    out: Dict[str, Tuple[List[str], List[List[Any]]]] = {}

    combined_panels: List[Dict[str, Any]] = [
        combined.get("panel", {})
        for combined in (combined_outputs or [])
        if combined.get("panel")
    ]
    table_panels: List[Dict[str, Any]] = list(all_panel_data) + combined_panels

    cols = ["dataset"] + [f"R@{k}" for k in rv_sorted] + [f"mAP@{k}" for k in rv_sorted]
    rows = [
        [d["dataset_name"]]
        + [d.get("recalls", {}).get(k) for k in rv_sorted]
        + [d.get("map_at_k", {}).get(k) for k in rv_sorted]
        for d in table_panels
    ]
    if rows:
        out["retrieval"] = (cols, rows)

    cols = ["dataset", "variant"] + [f"ECE_R@{k}" for k in rv_sorted]
    rows = []
    for d in table_panels:
        for dataset_label, ece_by_variant in _dual_norm_groups(d, "ece"):
            for variant in CALIBRATION_ECE_VARIANTS:
                rec = _ece_field(ece_by_variant.get(variant), "ece_recall") or {}
                if not rec:
                    continue
                rows.append(
                    [dataset_label, variant]
                    + [rec.get(k, rec.get(int(k))) for k in rv_sorted]
                )
    if rows:
        out["calibration_ece"] = (cols, rows)

    cols = ["dataset", "variant", "ECE_AP"]
    rows = []
    for d in table_panels:
        for dataset_label, ece_by_variant in _dual_norm_groups(d, "ece"):
            for variant in CALIBRATION_ECE_VARIANTS:
                val = _ece_field(ece_by_variant.get(variant), "ece_ap")
                if val is None:
                    continue
                rows.append([dataset_label, variant, float(val)])
    if rows:
        out[CALIBRATION_ECE_AP_PANEL_KEY] = (cols, rows)

    pcts: List[int] = []
    seen = set()
    for d in table_panels:
        for _, rejection in _dual_norm_groups(d, "rejection_curves"):
            for p in rejection_delta_pcts_from_stats(rejection or {}):
                if p not in seen:
                    seen.add(p)
                    pcts.append(p)
    pcts = sorted(pcts)
    variant_order = {v: i for i, v in enumerate(REJECTION_CURVES_METHOD_ORDER)}
    cols = (
        ["dataset", "variant", "n_rank"]
        + [f"delta_at_{p}" for p in pcts]
        + ["correctness_at_0"]
        + [f"correctness_at_{p}" for p in pcts]
    )
    rows = []
    for d in table_panels:
        for dataset_label, rejection in _dual_norm_groups(d, "rejection_curves"):
            for variant, stats in (rejection or {}).items():
                if not isinstance(stats, dict):
                    continue
                method = str(stats.get("method") or variant)
                if method == "sue_log" or str(variant) == "sue_log":
                    continue
                n_rank = stats.get("n_rank")
                row: List[Any] = [dataset_label, method, n_rank]
                for p in pcts:
                    val = stats.get(f"delta_at_{p}")
                    if val is None and p == 90:
                        val = stats.get("improvement")
                    row.append(val)
                row.append(stats.get("correctness_at_0", stats.get("pair_hit_rate_at_0")))
                for p in pcts:
                    row.append(
                        stats.get(f"correctness_at_{p}", stats.get(f"pair_hit_rate_at_{p}"))
                    )
                rows.append(row)
    if rows:
        rows.sort(
            key=lambda r: (
                str(r[0]),
                variant_order.get(str(r[1]), len(variant_order)),
                str(r[1]),
                int(r[2] or 0),
            )
        )
        out[REJECTION_CURVES_TABLE_KEY] = (cols, rows)

    descriptor_stats = _build_descriptor_stats_table(all_panel_data, combined_panels)
    if descriptor_stats is not None:
        out["descriptor_stats"] = descriptor_stats

    uncertainty_stats = _build_uncertainty_stats_table(all_panel_data, combined_panels)
    if uncertainty_stats is not None:
        out["uncertainty_stats"] = uncertainty_stats

    return out


def build_eval_summary_html_panels(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """HTML summaries for all cross-dataset tables (no row pagination)."""
    panels: Dict[str, Any] = {}
    for key, (cols, rows) in _build_eval_summary_tables(
        cfg, all_panel_data, combined_outputs
    ).items():
        panels[f"{key}/summary_html"] = _html_table(
            cols,
            rows,
            label_columns=_SUMMARY_TABLE_LABEL_COLS.get(key, 1),
        )
    return panels


def collect_dataset_plot_media(cfg: Dict[str, Any], panel: Dict[str, Any]) -> Dict[str, Any]:
    """W&B image panels for one dataset (ECE / bins / rejection / uncertainty stats)."""
    name = str(panel.get("dataset_name") or "")
    out_dir = Path(cfg["log_dir"]) / "eval" / name
    media: Dict[str, Any] = {}
    media.update(_bins_distribution_log_keys(name, collect_bins_distribution_images(out_dir)))
    media.update(_ece_curves_log_keys(name, collect_ece_curve_images(out_dir)))
    curve_ap_images = collect_ece_curve_ap_images(out_dir)
    if curve_ap_images:
        media.update(_ece_curves_ap_log_keys(name, curve_ap_images))
    media.update(_rejection_curves_log_keys(name, collect_rejection_curve_images(out_dir)))
    media.update(
        _uncertainty_stats_curves_log_keys(
            name, collect_uncertainty_stats_curve_images(out_dir, panel)
        )
    )
    return media


def collect_combined_plot_media(combined: Dict[str, Any]) -> Dict[str, Any]:
    """W&B image panels for one combined-eval variant."""
    name = str(combined["name"])
    out_dir = Path(combined["bins_output_dir"])
    media: Dict[str, Any] = {}
    media.update(
        _bins_distribution_log_keys(
            name, collect_combined_bins_distribution_images(out_dir, name)
        )
    )
    media.update(_ece_curves_log_keys(name, collect_ece_curve_images(out_dir)))
    curve_ap_images = collect_ece_curve_ap_images(out_dir)
    if curve_ap_images:
        media.update(_ece_curves_ap_log_keys(name, curve_ap_images))
    media.update(
        _rejection_curves_log_keys(name, collect_rejection_curve_images(out_dir, name))
    )
    pd_rejection = collect_rejection_curve_images(
        out_dir, name, name_suffix="_per_dataset"
    )
    if pd_rejection:
        media.update(_rejection_curves_log_keys(f"{name}_per_dataset", pd_rejection))
    media.update(
        _uncertainty_stats_curves_log_keys(
            name, collect_uncertainty_stats_curve_images(out_dir, combined["panel"])
        )
    )
    return media


def build_eval_panel_payload(
    cfg: Dict[str, Any],
    all_panel_data: List[Dict[str, Any]],
    combined_outputs: Optional[List[Dict[str, Any]]] = None,
    *,
    include_dataset_media: bool = True,
) -> Dict[str, Any]:
    """Build batched W&B eval panels (summary tables + plot galleries)."""
    import wandb as _wandb

    tables: Dict[str, Any] = {}
    media: Dict[str, Any] = {}

    for key, (cols, rows) in _build_eval_summary_tables(
        cfg, all_panel_data, combined_outputs
    ).items():
        tables[f"{key}/table"] = _wandb.Table(columns=cols, data=rows)

    if include_dataset_media:
        for panel in all_panel_data:
            media.update(collect_dataset_plot_media(cfg, panel))
    for combined in combined_outputs or []:
        media.update(collect_combined_plot_media(combined))

    return {**tables, **media}


def build_eval_outputs(
    dataset_name: str,
    args: Dict[str, Any],
    recalls,
    map_at_k,
    uncertainty_ece_results,
    uncertainty_raw_scores,
    baseline_results,
    uncertainty_stats,
    dataset_output_dir: Optional[Path] = None,
    save_plots: bool = False,
    base_dataset_name: Optional[str] = None,
    query_folder_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Build W&B panel data and image paths for one dataset eval."""
    panel_data = _build_panel_data(
        dataset_name,
        args,
        recalls,
        map_at_k,
        uncertainty_ece_results,
        uncertainty_raw_scores,
        baseline_results,
        uncertainty_stats,
        base_dataset_name=base_dataset_name,
        query_folder_name=query_folder_name,
    )

    wandb_images = None
    if dataset_output_dir is not None:
        wandb_images = collect_wandb_images(dataset_name, dataset_output_dir, args, save_plots)

    return panel_data, wandb_images
