"""Combined multi-dataset eval: concatenate shards, then the per-dataset protocol."""
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval_metrics.eval_ece_sh import (
    mean_query_uncertainty,
    scores_to_uncertainty,
    vmf_like_uncertainty_loss,
)
from eval_metrics.eval_tasks import (
    is_pairwise_rank_flat_key,
    resolve_eval_metrics,
)
from eval_metrics.score_scaling import percentile_minmax_to_unit_interval
from eval_metrics.log_utils import log_combined_panel
from eval_metrics.method_eval import (
    ece_extras_enabled,
    evaluate_method,
    merge_method_eval,
)
from eval_metrics.recall import (
    cal_recall,
    recall_hits_at_k,
)
from eval_metrics.rejection_curves import (
    plot_rejection_curve_sets,
)
from eval_metrics.uncertainty import compute_score_statistics
from eval_metrics.scores.joint_kappa import per_example_mean
from eval_metrics.scores import (
    BASELINE_QUERY_SCORE_NAMES,
    HYBRID_QUERY_SCORE_NAMES,
    LEARNED_QUERY_SCORE_NAMES,
    MethodScore,
    method_score,
)
from eval_metrics.scores.baselines import skip_sue_dataset

logger = logging.getLogger(__name__)


def _combined_shards_from_panels(
    all_panel_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build combined-eval shard dicts from per-dataset panel rows."""
    shards: List[Dict[str, Any]] = []
    for panel in all_panel_data:
        pool = panel.get("pool")
        if not pool:
            continue
        shards.append({
            "name": panel["dataset_name"],
            **pool,
            "raw_scores": panel.get("raw_scores", {}),
            "base_dataset_name": panel.get("base_dataset_name", ""),
            "query_folder_name": panel.get("query_folder_name"),
        })
    return shards


def _shard_base_name(shard: Dict[str, Any]) -> str:
    return (shard.get("base_dataset_name") or shard.get("name") or "").lower()


def _is_sf_xl(shard: Dict[str, Any]) -> bool:
    bn = _shard_base_name(shard)
    return "sf_xl" in bn or "sf-xl" in bn


def _is_pitts30(shard: Dict[str, Any]) -> bool:
    return "pitts30" in _shard_base_name(shard)


def _is_msls_query(shard: Dict[str, Any]) -> bool:
    return "msls" in _shard_base_name(shard) and shard.get("query_folder_name") == "query"


def _is_amstertime(shard: Dict[str, Any]) -> bool:
    return "amstertime" in _shard_base_name(shard)


def _matches_core_four(shard: Dict[str, Any]) -> bool:
    """all sf-xl shards, msls-val query, pitts30k."""
    return _is_sf_xl(shard) or _is_pitts30(shard) or _is_msls_query(shard)


def _matches_core_five(shard: Dict[str, Any]) -> bool:
    return _matches_core_four(shard) or _is_amstertime(shard)


# Weather splits only. The mixed ``queries`` folder is a separate shard.
_SVOX_CONDITION_FOLDERS = frozenset({
    "queries_night",
    "queries_overcast",
    "queries_rain",
    "queries_snow",
    "queries_sun",
})


def _is_svox_condition(shard: Dict[str, Any]) -> bool:
    return (
        "svox" in _shard_base_name(shard)
        and shard.get("query_folder_name") in _SVOX_CONDITION_FOLDERS
    )


def _matches_core_and_svox_conditions(shard: Dict[str, Any]) -> bool:
    """combined_6 shards plus the five SVOX weather splits. Not SVOX ``queries``."""
    return _matches_core_four(shard) or _is_svox_condition(shard)


def _new_combined_panel(output_subdir: str) -> Dict[str, Any]:
    return {
        "dataset_name": output_subdir,
        "recalls": {},
        "map_at_k": {},
        "ece": {},
        "ece_norm_per_dataset": {},
        "raw_scores": {},
        "variance_stat": {},
    }


def _use_combined_method(
    raw_scores: Dict[str, Any],
    key: str,
    output_subdir: str,
) -> bool:
    """Skip a method unless every dataset in the combined pool has its scores."""
    if raw_scores.get(key) is not None:
        return True
    logger.info(
        "%s: skipping combined %s (missing on at least one dataset).",
        output_subdir,
        key,
    )
    return False


def _stack_raw_score(arrays: List[Any]) -> np.ndarray:
    """Concatenate per-dataset scores along the query axis (1D) or stack pair matrices (2D)."""
    chunks = [np.asarray(a, dtype=np.float64) for a in arrays]
    if all(a.ndim <= 1 for a in chunks):
        return np.concatenate([a.reshape(-1) for a in chunks])
    return np.vstack([a if a.ndim > 1 else a[:, None] for a in chunks])


def _offset_concat_retrieval(
    shards: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    pred_chunks: List[np.ndarray] = []
    dist_chunks: List[np.ndarray] = []
    positives: List[np.ndarray] = []
    db_offset = 0
    for d in shards:
        preds = np.asarray(d["predictions"])
        dists = np.asarray(d["distances"])
        max_db = int(preds.max()) + 1 if preds.size else 0
        pred_chunks.append(preds + db_offset)
        dist_chunks.append(dists)
        for pos in d["positives_per_query"]:
            positives.append(np.asarray(pos) + db_offset)
        db_offset += max_db
    return np.vstack(pred_chunks), np.vstack(dist_chunks), positives


def _concat_combined_pool(shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate shards along the query axis."""
    predictions, distances, positives = _offset_concat_retrieval(shards)
    # Methods that exist on every shard (e.g. skip SUE if one dataset has no UTMs).
    keys = set.intersection(*(set(d.get("raw_scores") or {}) for d in shards))
    raw_scores: Dict[str, np.ndarray] = {}
    for key in keys:
        if is_pairwise_rank_flat_key(key):
            continue
        arrs = [(d.get("raw_scores") or {}).get(key) for d in shards]
        if any(a is None for a in arrs):
            continue
        raw_scores[key] = _stack_raw_score(arrs)
    pooled: Dict[str, Any] = {
        "predictions": predictions,
        "distances": distances,
        "positives_per_query": positives,
        "raw_scores": raw_scores,
    }
    pair_keys = set.intersection(
        *(set(d.get("pair_scores") or {}) for d in shards)
    )
    if pair_keys:
        pooled["pair_scores"] = {
            name: _stack_raw_score([d["pair_scores"][name] for d in shards])
            for name in pair_keys
        }
    return pooled


def _pooled_pair_matrix(pooled: Dict[str, Any], name: str) -> Optional[np.ndarray]:
    """``[N_q, K]`` pair scores for one method on the concatenated pool."""
    stacked = (pooled.get("pair_scores") or {}).get(name)
    if stacked is None:
        return None
    pair = np.asarray(stacked, dtype=np.float64)
    preds = np.asarray(pooled["predictions"])
    if pair.shape != preds.shape:
        raise ValueError(
            f"combined {name} pair_scores {tuple(pair.shape)} != "
            f"predictions {tuple(preds.shape)}"
        )
    return pair


def _methods_from_pooled_shards(
    pooled: Dict[str, Any],
    cfg,
    output_subdir: str,
) -> List[MethodScore]:
    """Wrap concatenated shard scores as the same MethodScore list used per-dataset."""
    raw = pooled["raw_scores"]
    uncertainty_loss = cfg.get("uncertainty_loss", "gaussian_nll")
    is_vmf = str(uncertainty_loss).lower() == "vmf"
    floor = bool(cfg.get("ece_vmf_kappa_floor", False)) if is_vmf else False
    two_sided = bool(cfg.get("ece_two_sided", True))

    methods = []
    if _use_combined_method(raw, "kappa", output_subdir):
        methods.append(
            method_score(
                "kappa",
                raw["kappa"],
                higher_is_confidence=is_vmf,
                uncertainty_loss=uncertainty_loss,
                use_vmf_kappa_floor=floor,
                percentile_two_sided=two_sided,
            )
        )
    if is_vmf:
        for name in LEARNED_QUERY_SCORE_NAMES:
            if not _use_combined_method(raw, name, output_subdir):
                continue
            methods.append(
                method_score(
                    name,
                    query_scores=raw[name],
                    higher_is_confidence=True,
                    uncertainty_loss="vmf",
                    pair_scores=_pooled_pair_matrix(pooled, name),
                    use_vmf_kappa_floor=floor,
                    percentile_two_sided=two_sided,
                )
            )
    if cfg.get("skip_baselines"):
        return methods
    for name in BASELINE_QUERY_SCORE_NAMES:
        if not _use_combined_method(raw, name, output_subdir):
            continue
        methods.append(
            method_score(
                name,
                raw[name],
                higher_is_confidence=False,
                uncertainty_loss="gaussian_nll",
                pair_scores=_pooled_pair_matrix(pooled, name),
                percentile_two_sided=False,
            )
        )
    for name in HYBRID_QUERY_SCORE_NAMES:
        if not _use_combined_method(raw, name, output_subdir):
            continue
        methods.append(
            method_score(
                name,
                raw[name],
                higher_is_confidence=False,
                uncertainty_loss="gaussian_nll",
                percentile_two_sided=True,
            )
        )
    return methods


def _minmax_by_shard_counts(arr: np.ndarray, counts: List[int]) -> np.ndarray:
    """p1–p99 stretch independently on each dataset shard (rows)."""
    arr = np.asarray(arr, dtype=np.float64)
    if int(arr.shape[0]) != int(sum(counts)):
        raise ValueError(
            f"shard counts sum {sum(counts)} != concatenated rows {arr.shape[0]}"
        )
    parts = []
    start = 0
    for n in counts:
        parts.append(percentile_minmax_to_unit_interval(arr[start:start + n]))
        start += n
    if arr.ndim == 1:
        return np.concatenate(parts)
    return np.vstack(parts)


def _shard_query_counts(shards: List[Dict[str, Any]]) -> List[int]:
    return [int(np.asarray(d["predictions"]).shape[0]) for d in shards]


def _per_dataset_norm_scores(
    method: MethodScore,
    counts: List[int],
    *,
    invert: str = "reciprocal",
) -> Dict[str, Any]:
    """Convert to uncertainty, then p1–p99 min-max per original dataset.

    Does not flatten or score. ``query_norm`` / ``pair_norm`` are the unit copies
    used by ECE and rejection.
    """
    query_u = np.asarray(
        mean_query_uncertainty(
            method.query_scores,
            method.uncertainty_loss,
            method.use_vmf_kappa_floor,
            invert=invert,
        ),
        dtype=np.float64,
    ).reshape(-1)
    query_norm = _minmax_by_shard_counts(query_u, counts)
    pair_norm = None
    pair = method.pair_scores
    if pair is not None:
        pair_u = scores_to_uncertainty(
            pair,
            method.uncertainty_loss,
            method.use_vmf_kappa_floor,
            invert=invert,
        )
        pair_norm = _minmax_by_shard_counts(np.asarray(pair_u, dtype=np.float64), counts)
    return {
        "query": query_u,
        "query_norm": query_norm,
        "pair_norm": pair_norm,
    }


def _as_unit_uncertainty_method(
    method: MethodScore,
    query_norm: np.ndarray,
    *,
    pair_scores: Optional[np.ndarray] = None,
    name_suffix: str = "",
    log_name: Optional[str] = None,
) -> MethodScore:
    """Reuse ``method`` with query (and optional pair) scores already uncertainty in ``[0, 1]``.

    Does not min-max or flatten; ``evaluate_method`` flattens ``pair_scores`` if present.
    """
    pw_name = method.pairwise_name if pair_scores is not None else None
    pw_plot = method.pairwise_plot_filename if pair_scores is not None else None
    if name_suffix:
        if pw_name:
            pw_name = f"{pw_name}{name_suffix}"
        if pw_plot:
            pw_plot = pw_plot.replace(".png", f"{name_suffix}.png")
    return replace(
        method,
        name=f"{method.name}{name_suffix}" if name_suffix else method.name,
        log_name=log_name or method.log_name,
        query_scores=query_norm,
        pair_scores=pair_scores,
        pairwise_name=pw_name,
        pairwise_plot_filename=pw_plot,
        higher_is_confidence=False,
        uncertainty_loss="gaussian_nll",
        use_vmf_kappa_floor=False,
    )


def compute_and_log_combined(
    cfg,
    agg_data_list,
    output_subdir,
) -> Optional[Dict[str, Any]]:
    """Concatenate all datasets, then run the same ``evaluate_method`` protocol as per-dataset."""
    pooled = _concat_combined_pool(agg_data_list)
    predictions = pooled["predictions"]
    distances = pooled["distances"]
    positives = pooled["positives_per_query"]
    methods = _methods_from_pooled_shards(pooled, cfg, output_subdir)

    combined_output_dir = Path(cfg["log_dir"]) / "eval" / output_subdir
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    combined_panel = _new_combined_panel(output_subdir)
    eval_metrics = resolve_eval_metrics(cfg, training=False)
    ece_metrics = cfg.get("ece_metrics") or ["recall"]
    extras_on = ece_extras_enabled(
        cfg.get("ece_bin_mode", "percentile"), cfg.get("ece_extras", False)
    )
    eval_kwargs = dict(
        n_values=cfg["recall_values"],
        output_dir=combined_output_dir,
        ece_metrics=ece_metrics,
        zoom_threshold=cfg.get("ece_zoom_threshold", 0.001),
        bin_mode=cfg.get("ece_bin_mode", "percentile"),
        eval_metrics=list(eval_metrics),
        rejection_delta_pcts=cfg.get("rejection_delta_pcts"),
        ece_extras=extras_on,
    )

    counts = _shard_query_counts(agg_data_list)
    pd_eval_kwargs = dict(eval_kwargs)
    pd_eval_kwargs["ece_plot_suffix"] = "_per_dataset"
    pd_eval_kwargs["log_summary"] = False

    rejection_query_curves: List[Dict[str, Any]] = []
    rejection_pair_curves: List[Dict[str, Any]] = []
    pd_rejection_query_curves: List[Dict[str, Any]] = []
    pd_rejection_pair_curves: List[Dict[str, Any]] = []
    for method in methods:
        one = evaluate_method(
            method,
            predictions,
            positives,
            distances,
            **eval_kwargs,
        )
        merge_method_eval(combined_panel, one)
        rej = one.get("rejection") or {}
        rejection_query_curves.extend(rej.get("query_curves") or [])
        rejection_pair_curves.extend(rej.get("pair_curves") or [])

        pd_scores = _per_dataset_norm_scores(method, counts)
        pd_method = _as_unit_uncertainty_method(
            method, pd_scores["query_norm"], pair_scores=pd_scores["pair_norm"]
        )
        one_pd = evaluate_method(
            pd_method,
            predictions,
            positives,
            distances,
            **pd_eval_kwargs,
        )
        merge_method_eval(combined_panel, one_pd, per_dataset=True)
        rej = one_pd.get("rejection") or {}
        pd_rejection_query_curves.extend(rej.get("query_curves") or [])
        pd_rejection_pair_curves.extend(rej.get("pair_curves") or [])
        if (
            extras_on
            and "ece" in eval_metrics
            and method.higher_is_confidence
            and vmf_like_uncertainty_loss(method.uncertainty_loss)
        ):
            neg_scores = _per_dataset_norm_scores(method, counts, invert="neg")
            neg_method = _as_unit_uncertainty_method(
                method,
                neg_scores["query_norm"],
                pair_scores=neg_scores["pair_norm"],
                name_suffix="_neg",
                log_name=f"{method.log_name} (−κ)",
            )
            one_pd_neg = evaluate_method(
                neg_method,
                predictions,
                positives,
                distances,
                **{**pd_eval_kwargs, "eval_metrics": ["ece"]},
            )
            merge_method_eval(combined_panel, one_pd_neg, per_dataset=True)

    if len(positives) > 0:
        orig_rec = cal_recall(
            predictions,
            positives,
            cfg["recall_values"],
            drop_empty_gt=bool(cfg.get("drop_empty_gt", False)),
        )
        combined_panel["recalls"] = {
            int(k_val): float(v) for k_val, v in zip(cfg["recall_values"], orig_rec)
        }

    q_kappa = pooled["raw_scores"].get("kappa")
    if q_kappa is not None:
        combined_panel["variance_stat"] = compute_score_statistics(q_kappa)
    combined_panel["pool"] = {
        "predictions": predictions,
        "positives_per_query": positives,
        "distances": distances,
    }

    if (
        "rejection" in eval_metrics
        and not cfg.get("skip_rejection_curves")
        and (rejection_query_curves or rejection_pair_curves)
    ):
        rejection_path = combined_output_dir / f"{output_subdir}_rejection_curves.png"
        saved_path, pw_path = plot_rejection_curve_sets(
            rejection_query_curves,
            rejection_pair_curves,
            rejection_path,
            title=f"{output_subdir} rejection curves (none)",
        )
        if saved_path is not None:
            logger.info("Combined rejection curves saved to %s", saved_path)
        if pw_path is not None:
            logger.info("Pairwise rejection curves saved to %s", pw_path)
    if (
        "rejection" in eval_metrics
        and not cfg.get("skip_rejection_curves")
        and (pd_rejection_query_curves or pd_rejection_pair_curves)
    ):
        pd_path = combined_output_dir / f"{output_subdir}_rejection_curves_per_dataset.png"
        saved_path, pw_path = plot_rejection_curve_sets(
            pd_rejection_query_curves,
            pd_rejection_pair_curves,
            pd_path,
            title=f"{output_subdir} rejection curves (per_dataset)",
        )
        if saved_path is not None:
            logger.info("Combined per-dataset-norm rejection curves saved to %s", saved_path)
        if pw_path is not None:
            logger.info("Combined per-dataset-norm pairwise rejection curves saved to %s", pw_path)

    log_combined_panel(output_subdir, combined_panel)

    if len(positives) > 0:
        hits = recall_hits_at_k(predictions, positives, n_rank=1)
        logger.info(
            "%s: top-1 positive rate (combined) = %.4f (random classifier ≈ this area under PR)",
            output_subdir,
            float(np.mean(hits)),
        )

    try:
        import matplotlib.pyplot as plt
        with plt.style.context("default"):
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            ax = axs[0]
            for d in agg_data_list:
                ds_mean_var = per_example_mean(d["q_var"])
                ds_label = d.get("name", "unknown")
                ax.hist(
                    ds_mean_var,
                    bins=50,
                    alpha=0.5,
                    label=f"{ds_label} (n={len(ds_mean_var)})",
                    edgecolor="black",
                    linewidth=0.3,
                )
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.set_title("Per-dataset query uncertainty distribution")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axs[1]
            if q_kappa is not None:
                combined_mean_var = np.asarray(q_kappa).reshape(-1)
                ax.hist(
                    combined_mean_var,
                    bins=50,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.axvline(
                    np.median(combined_mean_var),
                    color="tomato",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Median = {np.median(combined_mean_var):.4f}",
                )
                ax.axvline(
                    np.mean(combined_mean_var),
                    color="orange",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Mean = {np.mean(combined_mean_var):.4f}",
                )
                ax.set_title(f"Combined query uncertainty (n={len(combined_mean_var)})")
                ax.legend(fontsize=8)
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            dist_path = combined_output_dir / f"{output_subdir}_kappa_distribution.png"
            plt.savefig(dist_path, dpi=150)
            plt.close(fig)
        logger.info("Combined kappa distribution plot saved to %s", dist_path)
    except ImportError:
        logger.warning("matplotlib not installed, skipping combined kappa distribution plot.")
    except Exception as e:
        logger.warning("Error saving combined kappa distribution plot: %s", e)

    logger.info(
        "%s: combined metrics computed over %d queries from %d datasets.",
        output_subdir,
        len(positives),
        len(agg_data_list),
    )

    return {
        "name": output_subdir,
        "panel": combined_panel,
        "bins_output_dir": combined_output_dir,
    }



def _maybe_run_combined(
    cfg,
    shards: List[Dict[str, Any]],
    *,
    output_subdir: str,
    combined_outputs: List[Dict[str, Any]],
    description: str,
    min_shards: int = 2,
    required_predicate=None,
) -> None:
    if len(shards) < min_shards:
        return
    if required_predicate is not None and not required_predicate(shards):
        return
    logger.info("=" * 30 + "\nComputing %s (%s)...", output_subdir, description)
    out = compute_and_log_combined(cfg, shards, output_subdir)
    if out:
        combined_outputs.append(out)


def _filter_shards_all(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return agg_list


def _filter_shards_sue_ok(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if not skip_sue_dataset(_shard_base_name(d))]


def _filter_shards_no_amstertime(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if not _is_amstertime(d)]


def _filter_shards_no_pitts30(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if not _is_pitts30(d)]


def _filter_shards_core_five(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if _matches_core_five(d)]


def _filter_shards_core_four(agg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in agg_list if _matches_core_four(d)]


def _filter_shards_core_and_svox_conditions(
    agg_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [d for d in agg_list if _matches_core_and_svox_conditions(d)]


def _has_core_four_requirements(shards: List[Dict[str, Any]]) -> bool:
    """Require at least one sf-xl shard, msls(query), and pitts30k shard."""
    return (
        any(_is_sf_xl(s) for s in shards)
        and any(_is_msls_query(s) for s in shards)
        and any(_is_pitts30(s) for s in shards)
    )


def _has_core_five_requirements(shards: List[Dict[str, Any]]) -> bool:
    """Core four requirements plus at least one amstertime shard."""
    return _has_core_four_requirements(shards) and any(_is_amstertime(s) for s in shards)


def _has_svox_condition_requirements(shards: List[Dict[str, Any]]) -> bool:
    """Core four requirements plus all five SVOX weather splits."""
    folders = {s.get("query_folder_name") for s in shards if _is_svox_condition(s)}
    return _has_core_four_requirements(shards) and _SVOX_CONDITION_FOLDERS <= folders


# Each view pools the same shard set for kappa, baselines (l2/pa/sue/sue_log), and vMF extras.
_DEFAULT_COMBINED_VIEWS = (
    {
        "output_subdir": "combined_6",
        "description": "all sf-xl shards, msls-val query, pitts30k",
        "filter_fn": _filter_shards_core_four,
        "required_predicate": _has_core_four_requirements,
    },
    {
        "output_subdir": "combined_11",
        "description": (
            "all sf-xl shards, msls-val query, pitts30k, "
            "svox night/sun/snow/rain/overcast (not svox queries)"
        ),
        "filter_fn": _filter_shards_core_and_svox_conditions,
        "required_predicate": _has_svox_condition_requirements,
    },
)

# Enabled with --combined_extras.
_EXTRA_COMBINED_VIEWS = (
    {
        "output_subdir": "combined_all",
        "description": "all evaluated shards except Nordland / AmsterTime / SPED (SUE-incompatible)",
        "filter_fn": _filter_shards_sue_ok,
    },
    {
        "output_subdir": "combined_all_full",
        "description": "all evaluated shards, including Nordland / AmsterTime / SPED",
        "filter_fn": _filter_shards_all,
    },
    {
        "output_subdir": "combined_all_no_amst",
        "description": "all evaluated shards without amstertime",
        "filter_fn": _filter_shards_no_amstertime,
    },
    {
        "output_subdir": "combined_all_no_pitts",
        "description": "all evaluated shards without pitts30k",
        "filter_fn": _filter_shards_no_pitts30,
    },
    {
        "output_subdir": "combined_7",
        "description": "all sf-xl shards, msls-val query, pitts30k, amstertime",
        "filter_fn": _filter_shards_core_five,
        "required_predicate": _has_core_five_requirements,
    },
)


def _iter_combined_views(cfg) -> Tuple[Dict[str, Any], ...]:
    views = list(_DEFAULT_COMBINED_VIEWS)
    if cfg.get("combined_extras"):
        views.extend(_EXTRA_COMBINED_VIEWS)
    return tuple(views)


def run_combined_eval_variants(cfg, all_panel_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run combined views; every selected eval metric uses the same shards per view.

    Default:
      1. combined_6 — all sf-xl shards + msls-val(query) + pitts30k
      2. combined_11 — combined_6 plus the five SVOX weather splits
         (night, sun, snow, rain, overcast). The mixed SVOX ``queries``
         folder is not included. Skipped unless all five splits are present.

    With --combined_extras:
      3. combined_all — all evaluated shards except Nordland / AmsterTime / SPED
      4. combined_all_full — all evaluated shards (original combined_all)
      5. combined_all_no_amst — all shards, minus amstertime
      6. combined_all_no_pitts — all shards, minus pitts30k
      7. combined_7 — all sf-xl shards + msls-val(query) + pitts30k + amstertime
    """
    combined_outputs: List[Dict[str, Any]] = []
    if cfg.get("skip_combined"):
        logger.info("skip_combined: skipping combined eval views.")
        return combined_outputs
    agg_list = _combined_shards_from_panels(all_panel_data)
    if len(agg_list) <= 1:
        logger.info("Only 1 dataset evaluated; skipping combined eval.")
        return combined_outputs
    if cfg["model_mode"] != "uncertainty":
        return combined_outputs

    for view in _iter_combined_views(cfg):
        shards = view["filter_fn"](agg_list)
        _maybe_run_combined(
            cfg,
            shards,
            output_subdir=view["output_subdir"],
            combined_outputs=combined_outputs,
            description=view["description"],
            required_predicate=view.get("required_predicate"),
        )

    return combined_outputs
