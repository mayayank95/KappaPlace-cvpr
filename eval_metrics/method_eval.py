"""Per-method uncertainty evals (ECE, rejection) on precomputed scores."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_metrics.eval_ece_sh import (
    compute_ece,
    vmf_like_uncertainty_loss,
)
from eval_metrics.eval_tasks import is_pairwise_rank_flat_key, pairwise_ranks
from eval_metrics.log_utils import log_metric_summary
from eval_metrics.rejection_curves import (
    collect_rejection_curve_stats,
    flatten_pairs_up_to_rank,
    query_rejection_curves_for_scores,
    skip_rejection_method,
)
from eval_metrics.scores import MethodScore


def _is_kappa_method(method: MethodScore) -> bool:
    return bool(method.higher_is_confidence) and vmf_like_uncertainty_loss(
        method.uncertainty_loss
    )


def _primary_ece_bin_mode(bin_mode: str) -> str:
    """``all`` uses equal-width percentile bins as the unsuffixed ECE line."""
    mode = str(bin_mode).lower()
    if mode == "all":
        return "percentile"
    return mode


def ece_extras_enabled(bin_mode: str, ece_extras: bool = False) -> bool:
    """Nested ECE variants (equal-mass, opposite clip, −κ). Off by default."""
    return bool(ece_extras) or str(bin_mode or "").lower() == "all"


def _ece_bin_extras(
    method: MethodScore, bin_mode: str, *, enabled: bool
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Extra ECE binning passes saved alongside the primary ``bin_mode``.

    Off unless ``--ece_extras`` (or ``--ece_bin_mode all``). Each item is
    ``(key_suffix, compute_ece kwargs overlay, log_label)``. Invert/reciprocal
    copies are skipped (they already share the parent scores).
    """
    name = str(method.name)
    if not enabled or name.endswith("_neg") or name.endswith("_reciprocal"):
        return []
    mode = _primary_ece_bin_mode(bin_mode)
    extras: List[Tuple[str, Dict[str, Any], str]] = [
        ("_equal_mass", {"bin_mode": "equal_mass"}, "ECE (equal-mass)"),
    ]
    two_sided_primary = mode != "zoom" and bool(method.percentile_two_sided)
    one_sided_primary = mode != "zoom" and not bool(method.percentile_two_sided)
    if not one_sided_primary:
        extras.append(
            (
                "_one_sided",
                {"bin_mode": "percentile", "percentile_two_sided": False},
                "ECE (one-sided)",
            )
        )
    if not two_sided_primary:
        extras.append(
            (
                "_two_sided",
                {"bin_mode": "percentile", "percentile_two_sided": True},
                "ECE (two-sided)",
            )
        )
    return extras


def _method_rejection(
    method: MethodScore,
    predictions: np.ndarray,
    positives_per_query,
    n_values: Sequence[int],
    rejection_delta_pcts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Query-level rejection curves and Δ@pcts stats. No plotting."""
    query_curves = query_rejection_curves_for_scores(
        predictions,
        positives_per_query,
        method.query_scores,
        method_name=method.name,
        higher_is_confidence=method.higher_is_confidence,
        recall_values=n_values,
    )
    stats = collect_rejection_curve_stats(
        query_curves,
        {"curves": []},
        rejection_delta_pcts=rejection_delta_pcts,
    )
    return {
        "stats": stats,
        "query_curves": query_curves,
        "pair_curves": [],
    }


def _empty_rejection() -> Dict[str, Any]:
    return {"stats": {}, "query_curves": [], "pair_curves": []}


def _rejection_stats_at_level(
    stats: Optional[Dict[str, Any]], level: str
) -> Dict[str, Any]:
    """Keep rejection-delta entries whose curve ``level`` is ``query`` or ``match``."""
    return {
        label: entry
        for label, entry in (stats or {}).items()
        if isinstance(entry, dict) and str(entry.get("level") or "") == level
    }


def _ece_clip_alts_from_results(
    ece: Dict[str, Any], name: str
) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for suffix, label in (
        ("_one_sided", "ECE (one-sided)"),
        ("_two_sided", "ECE (two-sided)"),
    ):
        key = f"{name}{suffix}"
        if key in ece:
            out.append((label, ece[key]))
    return out


def _pair_raw_sample(
    raw: Dict[str, Any], pw_name: str, pair_ranks: Sequence[int]
) -> Optional[np.ndarray]:
    for k in sorted((int(r) for r in pair_ranks), reverse=True):
        arr = raw.get(f"{pw_name}_r{k}")
        if arr is not None:
            return arr
    return None


def _metric_log_kwargs(
    method_key: str,
    one: Dict[str, Any],
    *,
    pairwise: bool,
    pair_ranks: Sequence[int],
    query_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    ece = one.get("ece") or {}
    rej = (one.get("rejection") or {}).get("stats") or {}
    raw = one.get("raw") or {}
    clip_alts = _ece_clip_alts_from_results(ece, method_key)
    alt, alt_label = (clip_alts[0][1], clip_alts[0][0]) if clip_alts else (None, None)
    equal_mass = ece.get(f"{method_key}_equal_mass")
    if pairwise:
        scores = _pair_raw_sample(raw, method_key, pair_ranks)
        level = "match"
    else:
        scores = raw.get(method_key, query_scores)
        level = "query"
    return {
        "ece_result": ece.get(method_key),
        "scores": scores,
        "ece_neg": ece.get(f"{method_key}_neg"),
        "ece_bin_alt": alt,
        "ece_bin_alt_label": alt_label,
        "ece_clip_alts": clip_alts,
        "ece_equal_mass": equal_mass,
        "rejection": _rejection_stats_at_level(rej, level),
    }


def _has_metric_log(kwargs: Dict[str, Any]) -> bool:
    return bool(
        kwargs.get("ece_result")
        or kwargs.get("rejection")
    )


def log_evaluate_method(
    method: MethodScore,
    one: Dict[str, Any],
    *,
    n_values: Sequence[int],
    one_pd: Optional[Dict[str, Any]] = None,
) -> None:
    """Log query-level then pairwise summaries; ``one_pd`` adds per-dataset lines."""
    pair_ranks = pairwise_ranks(n_values)
    none_q = _metric_log_kwargs(
        method.name,
        one,
        pairwise=False,
        pair_ranks=pair_ranks,
        query_scores=method.query_scores,
    )
    pd_q = (
        _metric_log_kwargs(
            method.name,
            one_pd,
            pairwise=False,
            pair_ranks=pair_ranks,
            query_scores=method.query_scores,
        )
        if one_pd is not None
        else None
    )
    log_metric_summary(
        method.log_name,
        none_q["ece_result"],
        none_q["scores"],
        ece_neg=none_q["ece_neg"],
        ece_bin_alt=none_q["ece_bin_alt"],
        ece_bin_alt_label=none_q["ece_bin_alt_label"],
        ece_clip_alts=none_q["ece_clip_alts"],
        ece_equal_mass=none_q["ece_equal_mass"],
        rejection=none_q["rejection"],
        pd=pd_q,
    )
    pw_name = method.pairwise_name
    if not pw_name:
        return
    none_pw = _metric_log_kwargs(pw_name, one, pairwise=True, pair_ranks=pair_ranks)
    pd_pw = (
        _metric_log_kwargs(pw_name, one_pd, pairwise=True, pair_ranks=pair_ranks)
        if one_pd is not None
        else None
    )
    if not _has_metric_log(none_pw) and not (pd_pw and _has_metric_log(pd_pw)):
        return
    scores = none_pw["scores"] if none_pw["scores"] is not None else method.query_scores
    log_metric_summary(
        pw_name,
        none_pw["ece_result"],
        scores,
        ece_neg=none_pw["ece_neg"],
        ece_bin_alt=none_pw["ece_bin_alt"],
        ece_bin_alt_label=none_pw["ece_bin_alt_label"],
        ece_clip_alts=none_pw["ece_clip_alts"],
        ece_equal_mass=none_pw["ece_equal_mass"],
        rejection=none_pw["rejection"],
        pd=pd_pw,
    )


def _png_with_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    if filename.endswith(".png"):
        return f"{filename[:-4]}{suffix}.png"
    return f"{filename}{suffix}"


def merge_method_eval(
    panel: Dict[str, Any],
    one: Dict[str, Any],
    *,
    per_dataset: bool = False,
) -> None:
    """Copy ``evaluate_method`` blocks onto the panel (same keys W&B reads)."""
    suffix = "_norm_per_dataset" if per_dataset else ""
    for name in ("ece",):
        block = one.get(name) or {}
        if block:
            panel.setdefault(f"{name}{suffix}", {}).update(block)
    if not per_dataset:
        raw = {
            k: v
            for k, v in (one.get("raw") or {}).items()
            if not is_pairwise_rank_flat_key(k)
        }
        if raw:
            panel.setdefault("raw_scores", {}).update(raw)
    stats = (one.get("rejection") or {}).get("stats") or {}
    if stats:
        panel.setdefault(f"rejection_curves{suffix}", {}).update(stats)


def evaluate_method(
    method: MethodScore,
    predictions: np.ndarray,
    positives_per_query,
    distances: Optional[np.ndarray],
    *,
    n_values: Sequence[int],
    output_dir,
    ece_metrics: List[str],
    zoom_threshold: float,
    bin_mode: str,
    eval_metrics: Sequence[str],
    rejection_delta_pcts: Optional[Sequence[int]] = None,
    ece_plot_suffix: str = "",
    log_summary: bool = True,
    ece_extras: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run the selected evals for one method (query-level, then pairwise)."""
    ece_results: Dict[str, Any] = {}
    raw_scores: Dict[str, Any] = {}
    rejection = _empty_rejection()
    eval_metrics = list(eval_metrics)
    primary_bin_mode = _primary_ece_bin_mode(bin_mode)
    extras_on = ece_extras_enabled(bin_mode, ece_extras)

    ece_kwargs = dict(
        n_values=list(n_values),
        output_dir=output_dir,
        metrics=ece_metrics,
        zoom_threshold=zoom_threshold,
        bin_mode=primary_bin_mode,
        vmf_kappa_floor=method.use_vmf_kappa_floor,
        percentile_two_sided=method.percentile_two_sided,
    )

    raw_scores[method.name] = method.query_scores
    kappa_alts = extras_on and _is_kappa_method(method)
    extras = _ece_bin_extras(method, primary_bin_mode, enabled=extras_on)

    for name in eval_metrics:
        if name == "ece":
            ece_results[method.name] = compute_ece(
                predictions,
                positives_per_query,
                method.query_scores,
                distances=distances,
                uncertainty_loss=method.uncertainty_loss,
                plot_filename=_png_with_suffix(f"ece_{method.name}.png", ece_plot_suffix),
                **ece_kwargs,
            )
            if kappa_alts:
                ece_results[f"{method.name}_neg"] = compute_ece(
                    predictions,
                    positives_per_query,
                    method.query_scores,
                    distances=distances,
                    uncertainty_loss=method.uncertainty_loss,
                    plot_filename=_png_with_suffix(
                        f"ece_{method.name}_neg.png", ece_plot_suffix
                    ),
                    invert="neg",
                    **ece_kwargs,
                )
            for suffix, overlay, _ in extras:
                ece_results[f"{method.name}{suffix}"] = compute_ece(
                    predictions,
                    positives_per_query,
                    method.query_scores,
                    distances=distances,
                    uncertainty_loss=method.uncertainty_loss,
                    plot_filename=_png_with_suffix(
                        f"ece_{method.name}{suffix}.png", ece_plot_suffix
                    ),
                    **{**ece_kwargs, **overlay},
                )
        elif name == "rejection":
            if skip_rejection_method(method.name):
                continue
            rejection = _method_rejection(
                method,
                predictions,
                positives_per_query,
                n_values,
                rejection_delta_pcts=rejection_delta_pcts,
            )

    out = {
        "ece": ece_results,
        "raw": raw_scores,
        "rejection": rejection,
    }
    if method.pairwise_name is None:
        if log_summary:
            log_evaluate_method(method, out, n_values=n_values)
        return out

    pw_name = method.pairwise_name
    pair_ranks = pairwise_ranks(n_values)
    pools: Dict[int, Dict[str, Any]] = {}
    if method.pair_scores is not None:
        k_avail = min(int(method.pair_scores.shape[1]), int(predictions.shape[1]))
        for n_rank in pair_ranks:
            k = min(int(n_rank), k_avail)
            flat = flatten_pairs_up_to_rank(
                predictions, positives_per_query, method.pair_scores, k
            )
            if flat is None:
                continue
            pools[int(n_rank)] = flat
            raw_scores[f"{pw_name}_r{int(n_rank)}"] = flat["scores"]
    if not pools:
        if log_summary:
            log_evaluate_method(method, out, n_values=n_values)
        return out

    pair_ece_kwargs = {**ece_kwargs, "n_values": [1]}
    ece_pw = None
    ece_recall: Dict[int, float] = {}
    ece_map: Dict[int, float] = {}
    ece_recall_neg: Dict[int, float] = {}
    ece_map_neg: Dict[int, float] = {}
    ece_neg: Optional[Dict[str, Any]] = None
    extra_holders: Dict[str, Dict[str, Any]] = {
        suffix: {"result": None, "recall": {}, "map": {}}
        for suffix, _, _ in extras
    }
    pair_curves: List[Dict[str, Any]] = []
    plot_k = max(pools) if pools else None
    pw_plot = method.pairwise_plot_filename or f"ece_{pw_name}.png"

    for n_rank, flat in pools.items():
        key = f"{pw_name}_r{n_rank}"
        preds = flat["predictions"]
        pos = flat["positives_per_query"]
        scores = raw_scores[key]
        ece_plot = pw_plot if n_rank == plot_k else f"ece_{key}.png"
        if "ece" in eval_metrics:
            one = compute_ece(
                preds,
                pos,
                scores,
                uncertainty_loss=method.uncertainty_loss,
                plot_filename=_png_with_suffix(ece_plot, ece_plot_suffix),
                **pair_ece_kwargs,
            )
            if ece_pw is None:
                ece_pw = dict(one)
            if "ece_recall" in one:
                ece_recall[n_rank] = float(one["ece_recall"][1])
            if "ece_map" in one:
                ece_map[n_rank] = float(one["ece_map"][1])
            if kappa_alts:
                one_neg = compute_ece(
                    preds,
                    pos,
                    scores,
                    uncertainty_loss=method.uncertainty_loss,
                    plot_filename=_png_with_suffix(
                        pw_plot.replace(".png", "_neg.png")
                        if n_rank == plot_k
                        else f"ece_{key}_neg.png",
                        ece_plot_suffix,
                    ),
                    invert="neg",
                    **pair_ece_kwargs,
                )
                if ece_neg is None:
                    ece_neg = dict(one_neg)
                if "ece_recall" in one_neg:
                    ece_recall_neg[n_rank] = float(one_neg["ece_recall"][1])
                if "ece_map" in one_neg:
                    ece_map_neg[n_rank] = float(one_neg["ece_map"][1])
            for suffix, overlay, _ in extras:
                extra_plot = (
                    pw_plot.replace(".png", f"{suffix}.png")
                    if n_rank == plot_k
                    else f"ece_{key}{suffix}.png"
                )
                one_ex = compute_ece(
                    preds,
                    pos,
                    scores,
                    uncertainty_loss=method.uncertainty_loss,
                    plot_filename=_png_with_suffix(extra_plot, ece_plot_suffix),
                    **{**pair_ece_kwargs, **overlay},
                )
                holder = extra_holders[suffix]
                if holder["result"] is None:
                    holder["result"] = dict(one_ex)
                if "ece_recall" in one_ex:
                    holder["recall"][n_rank] = float(one_ex["ece_recall"][1])
                if "ece_map" in one_ex:
                    holder["map"][n_rank] = float(one_ex["ece_map"][1])
        if "rejection" in eval_metrics:
            curves = query_rejection_curves_for_scores(
                preds,
                pos,
                scores,
                method_name=pw_name,
                higher_is_confidence=method.higher_is_confidence,
                recall_values=[1],
            )
            for curve in curves:
                curve["n_rank"] = int(n_rank)
                curve["level"] = "match"
                curve["label"] = f"{pw_name} R@{int(n_rank)}"
            pair_curves.extend(curves)

    if ece_pw is not None:
        if ece_recall:
            ece_pw["ece_recall"] = ece_recall
        if ece_map:
            ece_pw["ece_map"] = ece_map
        ece_results[pw_name] = ece_pw
    if ece_neg is not None:
        if ece_recall_neg:
            ece_neg["ece_recall"] = ece_recall_neg
        if ece_map_neg:
            ece_neg["ece_map"] = ece_map_neg
        ece_results[f"{pw_name}_neg"] = ece_neg
    for suffix, holder in extra_holders.items():
        result = holder["result"]
        if result is None:
            continue
        if holder["recall"]:
            result["ece_recall"] = holder["recall"]
        if holder["map"]:
            result["ece_map"] = holder["map"]
        ece_results[f"{pw_name}{suffix}"] = result
    if pair_curves:
        rejection["pair_curves"] = pair_curves
        rejection["stats"] = collect_rejection_curve_stats(
            rejection["query_curves"],
            {"curves": pair_curves},
            rejection_delta_pcts=rejection_delta_pcts,
        )

    if log_summary:
        log_evaluate_method(method, out, n_values=n_values)
    return out
