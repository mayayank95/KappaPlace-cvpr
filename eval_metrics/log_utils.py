"""Shared logging helpers for evaluation metric summaries."""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from eval_metrics.rejection_curves import rejection_delta_pcts_from_entry
from eval_metrics.scores import METHOD_LOG_NAMES
from eval_metrics.uncertainty import compute_variance_summary

logger = logging.getLogger(__name__)


def _ece_ap_value(ece_result: Optional[Any]) -> Optional[float]:
    """Extract scalar ECE-AP from a full ``compute_ece`` dict or panel payload."""
    if not isinstance(ece_result, dict) or "ece_ap" not in ece_result:
        return None
    val = ece_result["ece_ap"]
    if val is None:
        return None
    fval = float(val)
    return fval if np.isfinite(fval) else None


def format_at_k(
    prefix: str,
    values: Dict[Any, Any],
    *,
    precision: int = 3,
    signed: bool = False,
    skip_r1: bool = False,
) -> str:
    """Format like ECE: ``ECE_R@1/5/10/20: 0.123/0.234/0.345/0.456``.

    ``skip_r1``: a lone rank-1 value is printed as ``prefix: 0.123``.
    """
    if not values:
        return ""
    items = []
    for key, val in values.items():
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fval):
            continue
        items.append((int(key), fval))
    if not items:
        return ""
    items.sort(key=lambda kv: kv[0])
    spec = f"{{:+.{precision}f}}" if signed else f"{{:.{precision}f}}"
    if skip_r1 and len(items) == 1 and items[0][0] == 1:
        return f"{prefix}: {spec.format(items[0][1])}"
    return (
        f"{prefix}@"
        + "/".join(str(k) for k, _ in items)
        + ": "
        + "/".join(spec.format(v) for _, v in items)
    )


def format_ece(ece_result: Optional[Dict]) -> str:
    if not ece_result:
        return "N/A"
    parts = []
    if "ece_recall" in ece_result:
        line = format_at_k("ECE_R", ece_result["ece_recall"], precision=3)
        if line:
            parts.append(line)
    if "ece_map" in ece_result:
        line = format_at_k("ECE_mAP", ece_result["ece_map"], precision=3)
        if line:
            parts.append(line)
    return ", ".join(parts) if parts else str(ece_result)


_NONE_SUFFIX = " (none)"
_PD_SUFFIX = " (per_dataset)"


def _norm_suffix(has_pd: bool, *, per_dataset: bool = False) -> str:
    if not has_pd:
        return ""
    return _PD_SUFFIX if per_dataset else _NONE_SUFFIX


def _log_ece_block(
    ece_result: Optional[Dict],
    *,
    indent: str = "  ",
    line_suffix: str = "",
) -> None:
    """Log ECE recall/map line and a separate ECE_AP line when available."""
    if not ece_result:
        return
    formatted = format_ece(ece_result)
    if formatted and formatted != "N/A":
        logger.info("%s%s%s", indent, formatted, line_suffix)
    ece_ap = _ece_ap_value(ece_result)
    if ece_ap is not None:
        logger.info("%sECE_AP: %.4f%s", indent, ece_ap, line_suffix)


def _as_at_k(value: Optional[Union[float, Dict[Any, Any]]]) -> Optional[Dict[int, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        out: Dict[int, float] = {}
        for key, val in value.items():
            if val is None:
                continue
            try:
                out[int(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return out or None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fval):
        return None
    return {1: fval}


def _rejection_deltas_by_method(
    stats: Optional[Dict[str, Any]],
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """method → pct → recall@K → delta."""
    grouped: Dict[str, Dict[int, Dict[int, float]]] = {}
    if not stats:
        return grouped
    for label, entry in stats.items():
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method") or label)
        n_rank = int(entry.get("n_rank") or 1)
        by_pct = grouped.setdefault(method, {})
        pcts = rejection_delta_pcts_from_entry(entry)
        for pct in pcts:
            val = entry.get(f"delta_at_{pct}")
            if val is None or not np.isfinite(float(val)):
                continue
            by_pct.setdefault(pct, {})[n_rank] = float(val)
        if not pcts:
            imp = entry.get("improvement")
            if imp is not None and np.isfinite(float(imp)):
                by_pct.setdefault(90, {})[n_rank] = float(imp)
    return grouped


def log_rejection_deltas(
    stats: Optional[Dict[str, Any]],
    *,
    indent: str = "  ",
    include_method: bool = False,
    line_suffix: str = "",
) -> None:
    """Log rejection deltas as ``Rejection_Δ@20_R@1/5/10/20: +1.2/+2.3/+3.4/+4.5``."""
    grouped = _rejection_deltas_by_method(stats)
    many = include_method or len(grouped) > 1
    for method, by_pct in grouped.items():
        extra = f" ({method})" if many else ""
        for pct in sorted(by_pct):
            line = format_at_k(
                f"Rejection_Δ@{pct}_R", by_pct.get(pct) or {}, precision=1, signed=True
            )
            if line:
                logger.info("%s%s%s%s", indent, line, extra, line_suffix)


def _log_at_k_line(
    prefix: str,
    value: Optional[Union[float, Dict[Any, Any]]],
    *,
    indent: str,
    precision: int,
    skip_r1: bool = False,
    signed: bool = False,
    line_suffix: str = "",
) -> None:
    mapping = _as_at_k(value)
    if not mapping:
        return
    line = format_at_k(
        prefix, mapping, precision=precision, signed=signed, skip_r1=skip_r1
    )
    if line:
        logger.info("%s%s%s", indent, line, line_suffix)


def _log_rejection_dual(
    none_stats: Optional[Dict[str, Any]],
    pd_stats: Optional[Dict[str, Any]],
    *,
    indent: str,
) -> None:
    """Emit Rejection_Δ lines as none then per_dataset for each percentile."""
    none_g = _rejection_deltas_by_method(none_stats)
    pd_g = _rejection_deltas_by_method(pd_stats)
    if len(none_g) > 1 or len(pd_g) > 1:
        log_rejection_deltas(none_stats, indent=indent, line_suffix=_NONE_SUFFIX)
        log_rejection_deltas(pd_stats, indent=indent, line_suffix=_PD_SUFFIX)
        return
    none_by_pct = next(iter(none_g.values()), {}) if none_g else {}
    pd_by_pct = next(iter(pd_g.values()), {}) if pd_g else {}
    for pct in sorted(set(none_by_pct) | set(pd_by_pct)):
        _log_at_k_line(
            f"Rejection_Δ@{pct}_R",
            none_by_pct.get(pct),
            indent=indent,
            precision=1,
            signed=True,
            line_suffix=_NONE_SUFFIX,
        )
        _log_at_k_line(
            f"Rejection_Δ@{pct}_R",
            pd_by_pct.get(pct),
            indent=indent,
            precision=1,
            signed=True,
            line_suffix=_PD_SUFFIX,
        )


def _log_nested_ece(
    label: str,
    result: Optional[Dict[str, Any]],
    result_pd: Optional[Dict[str, Any]],
    *,
    indent: str,
    nested: str,
    none_suf: str,
    pd_suf: str,
    has_pd: bool,
) -> None:
    if result is None and result_pd is None:
        return
    logger.info("%s%s", indent, label)
    _log_ece_block(result, indent=nested, line_suffix=none_suf)
    if has_pd:
        _log_ece_block(result_pd, indent=nested, line_suffix=pd_suf)


def _log_ece_family(
    ece_result: Optional[Dict[str, Any]],
    ece_pd: Optional[Dict[str, Any]],
    ece_neg: Optional[Dict[str, Any]],
    ece_neg_pd: Optional[Dict[str, Any]],
    ece_bin_alt: Optional[Dict[str, Any]],
    ece_bin_alt_pd: Optional[Dict[str, Any]],
    ece_bin_alt_label: Optional[str],
    *,
    indent: str,
    none_suf: str,
    pd_suf: str,
    has_pd: bool,
    ece_equal_mass: Optional[Dict[str, Any]] = None,
    ece_equal_mass_pd: Optional[Dict[str, Any]] = None,
    ece_clip_alts: Optional[List[Tuple[str, Optional[Dict[str, Any]]]]] = None,
    ece_clip_alts_pd: Optional[List[Tuple[str, Optional[Dict[str, Any]]]]] = None,
) -> None:
    """Main ECE, then equal-mass, then −κ, then one-sided / two-sided."""
    _log_ece_block(ece_result, indent=indent, line_suffix=none_suf)
    if has_pd:
        _log_ece_block(ece_pd, indent=indent, line_suffix=pd_suf)
    nested = indent + "  "
    _log_nested_ece(
        "ECE (equal-mass)",
        ece_equal_mass,
        ece_equal_mass_pd,
        indent=indent,
        nested=nested,
        none_suf=none_suf,
        pd_suf=pd_suf,
        has_pd=has_pd,
    )
    _log_nested_ece(
        "ECE (−κ)",
        ece_neg,
        ece_neg_pd,
        indent=indent,
        nested=nested,
        none_suf=none_suf,
        pd_suf=pd_suf,
        has_pd=has_pd,
    )
    clips = list(ece_clip_alts or [])
    if not clips and (ece_bin_alt is not None or ece_bin_alt_pd is not None):
        clips = [(ece_bin_alt_label or "ECE (alt bins)", ece_bin_alt)]
    pd_clips = {label: result for label, result in (ece_clip_alts_pd or [])}
    if ece_bin_alt_pd is not None:
        pd_clips.setdefault(ece_bin_alt_label or "ECE (alt bins)", ece_bin_alt_pd)
    for label, result in clips:
        _log_nested_ece(
            label,
            result,
            pd_clips.get(label),
            indent=indent,
            nested=nested,
            none_suf=none_suf,
            pd_suf=pd_suf,
            has_pd=has_pd,
        )


def _log_eval_metric_lines(
    ece_result: Optional[Dict[str, Any]],
    rejection: Optional[Dict[str, Any]] = None,
    *,
    indent: str = "  ",
    line_suffix: str = "",
    ece_pd: Optional[Dict[str, Any]] = None,
    rejection_pd: Optional[Dict[str, Any]] = None,
    has_pd: bool = False,
    ece_neg: Optional[Dict[str, Any]] = None,
    ece_neg_pd: Optional[Dict[str, Any]] = None,
    ece_bin_alt: Optional[Dict[str, Any]] = None,
    ece_bin_alt_pd: Optional[Dict[str, Any]] = None,
    ece_bin_alt_label: Optional[str] = None,
    ece_equal_mass: Optional[Dict[str, Any]] = None,
    ece_equal_mass_pd: Optional[Dict[str, Any]] = None,
    ece_clip_alts: Optional[List[Tuple[str, Optional[Dict[str, Any]]]]] = None,
    ece_clip_alts_pd: Optional[List[Tuple[str, Optional[Dict[str, Any]]]]] = None,
) -> None:
    none_suf = _norm_suffix(has_pd) or line_suffix
    pd_suf = _norm_suffix(has_pd, per_dataset=True)
    _log_ece_family(
        ece_result,
        ece_pd,
        ece_neg,
        ece_neg_pd,
        ece_bin_alt,
        ece_bin_alt_pd,
        ece_bin_alt_label,
        indent=indent,
        none_suf=none_suf,
        pd_suf=pd_suf,
        has_pd=has_pd,
        ece_equal_mass=ece_equal_mass,
        ece_equal_mass_pd=ece_equal_mass_pd,
        ece_clip_alts=ece_clip_alts,
        ece_clip_alts_pd=ece_clip_alts_pd,
    )
    if has_pd:
        _log_rejection_dual(rejection, rejection_pd, indent=indent)
    else:
        log_rejection_deltas(rejection, indent=indent, line_suffix=none_suf)


def log_stats(prefix: str, stats: Dict[str, float], *, debug: bool = False) -> None:
    log_fn = logger.debug if debug else logger.info
    log_fn(
        "  %s: min=%.4g, max=%.4g, mean=%.4g, median=%.4g",
        prefix,
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["median"],
    )


def _log_score_stats(
    ece_result: Optional[Dict[str, Any]],
    scores: Optional[np.ndarray],
    *,
    line_suffix: str = "",
) -> None:
    score_stats = ece_result.get("score_stats") if isinstance(ece_result, dict) else None
    if score_stats:
        if "variances_after_clamp" in score_stats:
            log_stats(
                f"variances (after clamp){line_suffix}",
                score_stats["variances_after_clamp"],
            )
        if "confidence_after_clamp" in score_stats:
            log_stats(
                f"confidence (after clamp){line_suffix}",
                score_stats["confidence_after_clamp"],
            )
        if "variances_before_clamp" in score_stats:
            log_stats(
                f"variances (before clamp){line_suffix}",
                score_stats["variances_before_clamp"],
                debug=True,
            )
        if "confidence_before_clamp" in score_stats:
            log_stats(
                f"confidence (before clamp){line_suffix}",
                score_stats["confidence_before_clamp"],
                debug=True,
            )
    elif scores is not None and getattr(scores, "size", 0):
        log_stats(f"scores{line_suffix}", compute_variance_summary(np.asarray(scores)))


def log_metric_summary(
    metric_name: str,
    ece_result: Optional[Dict[str, Any]],
    scores: np.ndarray,
    *,
    ece_neg: Optional[Dict[str, Any]] = None,
    ece_bin_alt: Optional[Dict[str, Any]] = None,
    ece_bin_alt_label: Optional[str] = None,
    ece_clip_alts: Optional[List[Tuple[str, Optional[Dict[str, Any]]]]] = None,
    ece_equal_mass: Optional[Dict[str, Any]] = None,
    rejection: Optional[Dict[str, Any]] = None,
    pd: Optional[Dict[str, Any]] = None,
) -> None:
    """Log ECE, rejection deltas, and score stats for one method.

    ``pd`` is an optional per-dataset counterpart with the same keys as
    the none-norm kwargs (``ece_result``, ``rejection``, …). ECE and rejection
    use per-dataset min-max. When set, each line is emitted twice with
    ``(none)`` / ``(per_dataset)``.
    """
    has_pd = pd is not None
    pd = pd or {}
    logger.info("\033[1m%s\033[0m", metric_name)
    _log_eval_metric_lines(
        ece_result,
        rejection,
        ece_pd=pd.get("ece_result"),
        rejection_pd=pd.get("rejection"),
        has_pd=has_pd,
        ece_neg=ece_neg,
        ece_neg_pd=pd.get("ece_neg"),
        ece_bin_alt=ece_bin_alt,
        ece_bin_alt_pd=pd.get("ece_bin_alt"),
        ece_bin_alt_label=ece_bin_alt_label or pd.get("ece_bin_alt_label"),
        ece_equal_mass=ece_equal_mass,
        ece_equal_mass_pd=pd.get("ece_equal_mass"),
        ece_clip_alts=ece_clip_alts,
        ece_clip_alts_pd=pd.get("ece_clip_alts"),
    )
    none_suf = _norm_suffix(has_pd)
    pd_suf = _norm_suffix(has_pd, per_dataset=True)
    _log_score_stats(ece_result, scores, line_suffix=none_suf)
    if has_pd:
        _log_score_stats(pd.get("ece_result"), pd.get("scores"), line_suffix=pd_suf)


def _is_primary_variant(name: str) -> bool:
    if "_reciprocal" in name:
        return False
    return re.search(r"(_neg|_one_sided|_two_sided|_equal_mass|_zoom|_r\d+)$", name) is None


def _panel_primary_names(panel: Dict[str, Any]) -> List[str]:
    """Main method keys (query + pairwise), skipping −κ / bin-alt / rank suffixes."""
    names: List[str] = []
    seen = set()
    for key in panel.get("ece") or {}:
        if _is_primary_variant(key) and key not in seen:
            names.append(key)
            seen.add(key)
    extra: set = set()
    for block_name in ("ece_norm_per_dataset",):
        extra |= set(panel.get(block_name) or {})
    for key in sorted(extra - seen):
        stem = re.sub(r"_r\d+$", "", str(key))
        if _is_primary_variant(stem) and stem not in seen:
            names.append(stem)
            seen.add(stem)
    return names


def _ece_metrics_only(ece_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(ece_result, dict):
        return ece_result
    return {k: v for k, v in ece_result.items() if k != "score_stats"}


def _ece_clip_alts_from_panel(ece: Dict[str, Any], name: str):
    out = []
    for suffix, label in (
        ("_one_sided", "ECE (one-sided)"),
        ("_two_sided", "ECE (two-sided)"),
    ):
        key = f"{name}{suffix}"
        if key in ece:
            out.append((label, _ece_metrics_only(ece[key])))
    return out


def _rejection_for_method(
    stats: Optional[Dict[str, Any]], method_name: str
) -> Dict[str, Any]:
    return {
        label: entry
        for label, entry in (stats or {}).items()
        if isinstance(entry, dict) and str(entry.get("method") or "") == method_name
    }


def _panel_method_log_kwargs(
    panel: Dict[str, Any], name: str, *, per_dataset: bool
) -> Dict[str, Any]:
    suffix = "_norm_per_dataset" if per_dataset else ""
    ece = panel.get(f"ece{suffix}") or {}
    rej = panel.get(f"rejection_curves{suffix}") or {}
    clip_alts = _ece_clip_alts_from_panel(ece, name)
    alt, alt_label = (clip_alts[0][1], clip_alts[0][0]) if clip_alts else (None, None)
    return {
        "ece_result": _ece_metrics_only(ece.get(name)),
        "ece_neg": _ece_metrics_only(ece.get(f"{name}_neg")),
        "ece_bin_alt": alt,
        "ece_bin_alt_label": alt_label,
        "ece_clip_alts": clip_alts,
        "ece_equal_mass": _ece_metrics_only(ece.get(f"{name}_equal_mass")),
        "rejection": _rejection_for_method(rej, name),
        "scores": None,
    }


def log_combined_panel(name: str, panel: Dict[str, Any]) -> None:
    """Log combined recall plus per-method none vs per-dataset metrics."""
    logger.info("\033[1m%s\033[0m", name)

    recalls = panel.get("recalls") or {}
    if recalls:
        line = format_at_k("Recall_R", recalls, precision=2)
        if line:
            logger.info("  %s", line)

    has_pd = bool(
        panel.get("ece_norm_per_dataset")
        or panel.get("rejection_curves_norm_per_dataset")
    )
    for variant in _panel_primary_names(panel):
        none_kw = _panel_method_log_kwargs(panel, variant, per_dataset=False)
        pd_kw = (
            _panel_method_log_kwargs(panel, variant, per_dataset=True) if has_pd else None
        )
        log_metric_summary(
            METHOD_LOG_NAMES.get(variant, variant),
            none_kw["ece_result"],
            none_kw["scores"],
            ece_neg=none_kw["ece_neg"],
            ece_bin_alt=none_kw["ece_bin_alt"],
            ece_bin_alt_label=none_kw["ece_bin_alt_label"],
            ece_clip_alts=none_kw["ece_clip_alts"],
            ece_equal_mass=none_kw["ece_equal_mass"],
            rejection=none_kw["rejection"],
            pd=pd_kw,
        )
