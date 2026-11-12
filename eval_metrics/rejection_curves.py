"""Rejection curves: correctness vs. fraction of most uncertain queries removed."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from eval_metrics.eval_tasks import pairwise_ranks
from eval_metrics.recall import recall_hits_at_k
from eval_metrics.scores import LEARNED_PAIRWISE_SCORE_NAMES, LEARNED_QUERY_SCORE_NAMES
from eval_metrics.scores.joint_kappa import per_example_mean

PathLike = Union[str, Path]

# log(SUE) is strictly monotone, so SUE and SUE-log share the same rejection ranking.
REJECTION_SKIP_METHODS = frozenset({"sue_log"})
REJECTION_METHOD_LABELS = {
    "kappa": "κ",
    "joint_kappa": "joint κ",
    "l2_joint_kappa": "L2·κ",
    "mls": "MLS",
    "l2": "L2",
    "pa": "PA",
    "sue": "SUE",
    "sue_log_kappa": "SUE-log+κ",
    "pairwise_joint_kappa": "pair joint κ",
    "pairwise_l2_joint_kappa": "pair L2·κ",
    "pairwise_mls": "pair MLS",
    "pairwise_l2": "pair L2",
}


def skip_rejection_method(name: str) -> bool:
    """True for methods that add no new rejection ranking (currently SUE-log)."""
    return str(name) in REJECTION_SKIP_METHODS


def rejection_method_label(name: str) -> str:
    """Short legend label for a rejection-curve method."""
    return REJECTION_METHOD_LABELS.get(str(name), str(name))


def all_shards_have_raw_score(
    agg_data_list: Optional[Sequence[Dict[str, Any]]],
    key: str,
) -> bool:
    """True only if every pooled dataset has a non-None ``raw_scores[key]``."""
    shards = list(agg_data_list or [])
    return bool(shards) and all(
        (d.get("raw_scores") or {}).get(key) is not None for d in shards
    )


DEFAULT_REJECTION_DELTA_PCTS: tuple[int, ...] = (20, 50, 90)
_DELTA_KEY_PREFIX = "delta_at_"


def normalize_rejection_delta_pcts(
    pcts: Optional[Sequence[float]] = None,
) -> List[int]:
    """Validate and sort rejection-delta percentages (default 20, 50, 90)."""
    if pcts is None:
        return list(DEFAULT_REJECTION_DELTA_PCTS)
    out = sorted({int(p) for p in pcts})
    if not out:
        raise ValueError("rejection_delta_pcts must include at least one percentage.")
    if any(p < 0 or p > 100 for p in out):
        raise ValueError("rejection_delta_pcts must be in [0, 100].")
    return out


def rejection_delta_pcts_from_entry(entry: Dict[str, Any]) -> List[int]:
    """Read ``delta_at_*`` keys from one stats entry."""
    pcts: List[int] = []
    for key in entry:
        if not isinstance(key, str) or not key.startswith(_DELTA_KEY_PREFIX):
            continue
        suffix = key[len(_DELTA_KEY_PREFIX):]
        if suffix.isdigit():
            pcts.append(int(suffix))
    return sorted(set(pcts))


def rejection_delta_pcts_from_stats(stats: Optional[Dict[str, Any]]) -> List[int]:
    """Union of ``delta_at_*`` percentages across a rejection-stats dict."""
    pcts: set[int] = set()
    for entry in (stats or {}).values():
        if isinstance(entry, dict):
            pcts.update(rejection_delta_pcts_from_entry(entry))
    return sorted(pcts)


def shard_matched_vector(
    preds: np.ndarray,
    positives,
    n_rank: int = 1,
) -> np.ndarray:
    """Per-query recall@K hit/miss. Callers must pass labeled queries only."""
    return recall_hits_at_k(preds, positives, n_rank=n_rank)


def compute_rejection_curve(
    matched: np.ndarray,
    scores: np.ndarray,
    *,
    higher_is_confidence: bool = False,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute correctness ratio after removing the most uncertain queries.

    Queries are ranked by uncertainty (highest removed first). At each rejection
    level, correctness is the mean per-query hit rate among the remaining
    queries (typically recall@K labels).

    Parameters
    ----------
    matched
        1 if the query is correct under the chosen recall@K, else 0.
    scores
        Per-query score used to rank uncertainty. When ``higher_is_confidence``
        is False, larger scores mean more uncertain (e.g. variance). When True,
        larger scores mean more confident and low scores are removed first.
    higher_is_confidence
        Whether ``scores`` increase with confidence rather than uncertainty.
    rejection_percentages
        Fractions of queries to remove, in [0, 100]. Default: 0, 10, …, 90.

    Returns
    -------
    dict with keys ``rejection_percentages``, ``correctness_ratio`` (percent),
    ``n_remaining``, ``improvement`` (end − start, percentage points).
    """
    y = np.asarray(matched, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or s.size != y.size:
        return {
            "rejection_percentages": np.array([], dtype=np.float64),
            "correctness_ratio": np.array([], dtype=np.float64),
            "n_remaining": np.array([], dtype=np.int64),
            "improvement": float("nan"),
        }

    if rejection_percentages is None:
        rejection_percentages = list(range(0, 91, 10))
    reject_pcts = np.asarray(rejection_percentages, dtype=np.float64)

    # Most uncertain queries removed first.
    order = np.argsort(s if not higher_is_confidence else -s, kind="stable")[::-1]
    y_sorted = y[order]
    n = y.size

    correctness = []
    n_remaining = []
    for pct in reject_pcts:
        n_remove = int(round(n * float(pct) / 100.0))
        n_remove = min(max(n_remove, 0), n)
        n_keep = n - n_remove
        if n_keep <= 0:
            correctness.append(float("nan"))
            n_remaining.append(0)
            continue
        correctness.append(float(np.mean(y_sorted[n_remove:]) * 100.0))
        n_remaining.append(n_keep)

    correctness_arr = np.asarray(correctness, dtype=np.float64)
    start = correctness_arr[0] if correctness_arr.size else float("nan")
    end = correctness_arr[-1] if correctness_arr.size else float("nan")
    improvement = float(end - start) if np.isfinite(start) and np.isfinite(end) else float("nan")

    return {
        "rejection_percentages": reject_pcts,
        "correctness_ratio": correctness_arr,
        "n_remaining": np.asarray(n_remaining, dtype=np.int64),
        "improvement": improvement,
    }


def plot_rejection_curves(
    curves: Sequence[Dict[str, Any]],
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[str] = None,
    annotate_improvement: bool = True,
    figsize: tuple = (6.0, 4.5),
    xlabel: str = "Percentage of removed most uncertain query (%)",
    ylabel: str = "Correctness ratio (%)",
) -> Any:
    """Plot one or more rejection curves (Fig. 5 style).

    Each entry in ``curves`` should contain ``label``, ``rejection_percentages``,
    and ``correctness_ratio``. Optional keys: ``color``, ``linestyle``,
    ``method``, ``n_rank``, ``improvement``.
    """
    import matplotlib.pyplot as plt

    colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd",
    )
    rank_styles = {1: "-", 5: "--", 10: "-.", 20: ":"}
    methods: List[str] = []
    for curve in curves:
        method = curve.get("method")
        if method is not None and method not in methods:
            methods.append(method)
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    width, height = figsize
    if len(curves) > 6:
        width = max(width, 8.5)
    fig, ax = plt.subplots(figsize=(width, height))
    for index, curve in enumerate(curves):
        x = np.asarray(curve["rejection_percentages"], dtype=np.float64)
        y = np.asarray(curve["correctness_ratio"], dtype=np.float64)
        label = curve.get("label", "")
        method = curve.get("method")
        color = curve.get("color") or method_colors.get(method) or colors[index % len(colors)]
        linestyle = curve.get("linestyle") or rank_styles.get(curve.get("n_rank"), "-")
        line, = ax.plot(
            x, y, label=label, color=color, linestyle=linestyle, linewidth=2
        )

        if annotate_improvement:
            imp = curve.get("improvement")
            if imp is None and y.size >= 2 and np.isfinite(y[0]) and np.isfinite(y[-1]):
                imp = float(y[-1] - y[0])
            if imp is not None and np.isfinite(imp) and y.size >= 1:
                sign = "+" if imp >= 0 else ""
                ax.annotate(
                    f"{sign}{imp:.1f}%",
                    xy=(x[-1], y[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=line.get_color(),
                    fontsize=10,
                    va="center",
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if len(curves) > 1:
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, frameon=True, fontsize=9,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.78 if len(curves) > 6 else 0.82, 1.0))
    else:
        fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _concentration_higher_is_confidence_for_rejection(uncertainty_loss: str) -> bool:
    """vMF κ is concentration: low κ = uncertain → remove lowest κ first."""
    return uncertainty_loss.lower() == "vmf"


def flatten_retrieval_up_to_rank(
    predictions: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    n_rank: int,
) -> Optional[Dict[str, Any]]:
    """Flatten ranks ``0 .. n_rank-1`` into one-candidate rows (labels only)."""
    preds = np.asarray(predictions)
    n_q, k_max = preds.shape
    k_lim = min(int(n_rank), k_max)
    if n_q == 0 or k_lim <= 0:
        return None
    return {
        "predictions": preds[:, :k_lim].reshape(-1, 1),
        "positives_per_query": [
            positives_per_query[q] for q in range(n_q) for _ in range(k_lim)
        ],
    }


def flatten_pairs_up_to_rank(
    predictions: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    pair_scores: np.ndarray,
    n_rank: int,
) -> Optional[Dict[str, Any]]:
    """Flatten ranks ``0 .. n_rank-1`` into one-candidate rows (same pool for every metric)."""
    preds = np.asarray(predictions)
    scores_2d = np.asarray(pair_scores, dtype=np.float64)
    if scores_2d.shape != preds.shape:
        raise ValueError(
            f"pair_scores shape {scores_2d.shape} must match predictions {preds.shape}"
        )
    geo = flatten_retrieval_up_to_rank(preds, positives_per_query, n_rank)
    if geo is None:
        return None
    k_lim = min(int(n_rank), int(preds.shape[1]))
    geo["scores"] = scores_2d[:, :k_lim].reshape(-1)
    return geo


def pool_pairs_up_to_rank(
    predictions: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    pair_scores: np.ndarray,
    n_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten retrieved pairs at ranks ``0 .. n_rank-1`` into hit labels and scores."""
    flat = flatten_pairs_up_to_rank(
        predictions, positives_per_query, pair_scores, n_rank
    )
    if flat is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    labels = recall_hits_at_k(
        flat["predictions"], flat["positives_per_query"], n_rank=1
    )
    return labels, flat["scores"]


def pool_pairs_from_shards(
    agg_data_list: Sequence[Dict[str, Any]],
    score_key: str,
    n_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool pair labels and scores across dataset shards up to rank ``n_rank``."""
    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    if not all_shards_have_raw_score(agg_data_list, score_key):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    for d in agg_data_list:
        y_flat, s_flat = pool_pairs_up_to_rank(
            d["predictions"],
            d["positives_per_query"],
            d["raw_scores"][score_key],
            n_rank,
        )
        labels.append(y_flat)
        scores.append(s_flat)
    if not labels:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(labels), np.concatenate(scores)


def compute_pairwise_rejection_curves(
    agg_data_list: Sequence[Dict[str, Any]],
    score_key: str,
    recall_values: Sequence[int],
    *,
    higher_is_confidence: bool,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """One rejection curve per recall@K, removing uncertain pairs independently."""
    curves: List[Dict[str, Any]] = []
    for n_rank in pairwise_ranks(recall_values):
        y, s = pool_pairs_from_shards(agg_data_list, score_key, n_rank)
        curve = compute_rejection_curve(
            y,
            s,
            higher_is_confidence=higher_is_confidence,
            rejection_percentages=rejection_percentages,
        )
        curve["label"] = f"R@{int(n_rank)}"
        curve["n_rank"] = int(n_rank)
        curves.append(curve)
    return curves


def compute_all_pairwise_rejection_curves(
    agg_data_list: Sequence[Dict[str, Any]],
    recall_values: Sequence[int],
    *,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Pair-level curves for all pairwise methods; one line per (method, recall@K)."""
    curves: List[Dict[str, Any]] = []
    for score_key, higher_is_confidence in PAIRWISE_REJECTION_SPECS:
        if not all_shards_have_raw_score(agg_data_list, score_key):
            continue
        for n_rank in pairwise_ranks(recall_values):
            y, s = pool_pairs_from_shards(agg_data_list, score_key, n_rank)
            curve = compute_rejection_curve(
                y,
                s,
                higher_is_confidence=higher_is_confidence,
                rejection_percentages=rejection_percentages,
            )
            curve["label"] = f"{score_key} R@{int(n_rank)}"
            curve["method"] = score_key
            curve["n_rank"] = int(n_rank)
            curves.append(curve)
    return curves


def save_combined_pairwise_rejection_curves(
    agg_data_list: Sequence[Dict[str, Any]],
    output_path: PathLike,
    recall_values: Sequence[int],
    *,
    title: Optional[str] = None,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> tuple[List[Dict[str, Any]], Optional[Path]]:
    """Save pair-level rejection curves (both methods, all recall@K) on one plot."""
    curves = compute_all_pairwise_rejection_curves(
        agg_data_list,
        recall_values,
        rejection_percentages=rejection_percentages,
    )
    if not curves or not any(c["correctness_ratio"].size for c in curves):
        return [], None
    out = Path(output_path)
    plot_rejection_curves(
        curves,
        output_path=out,
        title=title,
        xlabel="Percentage of removed most uncertain pairs (%)",
        ylabel="Pair hit rate (%)",
    )
    return curves, out


PAIRWISE_REJECTION_SPECS = tuple(
    (name, True) for name in LEARNED_PAIRWISE_SCORE_NAMES
) + (
    ("pairwise_l2", False),
)


def _maybe_append_raw_rejection_variant(
    variants: List[tuple],
    agg_data_list: Sequence[Dict[str, Any]],
    key: str,
    *,
    higher_is_confidence: bool,
    n_rank: int = 1,
    label: Optional[str] = None,
) -> None:
    if not all_shards_have_raw_score(agg_data_list, key):
        return
    matched_sh: List[np.ndarray] = []
    score_sh: List[np.ndarray] = []
    for d in agg_data_list:
        matched = shard_matched_vector(
            d["predictions"], d["positives_per_query"], n_rank
        )
        scores = np.asarray(d["raw_scores"][key], dtype=np.float64).reshape(-1)
        matched_sh.append(matched)
        score_sh.append(scores)
    variants.append((label or key, matched_sh, score_sh, higher_is_confidence))


def build_rejection_variants(
    agg_data_list: Sequence[Dict[str, Any]],
    kappa_sh_matched: Optional[Sequence[np.ndarray]] = None,
    *,
    uncertainty_loss: str = "gaussian_nll",
    n_rank: int = 1,
) -> List[tuple]:
    """Build (label, matched_shards, score_shards, higher_is_confidence) specs.

    Each shard dict needs ``predictions``, ``positives_per_query``, ``q_var``,
    and optional ``raw_scores`` (baselines / joint_kappa).

    ``n_rank`` selects query-level recall@K labels (default R@1).

    For vMF, ``q_var`` holds κ (concentration): low κ means high uncertainty, so
    ``higher_is_confidence=True`` removes the least confident queries first.
    For Gaussian NLL, higher variance means higher uncertainty
    (``higher_is_confidence=False``).
    """
    if not agg_data_list:
        return []

    rank = max(1, int(n_rank))
    matched_sh: List[np.ndarray] = []
    kappa_score_sh: List[np.ndarray] = []
    for i, d in enumerate(agg_data_list):
        scores = per_example_mean(d["q_var"])
        kappa_score_sh.append(scores)
        if kappa_sh_matched is not None and rank == 1 and i < len(kappa_sh_matched):
            # Prefer caller-provided labels when already GT-filtered (same length).
            provided = np.asarray(kappa_sh_matched[i], dtype=np.float64).reshape(-1)
            if provided.shape[0] == scores.shape[0]:
                matched_sh.append(provided)
                continue
        matched_sh.append(
            shard_matched_vector(d["predictions"], d["positives_per_query"], rank)
        )

    kappa_hic = _concentration_higher_is_confidence_for_rejection(uncertainty_loss)

    variants: List[tuple] = [
        (
            "kappa",
            matched_sh,
            kappa_score_sh,
            kappa_hic,
        ),
    ]

    for b_name in ["l2", "pa", "sue"]:
        _maybe_append_raw_rejection_variant(
            variants,
            agg_data_list,
            b_name,
            higher_is_confidence=False,
            n_rank=rank,
        )

    # vMF top-1 pair scores (pairwise variants are on the separate pair-level plot).
    for key in LEARNED_QUERY_SCORE_NAMES:
        _maybe_append_raw_rejection_variant(
            variants,
            agg_data_list,
            key,
            higher_is_confidence=True,
            n_rank=rank,
        )

    return variants


def query_curve_label(method: str, n_rank: int) -> str:
    """R@1 keeps the bare method name for report UI compatibility."""
    return str(method) if int(n_rank) == 1 else f"{method} R@{int(n_rank)}"


def compute_all_query_rejection_curves(
    agg_data_list: Sequence[Dict[str, Any]],
    *,
    uncertainty_loss: str = "gaussian_nll",
    recall_values: Optional[Sequence[int]] = None,
    kappa_sh_matched: Optional[Sequence[np.ndarray]] = None,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Query-level curves for all methods × recall@K (same Ks as ``recall_values``)."""
    ranks = [max(1, int(n)) for n in (recall_values or [1])]
    curves: List[Dict[str, Any]] = []
    for n_rank in ranks:
        variants = build_rejection_variants(
            agg_data_list,
            kappa_sh_matched if n_rank == 1 else None,
            uncertainty_loss=uncertainty_loss,
            n_rank=n_rank,
        )
        for curve in compute_combined_rejection_curves(
            variants,
            rejection_percentages=rejection_percentages,
        ):
            method = str(curve["label"])
            if skip_rejection_method(method):
                continue
            curve["method"] = method
            curve["n_rank"] = int(n_rank)
            curve["level"] = "query"
            curve["label"] = query_curve_label(method, n_rank)
            curves.append(curve)
    return curves


def _curve_method_name(curve: Dict[str, Any]) -> str:
    return str(curve.get("method") or curve.get("label") or "")


def _group_curves_by_method(
    curves: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for curve in curves:
        method = _curve_method_name(curve)
        if not method:
            continue
        grouped.setdefault(method, []).append(curve)
    return grouped


def _has_rejection_values(curve: Dict[str, Any]) -> bool:
    y = np.asarray(curve.get("correctness_ratio", []), dtype=np.float64)
    return bool(y.size)


def _plot_one_method_rejection(
    curves: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    title: Optional[str],
    pairwise: bool,
) -> None:
    """One method, all R@K — same split as ``ece_{method}.png`` vs ``ece_pairwise_*.png``."""
    ordered = sorted(curves, key=lambda c: int(c.get("n_rank") or 0))
    plot_curves = []
    for curve in ordered:
        plotted = dict(curve)
        n_rank = curve.get("n_rank")
        plotted["label"] = (
            f"R@{int(n_rank)}" if n_rank is not None else str(curve.get("label") or "")
        )
        plot_curves.append(plotted)
    kwargs: Dict[str, Any] = {"output_path": output_path, "title": title}
    if pairwise:
        kwargs["xlabel"] = "Percentage of removed most uncertain pairs (%)"
        kwargs["ylabel"] = "Pair hit rate (%)"
    else:
        kwargs["ylabel"] = "Recall@K among remaining queries (%)"
    plot_rejection_curves(plot_curves, **kwargs)


def plot_rejection_curve_sets(
    query_curves: Sequence[Dict[str, Any]],
    pairwise_curves: Sequence[Dict[str, Any]],
    output_path: PathLike,
    *,
    title: Optional[str] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Save query-level one-PNG-per-K overlays and pairwise comparison plots.

    Query-level: one figure per recall@K with every method on the same axes
    (``rejection_curves.png`` for R@1, else ``rejection_curves_r{K}.png``).
    Pair-level: one overlay of all pairwise methods, plus per-method files.
    """
    query_path: Optional[Path] = None
    pairwise_path: Optional[Path] = None
    out = Path(output_path)
    base_title = title or out.stem
    query_curves = [
        c for c in query_curves if not skip_rejection_method(_curve_method_name(c))
    ]
    if query_curves and any(_has_rejection_values(c) for c in query_curves):
        ranks = sorted({int(c["n_rank"]) for c in query_curves if c.get("n_rank") is not None})
        for n_rank in ranks:
            rank_curves = [c for c in query_curves if int(c.get("n_rank", 1)) == n_rank]
            if not rank_curves:
                continue
            # One K per figure: solid lines, method names in the legend.
            plot_curves = []
            for curve in rank_curves:
                plotted = dict(curve)
                method = str(curve.get("method") or curve["label"])
                plotted["label"] = rejection_method_label(method)
                plotted["linestyle"] = "-"
                plot_curves.append(plotted)
            if n_rank == 1:
                rank_path = out
            else:
                rank_path = out.with_name(f"{out.stem}_r{n_rank}{out.suffix}")
            plot_rejection_curves(
                plot_curves,
                output_path=rank_path,
                title=f"{base_title} — query R@{n_rank}",
                ylabel=f"Recall@{n_rank} among remaining queries (%)",
            )
            if n_rank == 1:
                query_path = rank_path

    if pairwise_curves and any(_has_rejection_values(c) for c in pairwise_curves):
        pairwise_path = out.with_name(f"{out.stem}_pairwise{out.suffix}")
        plot_rejection_curves(
            list(pairwise_curves),
            output_path=pairwise_path,
            title=f"{base_title} — pair-level",
            xlabel="Percentage of removed most uncertain pairs (%)",
            ylabel="Pair hit rate (%)",
        )

    for method, method_curves in _group_curves_by_method(pairwise_curves).items():
        usable = [c for c in method_curves if _has_rejection_values(c)]
        if not usable:
            continue
        _plot_one_method_rejection(
            usable,
            out.with_name(f"{out.stem}_{method}{out.suffix}"),
            title=f"{base_title} — {method}",
            pairwise=True,
        )
    return query_path, pairwise_path


def save_rejection_curves_for_shards(
    agg_data_list: Sequence[Dict[str, Any]],
    output_path: PathLike,
    *,
    title: Optional[str] = None,
    kappa_sh_matched: Optional[Sequence[np.ndarray]] = None,
    uncertainty_loss: str = "gaussian_nll",
    recall_values: Optional[Sequence[int]] = None,
) -> tuple[List[Dict[str, Any]], Optional[Path], Dict[str, Any]]:
    """Compute query-level and pair-level rejection curves; save separate plots.

    Query-level writes **one PNG per ``recall_values`` K** with all methods
    overlaid (``rejection_curves.png`` for R@1, else ``rejection_curves_r{K}.png``).
    SUE-log is omitted (same ranking as SUE).

    Pair-level still skips R@1 (identical to query R@1).

    Returns ``(query_curves, query_plot_path, pairwise_result)`` where
    ``query_plot_path`` is the R@1 plot (if any) and ``pairwise_result`` has keys
    ``curves`` and ``path`` when the pair-level plot exists.
    """
    query_curves = compute_all_query_rejection_curves(
        agg_data_list,
        uncertainty_loss=uncertainty_loss,
        recall_values=recall_values,
        kappa_sh_matched=kappa_sh_matched,
    )
    pairwise_curves: List[Dict[str, Any]] = []
    if recall_values:
        pairwise_curves = compute_all_pairwise_rejection_curves(
            agg_data_list, recall_values
        )
    query_path, pw_saved = plot_rejection_curve_sets(
        query_curves,
        pairwise_curves,
        output_path,
        title=title,
    )
    pairwise_result: Dict[str, Any] = {}
    if pairwise_curves and pw_saved is not None:
        pairwise_result = {"curves": pairwise_curves, "path": pw_saved}
    return query_curves, query_path, pairwise_result


def _value_at_rejection_pct(curve: Dict[str, Any], pct: float) -> float:
    """Interpolate correctness / pair-hit rate at a rejection percentage."""
    x = np.asarray(curve.get("rejection_percentages", []), dtype=np.float64)
    y = np.asarray(curve.get("correctness_ratio", []), dtype=np.float64)
    if not x.size or not y.size:
        return float("nan")
    return float(np.interp(float(pct), x, y))


def deltas_at_rejection_pcts(
    curve: Dict[str, Any],
    pcts: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Return curve values and deltas from 0% rejection."""
    y0 = _value_at_rejection_pct(curve, 0.0)
    output = {"correctness_at_0": y0}
    for pct in normalize_rejection_delta_pcts(pcts):
        value = _value_at_rejection_pct(curve, pct)
        output[f"correctness_at_{pct}"] = value
        output[f"delta_at_{pct}"] = (
            value - y0 if np.isfinite(value) and np.isfinite(y0) else float("nan")
        )
    return output


def collect_rejection_curve_stats(
    query_curves: Sequence[Dict[str, Any]],
    pairwise_result: Optional[Dict[str, Any]] = None,
    *,
    rejection_delta_pcts: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Summarize query-level and pair-level rejection curves for logging / W&B tables."""
    pcts = normalize_rejection_delta_pcts(rejection_delta_pcts)
    stats: Dict[str, Any] = {}
    for curve in query_curves:
        if skip_rejection_method(_curve_method_name(curve)):
            continue
        if not curve.get("correctness_ratio", np.array([])).size:
            continue
        values = deltas_at_rejection_pcts(curve, pcts)
        entry: Dict[str, Any] = {
            "improvement": float(curve["improvement"]),
            "level": "query",
            **values,
        }
        if curve.get("method") is not None:
            entry["method"] = curve.get("method")
        if curve.get("n_rank") is not None:
            entry["n_rank"] = int(curve["n_rank"])
        stats[str(curve["label"])] = entry
    for curve in (pairwise_result or {}).get("curves") or []:
        if not curve.get("correctness_ratio", np.array([])).size:
            continue
        values = deltas_at_rejection_pcts(curve, pcts)
        pair_entry: Dict[str, Any] = {
            "improvement": float(curve["improvement"]),
            "level": "match",
            "method": curve.get("method"),
            "n_rank": curve.get("n_rank"),
            **values,
        }
        for key, val in values.items():
            if key.startswith("correctness_at_"):
                pair_entry["pair_hit_rate_at_" + key[len("correctness_at_"):]] = val
        stats[str(curve["label"])] = pair_entry
    return stats


def query_rejection_curves_for_scores(
    predictions: np.ndarray,
    positives_per_query,
    scores: np.ndarray,
    *,
    method_name: str,
    higher_is_confidence: bool,
    recall_values: Optional[Sequence[int]] = None,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Query-level rejection curves for one method's 1D scores. No plotting."""
    if skip_rejection_method(method_name):
        return []
    curves: List[Dict[str, Any]] = []
    for n_rank in (recall_values or [1]):
        matched = shard_matched_vector(predictions, positives_per_query, n_rank)
        curve = compute_rejection_curve(
            matched,
            scores,
            higher_is_confidence=higher_is_confidence,
            rejection_percentages=rejection_percentages,
        )
        curve["method"] = method_name
        curve["n_rank"] = int(n_rank)
        curve["level"] = "query"
        curve["label"] = query_curve_label(method_name, n_rank)
        curves.append(curve)
    return curves


def pairwise_rejection_curves_for_scores(
    predictions: np.ndarray,
    positives_per_query,
    pair_scores: np.ndarray,
    *,
    pairwise_name: str,
    higher_is_confidence: bool,
    recall_values: Optional[Sequence[int]] = None,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Pair-level rejection curves for one method's pair matrix. No plotting."""
    ranks = pairwise_ranks(recall_values or [])
    curves: List[Dict[str, Any]] = []
    for n_rank in ranks:
        y, s = pool_pairs_up_to_rank(
            predictions, positives_per_query, pair_scores, n_rank
        )
        curve = compute_rejection_curve(
            y,
            s,
            higher_is_confidence=higher_is_confidence,
            rejection_percentages=rejection_percentages,
        )
        curve["label"] = f"{pairwise_name} R@{int(n_rank)}"
        curve["method"] = pairwise_name
        curve["n_rank"] = int(n_rank)
        curve["level"] = "match"
        curves.append(curve)
    return curves


def save_rejection_stats_json(
    stats: Dict[str, Any],
    output_path: PathLike,
    *,
    dataset_name: Optional[str] = None,
) -> Path:
    """Persist rejection summary data for static reports."""
    import json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"stats": stats}
    if dataset_name is not None:
        payload["dataset"] = dataset_name
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def compute_combined_rejection_curves(
    variants: Sequence[tuple],
    *,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Pool shard-wise matched labels and scores; one curve per variant.

    Each variant is ``(label, shards_matched, shards_scores, higher_is_confidence)``.
    """
    curves: List[Dict[str, Any]] = []
    for spec in variants:
        if len(spec) != 4:
            continue
        label, shards_matched, shards_scores, higher_is_confidence = spec
        if len(shards_matched) != len(shards_scores) or not shards_matched:
            continue
        y = np.concatenate([
            np.asarray(m, dtype=np.float64).reshape(-1) for m in shards_matched
        ])
        s = np.concatenate([
            np.asarray(x, dtype=np.float64).reshape(-1) for x in shards_scores
        ])
        curve = compute_rejection_curve(
            y,
            s,
            higher_is_confidence=bool(higher_is_confidence),
            rejection_percentages=rejection_percentages,
        )
        curve["label"] = str(label)
        curves.append(curve)
    return curves


def save_combined_rejection_curves(
    variants: Sequence[tuple],
    output_path: PathLike,
    *,
    title: Optional[str] = None,
    rejection_percentages: Optional[Sequence[float]] = None,
) -> tuple[List[Dict[str, Any]], Optional[Path]]:
    """Compute pooled rejection curves and save a multi-method plot."""
    curves = compute_combined_rejection_curves(
        variants,
        rejection_percentages=rejection_percentages,
    )
    if not curves:
        return [], None
    out = Path(output_path)
    plot_rejection_curves(curves, output_path=out, title=title)
    return curves, out


def create_rejection_curves(
    matched_top1: np.ndarray,
    scores: np.ndarray,
    *,
    label: str = "",
    higher_is_confidence: bool = False,
    rejection_percentages: Optional[Sequence[float]] = None,
    output_path: Optional[PathLike] = None,
    title: Optional[str] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """Compute a rejection curve and optionally save a plot."""
    curve = compute_rejection_curve(
        matched_top1,
        scores,
        higher_is_confidence=higher_is_confidence,
        rejection_percentages=rejection_percentages,
    )
    curve["label"] = label

    if plot:
        plot_rejection_curves([curve], output_path=output_path, title=title)

    return curve


def _demo() -> None:
    rng = np.random.default_rng(0)
    n = 500
    matched = rng.random(n) < 0.55
    uncertainty = rng.exponential(1.0, size=n)
    uncertainty[~matched] += 0.8

    stun = create_rejection_curves(
        matched.astype(np.float64),
        uncertainty,
        label="Ours: STUN",
        higher_is_confidence=False,
        output_path="logs/rejection_curve_demo.png",
        title="Rejection curves (demo)",
    )
    print(f"STUN improvement: {stun['improvement']:+.1f} pp")
    print(f"Saved plot to logs/rejection_curve_demo.png")


if __name__ == "__main__":
    _demo()
