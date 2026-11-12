"""Uncertainty evaluation tasks (ECE, rejection) selected by --eval_metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

UNCERTAINTY_EVAL_METRICS: tuple[str, ...] = ("ece", "rejection")


def pairwise_ranks(recall_values: Sequence[int]) -> List[int]:
    """Pair-level @1 is the same sample as query-level; drop it for every pairwise eval."""
    return [int(n) for n in recall_values if int(n) != 1]


def is_pairwise_rank_flat_key(key: str) -> bool:
    """True for stored ``pairwise_*_r{K}`` copies; eval flattens from the 2D matrix instead."""
    name = str(key)
    if not name.startswith("pairwise_"):
        return False
    prefix, sep, rest = name.rpartition("_r")
    return bool(sep) and rest.isdigit()


def parse_eval_metrics_arg(value: Any) -> Optional[List[str]]:
    """Parse CLI/JSON ``eval_metrics`` into canonical names. ``None`` stays unspecified."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
    else:
        parts = [str(value).strip()]
    if not parts:
        return None
    if any(p.lower() == "all" for p in parts):
        return list(UNCERTAINTY_EVAL_METRICS)
    out: List[str] = []
    unknown: List[str] = []
    for part in parts:
        key = part.lower()
        if key not in UNCERTAINTY_EVAL_METRICS:
            unknown.append(part)
        elif key not in out:
            out.append(key)
    if unknown:
        raise ValueError(
            "Unknown eval_metrics: "
            + ", ".join(unknown)
            + f". Options: {', '.join(UNCERTAINTY_EVAL_METRICS)}, or all."
        )
    return out


def resolve_eval_metrics(cfg: Dict[str, Any], *, training: bool = False) -> List[str]:
    """Chosen evals: unspecified → all in eval, ``ece`` only in training."""
    chosen = parse_eval_metrics_arg(cfg.get("eval_metrics"))
    if not chosen:
        chosen = ["ece"] if training else list(UNCERTAINTY_EVAL_METRICS)
    return chosen
