"""Recall@K helpers shared by retrieval, ECE, and rejection curves."""
from typing import List, Optional

import numpy as np


def has_ground_truth_mask(
    positives_per_query: List,
    n_queries: Optional[int] = None,
) -> np.ndarray:
    """Boolean mask: True iff the query has at least one ground-truth positive."""
    n = int(n_queries if n_queries is not None else len(positives_per_query))
    mask = np.zeros(n, dtype=bool)
    n_avail = min(n, len(positives_per_query))
    if n_avail:
        mask[:n_avail] = [len(p) > 0 for p in positives_per_query[:n_avail]]
    return mask


def keep_labeled_queries(positives_per_query, *arrays, n_queries: Optional[int] = None):
    """Drop queries with no GT positives. Returns ``(positives, *arrays)``."""
    if n_queries is None:
        if arrays and arrays[0] is not None:
            n_queries = int(np.asarray(arrays[0]).shape[0])
        else:
            n_queries = len(positives_per_query)
    mask = has_ground_truth_mask(positives_per_query, n_queries)
    keep = np.flatnonzero(mask)
    positives = [
        positives_per_query[int(i)]
        for i in keep
        if int(i) < len(positives_per_query)
    ]
    filtered = tuple(
        None if a is None else np.asarray(a)[mask] for a in arrays
    )
    return (positives, *filtered)


def recall_hits_at_k(
    predictions: np.ndarray,
    positives_per_query: List,
    n_rank: int = 1,
) -> np.ndarray:
    """Per-query binary recall@K hit (1 if any GT in top-K, else 0).

    Length equals ``predictions.shape[0]``. Queries with no GT positives are 0.
    """
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    nq = preds.shape[0]
    if nq == 0:
        return np.zeros(0, dtype=np.float64)
    hits = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        pos = positives_per_query[i] if i < len(positives_per_query) else ()
        if len(pos) == 0:
            continue
        hits[i] = 1.0 if np.any(np.isin(preds[i, :int(n_rank)], pos)) else 0.0
    return hits


def cal_recall(
    predictions: np.ndarray,
    positives_per_query: List,
    n_values: List[int],
    *,
    drop_empty_gt: bool = False,
) -> np.ndarray:
    """Compute recall@K. Returns array of shape [len(n_values)], values in [0, 100].

    Mean of ``recall_hits_at_k``. Empty-GT queries are 0; they are included in
    the mean unless ``drop_empty_gt`` is True.
    """
    recalls = np.zeros(len(n_values))
    preds = np.asarray(predictions)
    num_queries = preds.shape[0]
    if num_queries == 0 or not n_values:
        return recalls
    if not drop_empty_gt:
        for i, n in enumerate(n_values):
            hits = recall_hits_at_k(preds, positives_per_query, n_rank=int(n))
            recalls[i] = float(np.mean(hits) * 100.0)
        return recalls
    gt_mask = has_ground_truth_mask(positives_per_query, num_queries)
    if not np.any(gt_mask):
        return recalls
    for i, n in enumerate(n_values):
        hits = recall_hits_at_k(preds, positives_per_query, n_rank=int(n))
        recalls[i] = float(np.mean(hits[gt_mask]) * 100.0)
    return recalls


# Back-compat alias used by older call sites.
_cal_recall = cal_recall
