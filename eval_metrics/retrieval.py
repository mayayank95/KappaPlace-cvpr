"""FAISS retrieval and recall / mAP computation."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from eval_metrics.eval_ece_sh import _cal_mapk
from eval_metrics.recall import _cal_recall

logger = logging.getLogger(__name__)


def _retrieval_search_k(args: Dict[str, Any], num_database: int) -> int:
    max_recall_k = max(args["recall_values"])
    return min(max_recall_k, num_database)


def run_retrieval(
    args: Dict[str, Any],
    test_ds,
    db_desc: np.ndarray,
    q_desc: np.ndarray,
    dataset_output_dir: Path,
    save_recalls: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[List], Optional[List]]:
    """
    Run FAISS search and compute recalls (and optionally mAP).

    Returns (predictions, distances, recalls, recalls_str, positives_per_query, map_at_k).
    """
    faiss_index = faiss.IndexFlatL2(args["descriptors_dimension"])
    faiss_index.add(db_desc)
    max_recall_k = max(args["recall_values"])
    search_k = _retrieval_search_k(args, db_desc.shape[0])
    distances, predictions = faiss_index.search(q_desc, search_k)

    recalls_str = "Labels not available"
    recalls = np.zeros(len(args["recall_values"]))
    positives_per_query = None
    map_at_k = None

    if args["use_labels"]:
        positives_per_query = test_ds.get_positives()[: len(q_desc)]
        drop_empty_gt = bool(args.get("drop_empty_gt", False))
        recalls = _cal_recall(
            predictions[:, :max_recall_k],
            positives_per_query,
            args["recall_values"],
            drop_empty_gt=drop_empty_gt,
        )
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args["recall_values"], recalls)])

        if not args.get("only_recalls"):
            map_at_k = [
                _cal_mapk(
                    predictions[:, :max_recall_k],
                    positives_per_query,
                    n,
                    drop_empty_gt=drop_empty_gt,
                )
                for n in args["recall_values"]
            ]

        if save_recalls and not args["dry_run"]:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            (dataset_output_dir / "recalls.txt").write_text(recalls_str)

    keep_k = max_recall_k
    if predictions.shape[1] > keep_k:
        predictions = predictions[:, :keep_k]
        distances = distances[:, :keep_k]

    distances = np.asarray(distances, dtype=np.float64)

    return predictions, distances, recalls, recalls_str, positives_per_query, map_at_k


def log_retrieval_results(
    dataset_name: str,
    recalls_str: str,
    recalls: np.ndarray,
    map_at_k: Optional[List],
    recall_values: List[int],
) -> None:
    """Log recall / mAP summary for a dataset."""
    if not recalls_str or recalls_str == "Labels not available":
        return
    logger.info("\033[1mRetrieval\033[0m")
    logger.info("  recall: %s", recalls_str)
    if map_at_k is not None:
        logger.info("  map: %s", ", ".join([f"mAP@{val}: {m:.2f}" for val, m in zip(recall_values, map_at_k)]))
    if recalls[0] == 0 and map_at_k is not None and map_at_k[0] == 0:
        logger.info(
            "R@1 and mAP@1 are 0 when no query has a positive at rank 1 "
            "(e.g. dry run with random descriptors, or frozen backbone with poor retrieval)."
        )
