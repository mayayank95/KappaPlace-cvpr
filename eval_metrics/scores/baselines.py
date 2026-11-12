"""Baseline uncertainty scores (L2 / PA / SUE) from retrieval distances."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


def _to_l2(distances: np.ndarray) -> np.ndarray:
    """FAISS IndexFlatL2 returns squared L2; convert to L2 (same scale as sue.py)."""
    return np.sqrt(np.maximum(distances, 0))


def compute_l2_scores(distances: np.ndarray) -> Dict[str, np.ndarray]:
    """L2 baseline scores: top-1 distance per query and full top-K pair distances."""
    dists = _to_l2(distances)
    return {"per_query": dists[:, 0], "pair": dists}


def compute_pa_scores(distances: np.ndarray) -> Optional[np.ndarray]:
    """PA-score = d1/d2 per query. Returns None when fewer than 2 retrievals available."""
    dists = _to_l2(distances)
    if dists.shape[1] < 2:
        return None
    return dists[:, 0] / dists[:, 1]


SUE_NUM_NN = 10  # SUE uses the top-10 retrieved poses (sue.py ``num_NN``).

# Filenames use 1-D mock UTM (not GPS). SUE is not computed there.
_SKIP_SUE_DATASETS = ("nordland", "amstertime", "spedtest")


def skip_sue_dataset(name: Optional[str]) -> bool:
    """True for Nordland / AmsterTime / SPED (and shard names that include them)."""
    n = str(name or "").lower()
    return any(tok in n for tok in _SKIP_SUE_DATASETS)


def compute_sue_scores(
    distances: np.ndarray,
    predictions: np.ndarray,
    database_utms: np.ndarray,
    *,
    num_nn: int = SUE_NUM_NN,
    slope: float = 350.0,
) -> Optional[np.ndarray]:
    """SUE spatial variance score per query (sue.py lines 95-131).

    Returns None when the retrieval pool has fewer than ``num_nn`` neighbors.
    """
    dists = _to_l2(distances)
    if dists.shape[1] < num_nn:
        return None
    total_queries = len(predictions)
    sue_scores = np.zeros(total_queries)
    weights = np.ones(num_nn)

    for itr in range(total_queries):
        top_preds = predictions[itr][:num_nn]
        nn_poses = database_utms[top_preds]

        for itr2 in range(num_nn):
            weights[itr2] = math.e ** ((-1 * abs(dists[itr][itr2])) * slope)
        weights = weights / sum(abs(weights))

        mean_pose = np.asarray([
            np.average(nn_poses[:, 0], weights=weights),
            np.average(nn_poses[:, 1], weights=weights),
        ])

        variance_lat_lat = 0.0
        variance_lon_lon = 0.0
        for k in range(num_nn):
            diff_lat = min(500, nn_poses[k, 0] - mean_pose[0])
            diff_lon = min(500, nn_poses[k, 1] - mean_pose[1])
            variance_lat_lat += weights[k] * diff_lat ** 2
            variance_lon_lon += weights[k] * diff_lon ** 2

        sue_scores[itr] = (variance_lat_lat + variance_lon_lon) / 2

    return sue_scores
