#!/usr/bin/env python3
"""Spearman ρ of pair scores vs LoFTR inliers (rebuttal Table 2).

GPS-positive pairs only. Sample 200 GPS-positive queries per split (--n_queries);
pair each with up to 40 GT database images. Occupancy-limited splits (SF-XL
occlusion) use every eligible query. LoFTR outdoor on 640×480 grayscale, then
MAGSAC F-matrix (thresh 0.5); the inlier count is the geometric target. Scores:
cosine / −L2² (same rank if descriptors are unit-norm),
joint_κ = √(κq²+κr²+2 κq κr cosθ), C_{Q↔R} = l2_joint_κ = κq κr cosθ.

Example:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 python -u loftr_inliers/kappa_inliers_correlation.py \\
    --config configs/datasets.json --datasets pitts30k,sf_xl,msls-val,amstertime \\
    --datasets_type test --n_queries 200 --max_gt_per_query 40 --seed 42 \\
    --model_mode uncertainty --uncertainty_loss vmf \\
    --var_head_agg --var_head_linear 1 --variance_activation softplus \\
    --backbone ResNet50 --descriptors_dimension 512 \\
    --resume_model logs/CosPlace/.../best_model_ece_recall_01.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature import LoFTR
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_paths import list_query_folders, query_folder_display_name, resolve_eval_path
from data.test_dataset import TestDataset
from eval_metrics.extraction import extract_descriptors
from utils.runtime import build_config_and_datasets, init_model

logger = logging.getLogger("loftr_inliers")

SCORE_KEYS = (
    "cosine",
    "neg_l2",
    "joint_kappa",
    "l2_joint_kappa",
)


def parse_study_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--n_queries",
        type=int,
        default=200,
        help="GPS-positive queries sampled per split (protocol: 200)",
    )
    p.add_argument("--max_gt_per_query", type=int, default=40)
    p.add_argument("--ransac_thresh", type=float, default=0.5)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Run output root. Default: <resume_model_dir>/loftr_inliers",
    )
    study, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return study


def _study_output_root(cfg: Dict[str, Any], study: argparse.Namespace) -> Path:
    if study.output_dir:
        return Path(study.output_dir)
    resume = cfg.get("resume_model")
    if resume:
        return Path(resume).expanduser().resolve().parent / "loftr_inliers"
    raise SystemExit("Cannot resolve output dir: pass --output_dir or --resume_model")


def _load_gray_resized(path: str, size: Tuple[int, int] = (640, 480)) -> Optional[np.ndarray]:
    """LoFTR outdoor expects a single-channel 640×480 image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, size)


@torch.no_grad()
def _loftr_inliers(
    matcher: LoFTR,
    img0: np.ndarray,
    img1: np.ndarray,
    device: torch.device,
    ransac_thresh: float = 0.5,
) -> int:
    t0 = torch.from_numpy(img0).float()[None, None].to(device) / 255.0
    t1 = torch.from_numpy(img1).float()[None, None].to(device) / 255.0
    res = matcher({"image0": t0, "image1": t1})
    mk0 = res["keypoints0"].detach().cpu().numpy()
    mk1 = res["keypoints1"].detach().cpu().numpy()
    if len(mk0) < 8:
        return 0
    _, inl = cv2.findFundamentalMat(mk0, mk1, cv2.USAC_MAGSAC, ransac_thresh, 0.999, 10000)
    if inl is None:
        return 0
    return int(inl.sum())


def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(m.sum())}
    sp, _ = spearmanr(x[m], y[m])
    pe, _ = pearsonr(x[m], y[m])
    return {"spearman": float(sp), "pearson": float(pe), "n": int(m.sum())}


def _eval_dataset(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    matcher: LoFTR,
    entry_name: str,
    eval_name: str,
    eval_ds_path: Path,
    query_folder: Path,
    desc_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    study: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    test_ds = TestDataset(
        str(eval_ds_path / "database"),
        str(query_folder),
        positive_dist_threshold=int(cfg.get("positive_dist_threshold", 25)),
        image_size=cfg.get("image_size"),
        use_labels=True,
        resize_test_imgs=bool(cfg.get("resize_test_imgs", False)),
    )
    cached = desc_cache.get(entry_name)
    db_desc, q_desc, db_var, q_var = extract_descriptors(
        args=cfg,
        model=model,
        device=device,
        test_ds=test_ds,
        dataset_name=eval_name,
        base_dataset_name=entry_name,
        cached_db_desc=None if cached is None else cached[0],
        cached_db_var=None if cached is None else cached[1],
    )
    desc_cache[entry_name] = (db_desc, db_var)

    q_paths = test_ds.queries_paths
    db_paths = test_ds.database_paths
    positives = test_ds.get_positives()
    logger.info("[%s] %s", eval_name, test_ds)

    eligible = [i for i, g in enumerate(positives) if len(g) > 0]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    if not eligible:
        logger.warning("[%s] No GT queries; skip.", eval_name)
        return None
    take = min(int(study.n_queries), len(eligible))
    sampled_q = rng.choice(eligible, size=take, replace=False)
    max_gt = int(study.max_gt_per_query)
    ransac_thresh = float(study.ransac_thresh)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for qi in tqdm(sampled_q, desc=eval_name, leave=True):
        gts = np.asarray(positives[qi], dtype=np.int64)
        if len(gts) > max_gt:
            gts = rng.choice(gts, size=max_gt, replace=False)
        img0 = _load_gray_resized(q_paths[qi])
        if img0 is None:
            continue
        qd = q_desc[qi]
        kq = float(np.mean(q_var[qi]))
        for di in gts:
            img1 = _load_gray_resized(db_paths[int(di)])
            if img1 is None:
                continue
            dd = np.asarray(db_desc[int(di)], dtype=np.float32)
            kr = float(np.mean(db_var[int(di)]))
            cos = float(np.dot(qd, dd))
            l2 = float(np.sum((qd - dd) ** 2))
            joint = float(np.sqrt(max(kq**2 + kr**2 + 2.0 * kq * kr * cos, 0.0)))
            cross = float(kq * kr * cos)
            try:
                inl = _loftr_inliers(matcher, img0, img1, device, ransac_thresh)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LoFTR fail q=%s db=%s: %s", qi, di, exc)
                continue
            rows.append(
                {
                    "q_idx": int(qi),
                    "db_idx": int(di),
                    "inliers": int(inl),
                    "cosine": cos,
                    "neg_l2": -l2,
                    "joint_kappa": joint,
                    "l2_joint_kappa": cross,
                }
            )

    if not rows:
        logger.warning("[%s] No pairs evaluated; skip.", eval_name)
        return None

    arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in SCORE_KEYS}
    inl = np.asarray([r["inliers"] for r in rows], dtype=np.float64)
    corrs = {k: _corr(arr[k], inl) for k in SCORE_KEYS}
    elapsed = time.time() - t0
    logger.info(
        "%s: n_pairs=%d n_q=%d Spearman cosine=%.3f joint=%.3f l2_joint=%.3f (%.1f min)",
        eval_name,
        len(rows),
        take,
        corrs["cosine"]["spearman"],
        corrs["joint_kappa"]["spearman"],
        corrs["l2_joint_kappa"]["spearman"],
        elapsed / 60.0,
    )
    return {
        "dataset": eval_name,
        "n_queries_sampled": int(take),
        "n_pairs": int(len(rows)),
        "max_gt_per_query": max_gt,
        "elapsed_sec": float(elapsed),
        "correlations": corrs,
        "mean_inliers": float(np.mean(inl)),
        "rows": rows,
    }


def _plot_summary(results: Sequence[Dict[str, Any]], out_png: Path) -> None:
    methods = ["cosine", "joint_kappa", "l2_joint_kappa"]
    labels = ["Cosine/−L2", "joint_κ", "l2_joint_κ"]
    datasets = [r["dataset"] for r in results]
    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(datasets)), 5))
    for i, (m, lab) in enumerate(zip(methods, labels)):
        vals = [r["correlations"][m]["spearman"] for r in results]
        ax.bar(x + (i - 1.0) * width, vals, width, label=lab)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("Spearman vs LoFTR inliers")
    ax.set_title("Score ↔ geometric inliers (GT pairs)")
    ax.legend()
    ax.set_ylim(-0.2, 1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    study = parse_study_args()
    cfg, entries, datasets_paths = build_config_and_datasets()
    if not cfg.get("resume_model"):
        raise SystemExit("--resume_model is required")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = _study_output_root(cfg, study)
    out_dir = output_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Output: %s", out_dir)
    logger.info("n_queries=%d max_gt_per_query=%d", study.n_queries, study.max_gt_per_query)

    device, model = init_model(cfg)
    model.eval()
    matcher = LoFTR(pretrained="outdoor").to(device).eval()
    logger.info("LoFTR on %s", device)

    filter_names = cfg.get("eval_query_folders")
    results: List[Dict[str, Any]] = []
    desc_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for entry in entries:
        name = entry["name"]
        eval_ds_path = resolve_eval_path(datasets_paths, name)
        query_folders = list_query_folders(eval_ds_path, filter_names)
        if not query_folders:
            logger.warning("[%s] No query folders; skip.", name)
            continue
        for q_folder in query_folders:
            eval_name = query_folder_display_name(name, q_folder.name, query_folders)
            res = _eval_dataset(
                cfg,
                model,
                device,
                matcher,
                name,
                eval_name,
                eval_ds_path,
                q_folder,
                desc_cache,
                study,
            )
            if res is None:
                continue
            slug = eval_name.replace(" ", "_").replace("/", "_")
            with (out_dir / f"{slug}_pairs.pkl").open("wb") as f:
                pickle.dump(res["rows"], f)
            results.append({k: v for k, v in res.items() if k != "rows"})

    payload = {
        "created": stamp,
        "resume_model": str(cfg.get("resume_model")),
        "n_queries": int(study.n_queries),
        "max_gt_per_query": int(study.max_gt_per_query),
        "seed": int(cfg.get("seed", 0)),
        "ransac_thresh": float(study.ransac_thresh),
        "datasets": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Spearman vs LoFTR inliers ===", flush=True)
    hdr = f"{'dataset':<22} {'n':>6} " + " ".join(f"{k:>14}" for k in SCORE_KEYS)
    print(hdr, flush=True)
    for r in results:
        cells = [
            f"{r['correlations'][k]['spearman']:14.4f}"
            if np.isfinite(r["correlations"][k]["spearman"])
            else f"{'nan':>14}"
            for k in SCORE_KEYS
        ]
        print(f"{r['dataset']:<22} {r['n_pairs']:6d} " + " ".join(cells), flush=True)

    if results:
        _plot_summary(results, out_dir / "spearman_summary.png")
        shutil.copy2(out_dir / "spearman_summary.png", output_root / "kappa_inliers_spearman.png")
        (output_root / "kappa_inliers_correlation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    logger.info("Wrote %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
