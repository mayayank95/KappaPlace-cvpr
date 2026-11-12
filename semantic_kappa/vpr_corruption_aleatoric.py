#!/usr/bin/env python3
"""Multi-corruption VPR aleatoric study: query-only noise → κ.

Rebuttal protocol: sample 100 random queries per split (--max_queries), no R@1
filter. Occupancy-limited splits (SF-XL occlusion) use every query. Corrupt the
query only; the database is unused. If κ captures aleatoric uncertainty,
increasing severity should lower κ.

Example:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 python -u semantic_kappa/vpr_corruption_aleatoric.py \\
    --datasets pitts30k,msls-val,sf_xl,amstertime --max_queries 100 \\
    --model_mode uncertainty --uncertainty_loss vmf \\
    --var_head_agg --var_head_linear 1 --variance_activation softplus \\
    --backbone ResNet50 --descriptors_dimension 512 \\
    --resume_model logs/CosPlace/.../best_model_ece_recall_01.pth
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corruptions import CORRUPTIONS
from data.dataset_paths import list_query_folders, query_folder_display_name, resolve_eval_path
from data.test_dataset import TestDataset, build_eval_transform
from utils import commons
from utils.runtime import build_config_and_datasets, init_model

logger = logging.getLogger("vpr_corruption_aleatoric")


def parse_study_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--max_queries",
        type=int,
        default=100,
        help="Random queries sampled per split (protocol: 100)",
    )
    p.add_argument(
        "--corruptions",
        type=str,
        default="blur,noise,jpeg,lowlight,occlusion,downsample",
        help="Comma-separated corruption names",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Timestamped run output root. Default: "
        "<resume_model_dir>/vpr_corrupt_kappa",
    )
    study, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return study


def _study_output_root(cfg: Dict[str, Any], study) -> Path:
    """Write next to the checkpoint, not under eval/ or reports/."""
    if study.output_dir:
        return Path(study.output_dir)
    resume = cfg.get("resume_model")
    if resume:
        return Path(resume).expanduser().resolve().parent / "vpr_corrupt_kappa"
    raise SystemExit("Cannot resolve output dir: pass --output_dir or --resume_model")


def mean_kappa(variances: np.ndarray) -> np.ndarray:
    v = np.asarray(variances, dtype=np.float64)
    if v.ndim == 1:
        return v
    return v.mean(axis=1)


def apply_corruption(
    pil: Image.Image,
    name: str,
    level: float,
    *,
    rng: np.random.Generator,
) -> Image.Image:
    """Corrupt a query image. The first ladder level is clean and returns the image unchanged."""
    img = pil.convert("RGB")
    if name == "blur":
        if float(level) <= 0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=float(level)))

    if name == "noise":
        if float(level) <= 0:
            return img
        arr = np.asarray(img, dtype=np.float32)
        noise = rng.normal(0.0, float(level), size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if name == "jpeg":
        q = int(level)
        if q >= 95:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(1, min(95, q)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if name == "lowlight":
        gain = float(level)
        if gain >= 0.999:
            return img
        darkened = ImageEnhance.Brightness(img).enhance(gain)
        arr = np.asarray(darkened, dtype=np.float32)
        # Mild sensor noise grows as light drops.
        sigma = 8.0 * (1.0 - gain)
        if sigma > 0:
            arr = arr + rng.normal(0.0, sigma, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if name == "occlusion":
        frac = float(level)
        if frac <= 0:
            return img
        w, h = img.size
        bw = max(1, int(round(math.sqrt(frac) * w)))
        bh = max(1, int(round(math.sqrt(frac) * h)))
        x0 = int(rng.integers(0, max(1, w - bw + 1)))
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0, y0, x0 + bw - 1, y0 + bh - 1], fill=(0, 0, 0))
        return out

    if name == "downsample":
        scale = float(level)
        if scale >= 0.999:
            return img
        w, h = img.size
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = img.resize((nw, nh), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    raise ValueError(f"Unknown corruption: {name}")


def sample_query_indices(n_queries: int, max_queries: int, seed: int) -> np.ndarray:
    """Uniform random query indices (no retrieval / R@1 filter)."""
    n = int(n_queries)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if n <= max_queries:
        return np.arange(n, dtype=np.int64)
    chosen = rng.choice(n, size=int(max_queries), replace=False)
    return np.sort(chosen.astype(np.int64))


@torch.inference_mode()
def forward_batch(
    model: torch.nn.Module,
    device: torch.device,
    tensors: Sequence[torch.Tensor],
) -> np.ndarray:
    batch = torch.stack(list(tensors), dim=0).to(device)
    _desc, var = model(batch)
    return mean_kappa(var.detach().float().cpu().numpy())


def extract_corrupt_kappas(
    model: torch.nn.Module,
    device: torch.device,
    paths: Sequence[str],
    corrupt_name: str,
    level: float,
    transform: transforms.Compose,
    seed: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Corrupt query images at ``level`` and return κ."""
    kappas: List[float] = []
    pending: List[torch.Tensor] = []
    for i, path in enumerate(paths):
        rng = np.random.default_rng(seed + 17 * i)
        pil = Image.open(path).convert("RGB")
        pil = apply_corruption(pil, corrupt_name, level, rng=rng)
        pending.append(transform(pil))
        if len(pending) >= batch_size:
            kappas.extend(forward_batch(model, device, pending).tolist())
            pending = []
    if pending:
        kappas.extend(forward_batch(model, device, pending).tolist())
    return np.asarray(kappas, dtype=np.float64)


def summarize_corruption(
    levels: Sequence[float],
    kappa_mat: np.ndarray,
) -> Dict[str, Any]:
    """kappa_mat: [n_queries, n_levels], columns already clean → hardest."""
    rel = kappa_mat / kappa_mat[:, :1]
    mean_k = np.nanmean(kappa_mat, axis=0)
    mean_rel = np.nanmean(rel, axis=0)

    diffs = np.diff(kappa_mat, axis=1)
    mono = np.all(diffs <= 1e-6, axis=1) if diffs.size else np.ones(len(kappa_mat), dtype=bool)
    frac_mono = float(np.mean(mono)) if len(mono) else float("nan")

    n_levels = kappa_mat.shape[1]
    sev = np.tile(np.arange(n_levels, dtype=np.float64), (kappa_mat.shape[0], 1)).reshape(-1)
    k_flat = kappa_mat.reshape(-1)
    rho, p = stats.spearmanr(sev, k_flat)

    return {
        "levels": [float(x) for x in levels],
        "mean_kappa": mean_k.tolist(),
        "mean_relative_kappa": mean_rel.tolist(),
        "frac_kappa_nonincreasing": frac_mono,
        "spearman_severity_vs_kappa": float(rho),
        "spearman_p": float(p),
    }


def plot_corruption_curves(
    out_png: Path,
    levels: Sequence[float],
    mean_rel: Sequence[float],
    title: str,
    param: str,
) -> None:
    """Save mean κ/κ₀ versus the corruption ladder."""
    x = np.arange(len(levels))
    labels = [f"{v:g}" for v in levels]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(x, mean_rel, "o-", color="#1f77b4", linewidth=2, label="mean κ/κ₀")
    ax1.set_ylabel("mean relative κ")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(param)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def process_dataset(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    entry_name: str,
    display_name: str,
    eval_ds_path: Path,
    query_folder: Path,
    study: argparse.Namespace,
    corrupt_names: Sequence[str],
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    database_folder = str(eval_ds_path / "database")
    test_ds = TestDataset(
        database_folder,
        str(query_folder),
        positive_dist_threshold=25,
        image_size=cfg.get("image_size"),
        use_labels=False,
        resize_test_imgs=bool(cfg.get("resize_test_imgs", False)),
    )
    logger.info("[%s] %s", display_name, test_ds)
    seed = int(cfg.get("seed", 0))
    image_size = int(cfg.get("image_size") or 512)
    # Square resize so corrupted queries can be batched (SF-XL sizes vary).
    transform = build_eval_transform((image_size, image_size))

    n_queries = len(test_ds.queries_paths)
    sel = sample_query_indices(n_queries, int(study.max_queries), seed)
    if sel.size == 0:
        logger.warning("[%s] No queries; skip.", display_name)
        return None
    paths = [test_ds.queries_paths[int(i)] for i in sel]
    out: Dict[str, Any] = {
        "dataset": display_name,
        "base_dataset": entry_name,
        "query_folder": query_folder.name,
        "n_queries": n_queries,
        "n_selected": int(len(sel)),
        "selected_indices": sel.tolist(),
        "corruptions": {},
    }

    for cname in corrupt_names:
        meta = CORRUPTIONS[cname]
        levels = meta["levels"]
        logger.info("[%s] corruption=%s levels=%s", display_name, cname, levels)

        kappa_mat = np.zeros((len(paths), len(levels)), dtype=np.float64)
        for t, level in enumerate(levels):
            kappa_mat[:, t] = extract_corrupt_kappas(
                model,
                device,
                paths,
                cname,
                level,
                transform,
                seed=seed + 1009 * sum(ord(ch) for ch in cname),
                batch_size=int(cfg.get("infer_batch_size", 16)),
            )

        summary = summarize_corruption(levels, kappa_mat)
        curve_name = f"{display_name}_{cname}.png"
        plot_corruption_curves(
            run_dir / curve_name,
            summary["levels"],
            summary["mean_relative_kappa"],
            title=f"{display_name}: {meta['label']}",
            param=meta["param"],
        )
        summary["curve_png"] = curve_name
        out["corruptions"][cname] = summary

    return out


def main() -> None:
    study = parse_study_args()
    cfg, entries, datasets_paths = build_config_and_datasets()
    commons.make_deterministic(cfg["seed"])
    if not cfg.get("resume_model"):
        raise SystemExit("--resume_model is required")

    corrupt_names = [
        c.strip() for c in str(study.corruptions).split(",") if c.strip() in CORRUPTIONS
    ]
    if not corrupt_names:
        raise SystemExit(f"No valid corruptions; choose from {list(CORRUPTIONS)}")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = _study_output_root(cfg, study)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Output: %s", run_dir)
    logger.info("max_queries=%d corruptions=%s", study.max_queries, corrupt_names)

    device, model = init_model(cfg)
    model.eval()
    filter_names = cfg.get("eval_query_folders")
    if filter_names:
        logger.info("Query folders filter: %s", ", ".join(filter_names))
    dataset_results: List[Dict[str, Any]] = []

    for entry in entries:
        name = entry["name"]
        eval_ds_path = resolve_eval_path(datasets_paths, name)
        query_folders = list_query_folders(eval_ds_path, filter_names)
        if not query_folders:
            if filter_names:
                logger.warning(
                    "[%s] No query folders matching %s under %s",
                    name,
                    filter_names,
                    eval_ds_path,
                )
            else:
                logger.warning("[%s] No query folders found in %s", name, eval_ds_path)
            continue
        # Rebuttal tables used every query split (SF-XL v1/v2/night/occlusion).
        for q_folder in query_folders:
            display = query_folder_display_name(name, q_folder.name, query_folders)
            result = process_dataset(
                cfg,
                model,
                device,
                name,
                display,
                eval_ds_path,
                q_folder,
                study,
                corrupt_names,
                run_dir,
            )
            if result is not None:
                dataset_results.append(result)

    payload = {
        "run_name": run_name,
        "resume_model": str(cfg.get("resume_model")),
        "backbone": cfg.get("backbone"),
        "max_queries": int(study.max_queries),
        "corruption_names": corrupt_names,
        "datasets": dataset_results,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", run_dir / "results.json")


if __name__ == "__main__":
    main()
