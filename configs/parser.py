from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from eval_metrics.eval_tasks import parse_eval_metrics_arg
from eval_metrics.rejection_curves import normalize_rejection_delta_pcts
from models.cosplace_uncertainty.cosplace_model.transformer_backbones import (
    DINOV2_BACKBONES,
    DINOV2_IMAGE_SIZE,
    DINOV2_PATCH,
)
from utils.early_stop_utils import canonical_early_stop_metrics, recall_values_needed_for_metrics

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    p = argparse.ArgumentParser(description="Config-first CLI: JSON config + CLI overrides.")

    # Config & paths
    p.add_argument(
        "--config",
        type=str,
        default="configs/datasets.json",
        help="Path to datasets config JSON (CosPlace: configs/datasets.json, MixVPR: configs/datasets_mixvpr.json, SALAD: configs/datasets_salad.json)",
    )
    p.add_argument("--save_config", action="store_true", help="Save merged configuration to logs folder")
    p.add_argument("--data_folder", type=str, default=None, help="Root folder containing raw datasets")
    p.add_argument("--logs_folder", type=str, default=None, help="Folder to save logs")
    p.add_argument("--dry_run", action="store_true", help="Print actions without performing file operations")
    p.add_argument("--debug", action="store_true", help="Enable extra debug-only checks and computations.")

    # Logging
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="UncertaintyAwareVPR", help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto from define function in runtime.py)")

    # Datasets
    p.add_argument("--datasets", type=str, default="all", help='Datasets to process (e.g. "all", "sf_xl", or "sf_xl,pitts30k")')
    p.add_argument("--datasets_type", type=str, default="all", help='Dataset splits to use (e.g. "all", "train", "test", "val")')

    # System
    p.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    p.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    p.add_argument("--seed", type=int, default=0, help="Random seed (-1 to disable deterministic mode)")
    p.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark (faster but non-deterministic)")

    # Model
    p.add_argument("--method", type=str, default="cosplace", help="Model method (e.g. cosplace, mixvpr, cosplace_pretrained, salad)")
    p.add_argument(
        "--backbone",
        type=str,
        default="ResNet18",
        help="Backbone: ResNet*/VGG16/EfficientNet_B* (CNN), ViT "
        "(CosPlace paper transformer), or DINOv2-S/B/L/G",
    )
    p.add_argument(
        "--descriptors_dimension",
        type=int,
        default=512,
        help="Descriptor dim (paper: 768 for ViT; DINOv2 defaults to its embed dim; CNN default 512)",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Square resize (paper: 224 for ViT and DINOv2; CNN default 512). "
        "DINOv2 requires a multiple of 14.",
    )
    p.add_argument(
        "--descriptor_activation",
        type=str,
        default="tanh",
        choices=["tanh", "gelu"],
        help="Activation after the transformer CLS linear layer. "
        "tanh matches the CosPlace paper; gelu is the alternate head.",
    )
    p.add_argument(
        "--descriptor_extra_linear",
        action="store_true",
        help="Add a second linear layer after the transformer CLS activation, "
        "even when the descriptor size already matches the CLS size "
        "(Linear → activation → Linear).",
    )
    p.add_argument("--train_all_layers", action="store_true", help="Train all backbone layers (default: freeze early layers)")
    p.add_argument("--resize_test_imgs", action="store_true", help="Resize test images to (image_size x image_size)")

    # Resume / checkpoint
    p.add_argument("--resume_train", type=str, default=None, help="Path to training checkpoint, e.g. logs/.../last_checkpoint.pth")
    p.add_argument("--resume_model", type=str, default=None, help="Path to model weights, e.g. logs/.../best_model.pth")
    p.add_argument("--ckpt_state_dict_key", type=str, default="model_state_dict",
        help="Key for state dict in checkpoint (default: model_state_dict, common alternatives: state_dict)")
    p.add_argument("--load_classifiers", type=str, default=None, help="Path to checkpoint to load classifier weights from. Frozen unless --trainable_classifiers is set.")
    p.add_argument("--trainable_classifiers", action="store_true", help="With --load_classifiers, keep the loaded class matrix trainable.")

    # CosPlace grouping
    p.add_argument("--M", type=int, default=10, help="Size of the cell in meters")
    p.add_argument("--alpha", type=int, default=30, help="Size of the margin in degrees")
    p.add_argument("--N", type=int, default=5, help="Min number of images per place")
    p.add_argument("--L", type=int, default=2, help="Smoothing for group boundaries")
    p.add_argument("--groups_num", type=int, default=8, help="Number of CosPlace spatial groups. Omit to use JSON config.")
    p.add_argument("--min_images_per_class", type=int, default=10, help="Minimum images per class for a group to be valid")

    # Training
    p.add_argument("--lr", type=float, default=0.00001, help="Learning rate for model optimizer")
    p.add_argument("--classifiers_lr", type=float, default=0.01, help="Learning rate for classifier optimizers")
    p.add_argument("--head_lr", type=float, default=1e-3,
        help="LR for head (var_head + final_l2) when freeze_model; ignored when freeze_model is off (uses --lr)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for training DataLoader")
    p.add_argument("--epochs_num", type=int, default=50, help="Total number of training epochs")
    p.add_argument("--iterations_per_epoch", type=int, default=10000, help="Number of training iterations per epoch")
    p.add_argument("--log_every_n_iterations", type=int, default=1000,
        help="Log training stats every N iterations within each epoch (rolling mean over last N); 0 disables.")
    p.add_argument("--losses", type=str, default="ce", help="Losses to use, comma-separated (e.g. 'ce', 'ce,uncertainty')")
    p.add_argument("--freeze_model", action="store_true", help="Freeze backbone/aggregation, train only the uncertainty head")
    p.add_argument("--freeze_batchnorm",  action="store_true",
        help="Freeze BatchNorm layers (set to eval mode) during training to keep running stats fixed.")
    p.add_argument("--patience", type=int, default=15, help="Patience for early stopping (epochs without improvement)")
    p.add_argument("--disable_early_stop", action="store_true", help="Disable early stopping and always run all epochs")
    p.add_argument("--early_stop_metric", type=str, default="recall",
        help="Early-stop metrics separated by commas (,). Example: recall,ece_recall. "
        "Tokens: recall; ece_recall (R@1 ECE); ece_recall_05; ece_recall_10. "
        "Each metric has its own patience counter; each saves best_model_<metric>.pth (best_model.pth when only recall).")
    p.add_argument("--phased_early_stop", action="store_true",
        help="Two-phase early stopping: Phase 1 tracks recall only; when recall plateaus "
        "(exhausts patience), Phase 2 activates the ECE metrics with fresh patience. "
        "Requires early_stop_metric to include both recall and at least one ece_recall_* metric.")

    # Data augmentation
    p.add_argument("--augmentation_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for data augmentation")
    p.add_argument("--brightness", type=float, default=0.7, help="ColorJitter brightness factor")
    p.add_argument("--contrast", type=float, default=0.7, help="ColorJitter contrast factor")
    p.add_argument("--hue", type=float, default=0.5, help="ColorJitter hue factor")
    p.add_argument("--saturation", type=float, default=0.7, help="ColorJitter saturation factor")
    p.add_argument("--random_resized_crop", type=float, default=0.5, help="RandomResizedCrop minimum scale (max is 1)")

    # Uncertainty
    p.add_argument("--model_mode", type=str, default="basic", help="Model mode: basic or uncertainty")
    p.add_argument("--uncertainty_lambda", type=float, default=1.0, help="Weight for the uncertainty loss")
    p.add_argument(
        "--uncertainty_lambda_mode",
        type=str,
        default="fixed",
        choices=["fixed", "kappa", "curriculum", "cosface", "grad_balance"],
        help="fixed multiplies the uncertainty loss by --uncertainty_lambda. "
        "kappa (vMF only) multiplies by uncertainty_lambda / mean(κ), detached, "
        "so the average pull on the descriptor cosine stays at uncertainty_lambda. "
        "curriculum is kappa scaling with a warmup until --uncertainty_curriculum_recall, "
        "then baby steps up to --uncertainty_lambda_final. "
        "cosface multiplies the whole vMF loss by detach(vmf_pull_alpha * s * mean(1-p)), "
        "where p is the target probability after the CosFace margin. "
        "grad_balance leaves both losses unscaled and, in the backward, rescales only "
        "the vMF descriptor gradient to vmf_grad_rho * EMA(||CosFace descriptor grad||).",
    )
    p.add_argument(
        "--uncertainty_lambda_final",
        type=float,
        default=None,
        help="Kappa mode only. If set, --uncertainty_lambda is the pull for the first "
        "--uncertainty_lambda_hold_epochs epochs, then the pull rises linearly to this "
        "value over the next --uncertainty_lambda_ramp_epochs epochs and stays there.",
    )
    p.add_argument(
        "--uncertainty_lambda_hold_epochs",
        type=int,
        default=30,
        help="Epochs to keep the pull at --uncertainty_lambda before the ramp. Used only with --uncertainty_lambda_final.",
    )
    p.add_argument(
        "--uncertainty_lambda_ramp_epochs",
        type=int,
        default=20,
        help="Epochs of the linear pull ramp after the hold. The last of these epochs is --uncertainty_lambda_final.",
    )
    p.add_argument(
        "--uncertainty_curriculum_recall",
        type=float,
        default=None,
        help="Curriculum mode. Keep the pull at --uncertainty_lambda until SF-XL validation "
        "R@1 reaches this value, then baby-step the pull.",
    )
    p.add_argument(
        "--uncertainty_curriculum_step_epochs",
        type=int,
        default=10,
        help="Epochs to stay at each baby step after the warmup. Used only in curriculum mode.",
    )
    p.add_argument(
        "--uncertainty_curriculum_factor",
        type=float,
        default=2.0,
        help="Multiply the pull by this factor at each baby step. Used only in curriculum mode.",
    )
    p.add_argument(
        "--vmf_pull_alpha",
        type=float,
        default=0.15,
        help="cosface mode. Weight on the CosFace positive-pair gradient magnitude. "
        "The vMF loss is multiplied by detach(alpha * s * mean(1 - p_target)). "
        "This is not the CosPlace heading width (--alpha).",
    )
    p.add_argument(
        "--vmf_grad_rho",
        type=float,
        default=1.0,
        help="grad_balance mode. Auxiliary budget: the vMF descriptor-gradient norm is "
        "set to rho times the EMA of the CosFace descriptor-gradient norm. "
        "The κ-head gradient is left unchanged.",
    )
    p.add_argument(
        "--vmf_grad_ema_decay",
        type=float,
        default=0.99,
        help="grad_balance mode. EMA decay for descriptor-gradient norms. "
        "0.99 keeps a long history; the first batch initializes the EMA.",
    )
    p.add_argument("--uncertainty_loss", type=str, default="vmf",
        help="Uncertainty loss type: gaussian_nll, gaussian_cosine, or vmf")
    p.add_argument("--gnll_mu_scale_mode", type=str, default="sqrt_dim", choices=["sqrt_dim", "none", "custom"],
        help="Scaling mode for Gaussian NLL mean vectors: sqrt_dim (default), none (1.0), or custom (use --gnll_mu_scale_value).")
    p.add_argument("--gnll_mu_scale_value", type=float, default=1.0,
        help="Custom scale value for Gaussian NLL mean vectors when --gnll_mu_scale_mode=custom.")
    # Variance head: orthogonal flags.
    p.add_argument("--var_head_agg", action="store_true",
        help="Prepend a deep-copy of the aggregation module to the variance head "
        "(operates on the backbone feature map instead of the descriptor). "
        "For SALAD the copy is a separate prenorm head: raw token and cluster "
        "sums, own weights, not the intra-L2 retrieval descriptor.")
    p.add_argument("--var_head_linear", type=str, default="1", choices=["none", "d", "1"],
        help="Linear layer in the variance head: 'none' (no Linear), "
        "'d' (Linear(fc_output_dim, fc_output_dim)), or '1' (Linear(fc_output_dim, 1)).")
    p.add_argument("--ece_two_sided", dest="ece_two_sided", action="store_true",
        help="KappaPlace default: two-sided p1–p99 clip for learned κ / joint κ / L2·κ when ECE "
        "bins are equal-width percentile. Baselines (L2 / PA / SUE) stay one-sided.")
    p.add_argument("--no_ece_two_sided", dest="ece_two_sided", action="store_false",
        help="One-sided percentile clip for the κ family as well (overrides the KappaPlace default).")
    p.set_defaults(ece_two_sided=True)
    p.add_argument("--ece_extras", action="store_true",
        help="Also compute nested ECE variants: equal-mass bins, the opposite percentile clip "
        "(one-sided vs two-sided), and −κ invert for vMF. Default off: KappaPlace protocol only "
        "(equal-width percentile; two-sided κ family, one-sided baselines).")
    p.add_argument("--var_init", action="store_true",
        help="Initialize variance head for stable start: small weights + bias tuned for ~0.1 initial variance")
    p.add_argument("--variance_activation", type=str, default=None, choices=["softplus", "sigmoid"],
        help="Activation for variance head output: softplus (positive, unbounded) or sigmoid (bounded [0,1])")

    # Evaluation
    p.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20], help="Recall@K values to compute")
    p.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for inference (validation and test)")
    p.add_argument(
        "--max_eval_queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries per query folder (debugging only; database stays complete).",
    )
    p.add_argument("--positive_dist_threshold", type=int, default=25, help="Distance in meters for a positive match")
    p.add_argument("--use_labels", action="store_true", help="Use UTM coordinates from image paths for evaluation")
    p.add_argument("--only_recalls", action="store_true", help="Only compute recalls, skip mAP and other metrics for speed")
    p.add_argument(
        "--eval_query_folders",
        type=str,
        default=None,
        help='Eval only these query folder names under each dataset (comma-separated), e.g. "query" or '
        '"queries,v1_test". Omit or use "all" to run every queries*/query* folder found.',
    )
    p.add_argument("--save_descriptors", action="store_true", help="Save extracted descriptors to disk")
    p.add_argument(
        "--eval_metrics",
        type=str,
        default=None,
        help="Uncertainty evaluations to run, comma-separated. "
        "Options: ece, rejection, or all. "
        "Default: all in eval.py; ece only during training.",
    )
    p.add_argument("--skip_rejection_curves", action="store_true",
        help="Skip rejection-curve plots in eval (deltas still run if rejection is in --eval_metrics).")
    p.add_argument(
        "--drop_empty_gt",
        action="store_true",
        help=        "Drop queries with no ground-truth positives from R@K, mAP, uncertainty eval, "
        "combined pool, and saved preds. Default: keep them (they count as misses).",
    )
    p.add_argument(
        "--rejection_delta_pcts",
        type=int,
        nargs="+",
        default=[20, 50, 90],
        help="Rejection-curve delta percentages to log (default: 20 50 90).",
    )
    p.add_argument("--skip_baselines", action="store_true", help="Skip expensive baseline calculations (L2, PA-score, SUE) in eval.py.")
    p.add_argument(
        "--skip_combined",
        action="store_true",
        help="Skip all combined-eval views (combined_6, combined_11, and extras).",
    )
    p.add_argument(
        "--combined_extras",
        action="store_true",
        help="Also run extra combined views: combined_all (drops Nordland / AmsterTime / SPED), "
        "combined_all_full (original all shards), combined_all_no_amst, combined_all_no_pitts, "
        "combined_7. Default: combined_6, plus combined_11 when the five SVOX weather "
        "splits are in the job (not the mixed SVOX queries folder). Ignored if --skip_combined is set.",
    )
    p.add_argument("--ece_metrics", type=str, default="recall",
        help="ECE metrics to compute (comma-separated). Default: recall only. Options: recall, map, ap.")
    p.add_argument("--ece_zoom_threshold", type=float, default=0.001,
        help="Fraction of data required in the last bin for ECE adaptive zoom (0.001 = 0.1%%).")
    p.add_argument("--ece_bin_mode", type=str, default="percentile",
        choices=["zoom", "percentile", "all"],
        help="Primary ECE binning. Default 'percentile' (KappaPlace): clip to a percentile "
        "window then equal-width bins (p99 = value 99%% of queries are at or below). "
        "κ-family uses two-sided [p1, p99]; baselines use one-sided [min, p99]. Outliers "
        "stay, folded into the edge bins. 'zoom' is legacy equal-width on raw min–max, "
        "then shrink the upper edge until the last bin is not almost empty. "
        "'all' is percentile plus --ece_extras.")
    p.add_argument("--ece_vmf_kappa_floor", action="store_true", help="Enable vMF kappa flooring (kappa<1 -> 1) before ECE inversion.")

    # Visualization
    p.add_argument("--num_preds_to_save", type=int, default=3, help="Number of predictions to save per query")
    p.add_argument("--num_queries_to_save", type=int, default=3, help="Number of queries to save predictions for")
    p.add_argument("--save_only_wrong_preds", action="store_true", help="Only save wrongly predicted queries")
    p.add_argument("--save_plots", action="store_true",
        help="Save evaluation plots (variance distribution, ECE, predictions) for val/test splits")

    return p.parse_args(), p
    

def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        logger.error(f"Config not found: {path}") # Log error before raising
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _explicit_cli_dests(parser: argparse.ArgumentParser) -> set[str]:
    """Return argparse destinations that were explicitly provided on the CLI."""
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
        if action.dest != "help"
    }

    explicit = set()
    for token in sys.argv[1:]:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def merge_cfg_with_cli(cfg: Dict[str, Any], args: argparse.Namespace, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """Merge priority: argparse defaults → JSON config → explicit CLI flags.

    For every argument the user *explicitly* passed on the command line, the CLI
    value wins over the JSON config.  Otherwise the JSON value wins over the
    argparse default."""
    cli = vars(args).copy()
    cli.pop("config", None)
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    explicit_cli = _explicit_cli_dests(parser)

    # 1. Start with argparse defaults for every known arg.
    merged = {k: defaults.get(k) for k in cli}
    # 2. JSON config overrides defaults.
    for k, v in cfg.items():
        merged[k] = v
    # 3. Explicit CLI flags override everything.
    for k, v in cli.items():
        if k in explicit_cli:
            merged[k] = v

    return merged


def normalize(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common fields to expected types/format."""
    out = dict(merged)

    # CosPlace paper ViT defaults when still on CNN defaults (512 / 512).
    backbone = out.get("backbone")
    if backbone == "ViT":
        if out.get("image_size") in (None, 512):
            out["image_size"] = 224
            logger.info(
                "backbone=%s: setting image_size=%d (CosPlace paper transformer default)",
                backbone,
                224,
            )
        if out.get("descriptors_dimension") in (None, 512):
            out["descriptors_dimension"] = 768
            logger.info(
                "backbone=%s: setting descriptors_dimension=%d "
                "(CosPlace paper transformer default)",
                backbone,
                768,
            )
        # Fixed pos-embed transformers need square resize at val/test.
        if not out.get("resize_test_imgs"):
            out["resize_test_imgs"] = True
            logger.info(
                "backbone=%s: enabling resize_test_imgs (required for transformer pos embeds)",
                backbone,
            )

    if backbone in DINOV2_BACKBONES:
        if out.get("image_size") in (None, 512):
            out["image_size"] = DINOV2_IMAGE_SIZE
            logger.info(
                "backbone=%s: setting image_size=%d (multiple of patch size %d)",
                backbone,
                DINOV2_IMAGE_SIZE,
                DINOV2_PATCH,
            )
        elif int(out["image_size"]) % DINOV2_PATCH != 0:
            raise ValueError(
                f"backbone={backbone} needs image_size divisible by {DINOV2_PATCH}, "
                f"got {out['image_size']}. Use {DINOV2_IMAGE_SIZE}, 322, 448, or 518."
            )
        embed = DINOV2_BACKBONES[backbone][1]
        if out.get("descriptors_dimension") in (None, 512):
            out["descriptors_dimension"] = embed
            logger.info(
                "backbone=%s: setting descriptors_dimension=%d (DINOv2 embed dim)",
                backbone,
                embed,
            )
        if not out.get("resize_test_imgs"):
            out["resize_test_imgs"] = True
            logger.info(
                "backbone=%s: enabling resize_test_imgs (required for DINOv2)",
                backbone,
            )

    if str(out.get("method", "")).lower() == "mixvpr":
        if out.get("image_size") in (None, 512):
            out["image_size"] = 320
            logger.info("method=mixvpr: setting image_size=320")
        elif int(out["image_size"]) != 320:
            raise ValueError(
                f"method=mixvpr needs image_size 320 (aggregator expects a 20×20 map), "
                f"got {out['image_size']}."
            )
        if not out.get("resize_test_imgs"):
            out["resize_test_imgs"] = True
            logger.info("method=mixvpr: enabling resize_test_imgs (MixVPR input is 320×320)")

    if str(out.get("method", "")).lower() == "salad":
        # Resume-checkpoint SALAD: 8448-d dino_salad or 544-d dino_salad_512_32, at 322×322.
        if out.get("image_size") in (None, 512):
            out["image_size"] = 322
            logger.info("method=salad: setting image_size=322")
        elif int(out["image_size"]) % 14 != 0:
            raise ValueError(
                f"method=salad needs image_size divisible by 14, got {out['image_size']}. "
                "Released dino_salad is evaluated at 322."
            )
        if out.get("descriptors_dimension") in (None, 512):
            out["descriptors_dimension"] = 8448
            logger.info(
                "method=salad: setting descriptors_dimension=8448 "
                "(pass 544 for dino_salad_512_32)"
            )
        elif int(out["descriptors_dimension"]) not in (8448, 544):
            raise ValueError(
                "method=salad descriptors_dimension must be 8448 (official) or "
                f"544 (dino_salad_512_32), got {out['descriptors_dimension']}."
            )
        if not out.get("resize_test_imgs"):
            out["resize_test_imgs"] = True
            logger.info("method=salad: enabling resize_test_imgs (SALAD input is 322×322)")

    # Paths: store as strings in config, but normalize to expanded string paths
    for k in ("data_folder", "logs_folder", "resume_train", "resume_model"):
        if k in out and out[k] is not None:
            out[k] = str(Path(out[k]).expanduser())

    for element in ("datasets", "datasets_type", "losses"):
        if element in out and out[element] is not None:
            v = out[element]
            if isinstance(v, list):
                continue
            v = str(v).strip()
            if v.lower() == "all":
                out[element] = "all"
            else:
                out[element] = [s.strip() for s in v.split(",") if s.strip()]

    if "ece_metrics" in out and out["ece_metrics"] is not None:
        v = out["ece_metrics"]
        if isinstance(v, list):
            out["ece_metrics"] = [str(x).strip().lower() for x in v if str(x).strip()]
        else:
            out["ece_metrics"] = [s.strip().lower() for s in str(v).split(",") if s.strip()]

    if out.get("eval_metrics") is not None:
        out["eval_metrics"] = parse_eval_metrics_arg(out["eval_metrics"])

    if out.get("rejection_delta_pcts") is not None:
        v = out["rejection_delta_pcts"]
        if isinstance(v, str):
            parts = [s.strip() for s in v.replace(",", " ").split() if s.strip()]
        else:
            parts = list(v)
        out["rejection_delta_pcts"] = normalize_rejection_delta_pcts(parts)

    if out.get("eval_query_folders") is not None:
        v = out["eval_query_folders"]
        if isinstance(v, list):
            names = [str(x).strip() for x in v if str(x).strip()]
        else:
            v = str(v).strip()
            if not v or v.lower() == "all":
                names = []
            else:
                names = [s.strip() for s in v.split(",") if s.strip()]
        out["eval_query_folders"] = names or None

    raw_es = out.get("early_stop_metric", "recall")
    out["early_stop_metrics"] = canonical_early_stop_metrics(raw_es)
    for m in out["early_stop_metrics"]:
        if m.startswith("ece_recall_"):
            if out.get("model_mode") != "uncertainty":
                raise ValueError(f"early_stop_metric {m} requires model_mode=uncertainty.")
            if not out.get("use_labels"):
                raise ValueError(f"early_stop_metric {m} requires use_labels.")

    rv = list(out.get("recall_values") or [1, 5, 10, 20])
    need_k = recall_values_needed_for_metrics(out["early_stop_metrics"])
    merged_rv = sorted(set(rv) | set(need_k))
    if merged_rv != rv:
        out["recall_values"] = merged_rv

    # Validate phased early stop
    if out.get("phased_early_stop"):
        es = out["early_stop_metrics"]
        has_ece = any(m.startswith("ece_recall_") for m in es)
        if not has_ece:
            raise ValueError(
                "phased_early_stop requires early_stop_metric to include at least one "
                "ece_recall_* metric (e.g. ece_recall,ece_recall_05,ece_recall_10)."
            )
        # Auto-add recall as Phase 1 metric if not already present.
        if "recall" not in es:
            out["early_stop_metrics"] = ["recall"] + es
            logger.info("phased_early_stop: auto-added 'recall' to early_stop_metrics for Phase 1.")

    out["var_head_agg"] = bool(out.get("var_head_agg", False))
    out["var_head_linear"] = str(out.get("var_head_linear", "d")).lower()
    out["ece_two_sided"] = bool(out.get("ece_two_sided", True))
    out["ece_extras"] = bool(out.get("ece_extras", False)) or str(
        out.get("ece_bin_mode") or ""
    ).lower() == "all"
    out["skip_combined"] = bool(out.get("skip_combined", False))
    out["combined_extras"] = bool(out.get("combined_extras", False))

    # Device auto-detection
    requested_device = str(out.get("device", "auto")).lower()
    out["device"] = "cuda" if requested_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"

    return out


def _resolve_log_dir(logs_folder: Optional[str],
                     resume_checkpoint: Optional[str] = None,
                     resume_model: Optional[str] = None) -> Optional[Path]:
    """Determine where log files should go based on resume mode and script name."""
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            timestamp = datetime.now().strftime("resume_train_%Y-%m-%d_%H-%M-%S")
            return resume_path.parent / timestamp

    elif resume_model:
        resume_path = Path(resume_model)
        if resume_path.exists():
            if "train" in Path(sys.argv[0]).name:
                timestamp = datetime.now().strftime("resume_model_%Y-%m-%d_%H-%M-%S")
                return resume_path.parent / timestamp
            else:  # eval: fixed folder per checkpoint
                eval_dir = resume_path.parent / "eval" / resume_path.stem
                return eval_dir

    if logs_folder:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(logs_folder) / timestamp

    return None


def setup_logging(logs_folder: Optional[str], dry_run: bool = False,
                  resume_checkpoint: Optional[str] = None, resume_model: Optional[str] = None,
                  use_wandb: bool = False):
    """Configure unified logging: console (INFO), info.log, debug.log.
    When use_wandb is True, create log_dir even in dry_run so W&B and eval paths (eval/dataset_name) work."""
    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S"))
    handlers.append(console_handler)

    log_dir = _resolve_log_dir(logs_folder, resume_checkpoint, resume_model)

    if log_dir and (not dry_run or use_wandb):
        log_dir.mkdir(parents=True, exist_ok=True)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        for filename, level in [("debug.log", logging.DEBUG), ("info.log", logging.INFO)]:
            fh = logging.FileHandler(log_dir / filename, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(file_formatter)
            handlers.append(fh)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)

    def exception_handler(type_, value, tb):
        logging.error("Uncaught exception occurred:", exc_info=(type_, value, tb))
    sys.excepthook = exception_handler

    return log_dir


def select_entries(entries: List[Dict[str, Any]], datasets: Any) -> List[Dict[str, Any]]:
    """
    Filter entries by 'datasets' selection.
    datasets can be:
      - "all" or None => no filtering
      - list of names => filter by entry["name"]
    """
    if datasets is None or (isinstance(datasets, str) and datasets.lower() == "all"):
        return entries

    if isinstance(datasets, list):
        wanted = {d.lower() for d in datasets}
        return [e for e in entries if str(e.get("name", "")).lower() in wanted]

    # fallback: no filtering
    return entries


def _log_config_summary(cfg: Dict[str, Any], entries: List[Dict[str, Any]], cfg_path: Path):
    """Log a summary of the resolved configuration."""
    logger.debug(f"Config file: {cfg_path}")
    logger.debug(f"data_folder: {cfg['data_folder']}")
    logger.debug(f"dry_run: {cfg['dry_run']}")
    logger.debug(f"entries to process: {[e.get('name') for e in entries]}")
    logger.debug(f"Using device: {cfg['device']}")
    logger.debug(f"method: {cfg.get('method')}, backbone: {cfg.get('backbone')}, "
                f"descriptors_dimension: {cfg.get('descriptors_dimension')}")
    if cfg.get('image_size') is not None:
        logger.debug(f"image_size: {cfg['image_size']}")
    if cfg.get("resume_train") is not None:
        logger.info(f"resume_train: {cfg['resume_train']}")


def build_config():
    """Parse CLI + JSON config, set up logging, validate, and return merged config + entries."""
    args, parser = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    merged = merge_cfg_with_cli(cfg, args, parser)
    merged = normalize(merged)

    log_dir = setup_logging(merged.get("logs_folder"),
                            dry_run=merged.get("dry_run", False),
                            resume_checkpoint=merged.get("resume_train"),
                            resume_model=merged.get("resume_model"),
                            use_wandb=merged.get("use_wandb", False))
    merged['log_dir'] = str(log_dir) if log_dir else None

    if "data_folder" not in merged:
        logger.critical("Missing required field: 'data_folder'")
        raise ValueError("Missing required field: 'data_folder'")

    entries = merged.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        logger.error("Config must include non-empty list field: 'entries'")
        raise ValueError("Config must include non-empty list field: 'entries'")

    entries = select_entries(entries, merged.get("datasets", None))

    if merged.get("save_config") and log_dir and not merged.get("dry_run"):
        outp = (log_dir / "merged_config.json").expanduser()
        outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info(f"Saved merged config to {outp}")

    _log_config_summary(merged, entries, cfg_path)
    return merged, entries


if __name__ == "__main__":
    build_config()