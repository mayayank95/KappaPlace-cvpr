"""Per-dataset evaluation: extraction, retrieval, uncertainty metrics, W&B payloads."""
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_wandb import build_eval_outputs
from eval_metrics.extraction import extract_descriptors
from eval_metrics.scores import collect_eval_scores
from eval_metrics.scores.joint_kappa import per_example_mean
from eval_metrics.eval_tasks import resolve_eval_metrics
from eval_metrics.method_eval import evaluate_method, merge_method_eval
from eval_metrics.retrieval import log_retrieval_results, run_retrieval
from eval_metrics.recall import has_ground_truth_mask, keep_labeled_queries
from eval_metrics.rejection_curves import (
    plot_rejection_curve_sets,
)
from eval_metrics.uncertainty import compute_uncertainty_statistics
logger = logging.getLogger(__name__)


def _as_var_rows(var: np.ndarray) -> np.ndarray:
    """``(N, D)`` variance rows. A per-example vector is one column."""
    arr = np.asarray(var)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


class EvalDatasetResult(NamedTuple):
    recalls: np.ndarray
    recalls_str: str
    map_at_k: Optional[List[float]]
    panel_data: Dict[str, Any]
    wandb_images: Optional[Dict[str, Any]]
    db_desc: np.ndarray
    db_var: np.ndarray


def _compute_uncertainty_block(
    args: Dict[str, Any],
    test_ds: TestDataset,
    predictions,
    distances,
    positives_per_query,
    q_var,
    db_var,
    q_desc,
    db_desc,
    dataset_output_dir: Path,
    wandb_step: Optional[int],
    dataset_name: Optional[str] = None,
):
    """Compute scores once, then ECE@k and rejection delta per method."""
    uncertainty_stats = None
    method_panel: Dict[str, Any] = {}
    pair_score_matrices: Dict[str, np.ndarray] = {}
    rejection_query_curves: List[Dict[str, Any]] = []
    rejection_pair_curves: List[Dict[str, Any]] = []
    save_plots = args.get("save_plots", False) or args.get("use_wandb", False)

    stats_output_dir = None
    if save_plots:
        if dataset_output_dir.exists() and not dataset_output_dir.is_dir():
            logger.warning(
                "Output path %s exists as a file; skipping variance distribution plot.",
                dataset_output_dir,
            )
        else:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            stats_output_dir = dataset_output_dir

    uncertainty_stats = compute_uncertainty_statistics(
        np.vstack([_as_var_rows(db_var), _as_var_rows(q_var)]),
        stats_output_dir,
        num_database=len(db_var),
    )

    if args["use_labels"] and positives_per_query is not None:
        if args["dry_run"]:
            q_var = q_var[: max(1, args.get("num_queries_to_save", 3))]

        if save_plots:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

        skip_baselines = bool(args.get("skip_baselines")) or wandb_step is not None
        methods, pair_score_matrices = collect_eval_scores(
            predictions,
            distances,
            q_var,
            db_var,
            q_desc,
            db_desc,
            uncertainty_loss=args.get("uncertainty_loss", "gaussian_nll"),
            database_utms=getattr(test_ds, "database_utms", None),
            dataset_name=dataset_name,
            skip_baselines=skip_baselines,
            ece_two_sided=bool(args.get("ece_two_sided", True)),
            ece_vmf_kappa_floor=bool(args.get("ece_vmf_kappa_floor", False)),
        )
        eval_metrics = resolve_eval_metrics(args, training=wandb_step is not None)
        if args.get("dry_run"):
            eval_metrics = [m for m in eval_metrics if m != "rejection"]

        ece_metrics = args.get("ece_metrics") or ["recall"]
        eval_kwargs = dict(
            n_values=args["recall_values"],
            output_dir=dataset_output_dir if save_plots else None,
            ece_metrics=ece_metrics,
            zoom_threshold=args.get("ece_zoom_threshold", 0.001),
            bin_mode=args.get("ece_bin_mode", "percentile"),
            eval_metrics=eval_metrics,
            rejection_delta_pcts=args.get("rejection_delta_pcts"),
            ece_extras=bool(args.get("ece_extras", False)),
        )
        for method in methods:
            one = evaluate_method(
                method,
                predictions,
                positives_per_query,
                distances,
                **eval_kwargs,
            )
            merge_method_eval(method_panel, one)
            rej = one.get("rejection") or {}
            rejection_query_curves.extend(rej.get("query_curves") or [])
            rejection_pair_curves.extend(rej.get("pair_curves") or [])

    return (
        method_panel.get("ece") or {},
        method_panel.get("raw_scores") or {},
        uncertainty_stats,
        pair_score_matrices,
        method_panel.get("rejection_curves") or {},
        rejection_query_curves,
        rejection_pair_curves,
    )


def eval_dataset(
    args,
    model,
    device,
    dataset_name,
    eval_ds_path,
    queries_folder_name="queries",
    wandb_step=None,
    log_dataset_info=True,
    base_dataset_name=None,
    cached_db_desc=None,
    cached_db_var=None,
    use_descriptor_cache=True,
) -> EvalDatasetResult:
    """
    Evaluate the model on a single dataset.
    Extracts features once, then computes recalls and uncertainty metrics.
    """
    model = model.eval()
    dataset_output_dir = Path(args["log_dir"]) / "eval" / dataset_name
    if not args["dry_run"] or args.get("use_wandb"):
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = TestDataset(
        f"{eval_ds_path}/database",
        f"{eval_ds_path}/{queries_folder_name}",
        positive_dist_threshold=args["positive_dist_threshold"],
        image_size=args.get("image_size"),
        use_labels=args["use_labels"],
        resize_test_imgs=args.get("resize_test_imgs", False),
    )
    if log_dataset_info:
        logger.info(f"Testing on {test_ds}")

    db_desc, q_desc, db_var, q_var = extract_descriptors(
        args,
        model,
        device,
        test_ds,
        dataset_name,
        base_dataset_name=base_dataset_name,
        cached_db_desc=cached_db_desc,
        cached_db_var=cached_db_var,
        use_descriptor_cache=use_descriptor_cache,
    )

    predictions, distances, recalls, recalls_str, positives_per_query, map_at_k = run_retrieval(
        args,
        test_ds,
        db_desc,
        q_desc,
        dataset_output_dir,
        save_recalls=use_descriptor_cache,
    )

    # Keep full query rows for saved-pred images (test_ds indices). Metrics use the
    # possibly-dropped copies below.
    viz_predictions, viz_distances, viz_q_var, viz_q_desc = (
        predictions,
        distances,
        q_var,
        q_desc,
    )
    drop_empty_gt = bool(args.get("drop_empty_gt", False))
    if drop_empty_gt and args["use_labels"] and positives_per_query is not None:
        nq = int(np.asarray(predictions).shape[0])
        mask = has_ground_truth_mask(positives_per_query, nq)
        if mask.size and not bool(np.all(mask)):
            logger.info(
                "drop_empty_gt: dropping %d/%d queries with no ground-truth positives.",
                int((~mask).sum()),
                nq,
            )
            positives_per_query, predictions, distances, q_var, q_desc = keep_labeled_queries(
                positives_per_query,
                predictions,
                distances,
                q_var,
                q_desc,
                n_queries=nq,
            )

    uncertainty_stats = None
    uncertainty_ece_results: Dict[str, Any] = {}
    uncertainty_raw_scores: Dict[str, Any] = {}
    pair_score_matrices: Dict[str, np.ndarray] = {}
    rejection_stats: Dict[str, Any] = {}
    rejection_query_curves: List[Dict[str, Any]] = []
    rejection_pair_curves: List[Dict[str, Any]] = []
    save_plots = args.get("save_plots", False) or args.get("use_wandb", False)

    if args["use_labels"]:
        log_retrieval_results(dataset_name, recalls_str, recalls, map_at_k, args["recall_values"])

    if args["model_mode"] == "uncertainty":
        (
            uncertainty_ece_results,
            uncertainty_raw_scores,
            uncertainty_stats,
            pair_score_matrices,
            rejection_stats,
            rejection_query_curves,
            rejection_pair_curves,
        ) = _compute_uncertainty_block(
            args,
            test_ds,
            predictions,
            distances,
            positives_per_query,
            q_var,
            db_var,
            q_desc,
            db_desc,
            dataset_output_dir,
            wandb_step,
            dataset_name=base_dataset_name or dataset_name,
        )

    if (
        args["model_mode"] == "uncertainty"
        and args["use_labels"]
        and positives_per_query is not None
        and not args.get("skip_rejection_curves")
        and save_plots
        and not args["dry_run"]
        and (rejection_query_curves or rejection_pair_curves)
    ):
        saved_path, pw_path = plot_rejection_curve_sets(
            rejection_query_curves,
            rejection_pair_curves,
            dataset_output_dir / "rejection_curves.png",
            title=f"{dataset_name} rejection curves",
        )
        if saved_path is not None:
            logger.info("Rejection curves saved to %s", saved_path)
        if pw_path is not None:
            logger.info("Pairwise rejection curves saved to %s", pw_path)

    if save_plots and args.get("num_preds_to_save", 0) != 0:
        preds_to_save = viz_predictions[:, : args["num_preds_to_save"]]
        pred_distances = viz_distances[:, : args["num_preds_to_save"]]
        query_variances = per_example_mean(viz_q_var)
        db_var_plot = db_var if args["model_mode"] == "uncertainty" else None
        visualizations.save_preds(
            preds_to_save,
            test_ds,
            str(dataset_output_dir),
            args["save_only_wrong_preds"],
            args["use_labels"],
            args["num_queries_to_save"],
            distances=pred_distances,
            query_variances=query_variances,
            db_variances=db_var_plot,
            q_desc=viz_q_desc,
            db_desc=db_desc,
            skip_empty_gt=drop_empty_gt,
        )

    panel_data, wandb_images = build_eval_outputs(
        dataset_name,
        args,
        recalls,
        map_at_k,
        uncertainty_ece_results,
        uncertainty_raw_scores,
        None,
        uncertainty_stats,
        dataset_output_dir=dataset_output_dir,
        save_plots=save_plots,
        base_dataset_name=base_dataset_name or dataset_name,
        query_folder_name=queries_folder_name,
    )

    if rejection_stats:
        panel_data["rejection_curves"] = rejection_stats

    if (
        args["model_mode"] == "uncertainty"
        and args["use_labels"]
        and positives_per_query is not None
        and not args["dry_run"]
    ):
        panel_data["pool"] = {
            "predictions": predictions,
            "positives_per_query": positives_per_query,
            "q_var": q_var,
            "distances": distances,
        }
        if pair_score_matrices:
            panel_data["pool"]["pair_scores"] = pair_score_matrices

    return EvalDatasetResult(
        recalls=recalls,
        recalls_str=recalls_str,
        map_at_k=map_at_k,
        panel_data=panel_data,
        wandb_images=wandb_images,
        db_desc=db_desc,
        db_var=db_var,
    )
