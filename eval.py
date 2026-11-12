"""Visual place recognition evaluation entry point."""
import logging
from typing import Any, Dict, List, Optional

from data.dataset_paths import list_query_folders, query_folder_display_name, resolve_eval_path
from eval_metrics.eval_tasks import resolve_eval_metrics
from eval_metrics.combined_eval import run_combined_eval_variants
from eval_metrics.dataset_eval import eval_dataset
from utils import commons, wandb_utils
from utils.runtime import build_config_and_datasets, init_model

logger = logging.getLogger(__name__)

__all__ = ["eval_dataset", "run_eval", "main"]


def run_eval(cfg, entries, datasets_paths, device, model) -> None:
    """Evaluate all configured dataset entries and query-folder splits."""
    cfg["eval_metrics"] = resolve_eval_metrics(cfg, training=False)
    logger.info("Eval metrics: %s", ", ".join(cfg["eval_metrics"]))
    all_panel_data: List[Dict[str, Any]] = []
    all_wandb_images: List[Optional[Dict[str, Any]]] = []
    query_folder_filter = cfg.get("eval_query_folders")
    if query_folder_filter:
        logger.info("Eval query folders filter: %s", ", ".join(query_folder_filter))

    for entry in entries:
        name = entry["name"]
        eval_ds_path = resolve_eval_path(datasets_paths, name)
        query_folders = list_query_folders(eval_ds_path, query_folder_filter)
        if not query_folders:
            if query_folder_filter:
                logger.warning(
                    "[%s] No query folders matching %s under %s",
                    name,
                    query_folder_filter,
                    eval_ds_path,
                )
            else:
                logger.warning("[%s] No query folders found in %s", name, eval_ds_path)
            continue

        shared_db_desc = shared_db_var = None
        for q_folder in query_folders:
            q_folder_name = q_folder.name
            display_name = query_folder_display_name(name, q_folder_name, query_folders)
            logger.info(
                "\n%s\nEvaluating dataset \033[1m%s\033[0m - queries folder: %s",
                "=" * 30,
                name,
                q_folder_name,
            )

            results = eval_dataset(
                cfg,
                model,
                device,
                display_name,
                eval_ds_path,
                queries_folder_name=q_folder_name,
                base_dataset_name=name,
                cached_db_desc=shared_db_desc,
                cached_db_var=shared_db_var,
            )

            if results.panel_data is not None:
                all_panel_data.append(results.panel_data)
                wandb_utils.log_eval_dataset_results(
                    cfg, results.panel_data, results.wandb_images
                )
            all_wandb_images.append(results.wandb_images)
            shared_db_desc, shared_db_var = results.db_desc, results.db_var

    combined_outputs = run_combined_eval_variants(cfg, all_panel_data)
    wandb_utils.log_eval_results(
        cfg,
        all_panel_data,
        all_wandb_images,
        combined_outputs,
        include_dataset_media=False,
    )
    logger.info("%s\nAll processes finished.", "=" * 30)


def main() -> None:
    cfg, entries, datasets_paths = build_config_and_datasets()
    commons.make_deterministic(cfg["seed"])
    wandb_utils.init_wandb(cfg, job_type="eval")
    device, model = init_model(cfg)
    commons.copy_resume_model_to_log_dir(cfg, logger)
    run_eval(cfg, entries, datasets_paths, device, model)
    wandb_utils.finish_run(cfg)


if __name__ == "__main__":
    main()
