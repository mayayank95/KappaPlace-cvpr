import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_dataset_paths(cfg, entries) -> Dict[str, Dict[str, Path]]:
    """Validate dataset entries and resolve their train/validation/test paths."""
    data_root = Path(cfg["data_folder"]).expanduser()
    dataset_paths: Dict[str, Dict[str, Path]] = {}

    for e in entries:
        name = e.get("name")
        if not name:
            logger.error("Validation failed: Each entry must have a non-empty 'name'")
            raise ValueError("Each entry must have a non-empty 'name'")

        path = e.get("path")
        if not path:
            logger.error(f"Validation failed: Entry '{name}' missing required field: path")
            raise ValueError(f"Entry '{name}' missing required field: path")

        src = (data_root / path).resolve()
        if not src.exists():
            logger.error(f"[{name}] Source not found: {src}")
            raise FileNotFoundError(f"[{name}] Source not found: {src}")

        dataset_paths[name] = {
            "train": (src / "train").resolve(strict=False),
            "validation": (src / "val").resolve(strict=False),
            "test": (src / "test").resolve(strict=False),
        }

    return dataset_paths


def resolve_eval_path(datasets_paths: Dict, name: str) -> Path:
    """Prefer the test split; fall back to val when test is missing."""
    eval_ds_path = datasets_paths[name]["test"]
    if not eval_ds_path.exists():
        eval_ds_path = datasets_paths[name]["validation"]
        logger.info("[%s] 'test' folder not found, using 'val' instead.", name)
    return eval_ds_path


def list_query_folders(
    eval_ds_path: Path,
    only_names: Optional[List[str]] = None,
) -> List[Path]:
    """Discover query folders under a dataset split; optionally keep only ``only_names``."""
    query_folders = sorted([p for p in eval_ds_path.glob("queries*") if p.is_dir()])
    if not query_folders:
        query_folders = sorted([p for p in eval_ds_path.glob("query*") if p.is_dir()])
    if only_names:
        wanted = {name.strip() for name in only_names if name.strip()}
        query_folders = [p for p in query_folders if p.name in wanted]
    return query_folders


def query_folder_display_name(
    entry_name: str, q_folder_name: str, query_folders: List[Path]
) -> str:
    if len(query_folders) == 1 and q_folder_name in ("queries", "query"):
        return entry_name
    return f"{entry_name}_{q_folder_name}"
