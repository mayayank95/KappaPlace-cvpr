import logging
from typing import Any, Dict

import torch

from models.model_mode import GeneralModelWrapper, build_model_mode, _stable_var_init
from models.salad import SaladPrenorm

logger = logging.getLogger(__name__)


def _load_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    state_dict_key: str = "model_state_dict",
    skip_var_head: bool = False,
):
    """Load model weights from a checkpoint file (strict=False to allow mismatches).
    If skip_var_head is True, do not load var_head keys (keep build-time init, e.g. var_init)."""
    logger.info(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        isinstance(checkpoint, dict)
        and state_dict_key in checkpoint
        and isinstance(checkpoint[state_dict_key], dict)
    ):
        state_dict = checkpoint[state_dict_key]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict) and any(k.startswith("core.") for k in state_dict):
        state_dict = {
            k[len("core.") :]: v for k, v in state_dict.items() if k.startswith("core.")
        }
    if skip_var_head:
        n_before = len(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if "var_head" not in k}
        n_skipped = n_before - len(state_dict)
        if n_skipped:
            logger.info(
                "Skipping %d var_head key(s) from checkpoint (var_init); variance head keeps build-time init.",
                n_skipped,
            )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_var = [k for k in missing if "var_head" in k]
    if missing_var:
        logger.warning("Checkpoint did not fill var_head keys: %s", missing_var)
    elif missing:
        logger.debug("State dict missing keys (left at default init): %s", missing)
    if unexpected:
        logger.debug("State dict unexpected keys (ignored): %s", unexpected)
    return set(state_dict)


def _seed_salad_prenorm_head(model: torch.nn.Module, loaded_keys) -> None:
    """Start the uncertainty SALAD from the retrieval aggregator.

    The copy is made at build time, before ``--resume_model`` overwrites the
    retrieval weights. If this checkpoint has no trained prenorm head, copy
    those loaded retrieval weights across. A resumed prenorm head is left as
    saved.
    """
    root = getattr(model, "module", model)
    if getattr(root, "_encoder_kind", None) != "SaladEncoder":
        return
    if not getattr(root, "_var_from_feature_map", False):
        return
    seq = getattr(getattr(root, "var_head", None), "seq", None)
    if seq is None or len(seq) == 0 or not isinstance(seq[0], SaladPrenorm):
        return
    prefix = "var_head.seq.0.aggregator."
    if loaded_keys and any(str(key).startswith(prefix) for key in loaded_keys):
        return
    seq[0].aggregator.load_state_dict(root.aggregator.state_dict())
    logger.info(
        "SALAD uncertainty prenorm head initialized from the retrieval aggregator."
    )


def get_model(args: Dict[str, Any]) -> torch.nn.Module:
    """Build model and optionally load pretrained weights."""
    method = args["method"]

    if method in ("cosplace", "mixvpr", "salad"):
        model = build_model_mode(args)
    elif method == "cosplace_pretrained":
        logger.info(
            f"Loading pretrained model from torch.hub: backbone={args['backbone']}, "
            f"dim={args['descriptors_dimension']}"
        )
        hub_model = torch.hub.load("gmberton/cosplace", "get_trained_model", args["backbone"], args["descriptors_dimension"])
        if args.get("model_mode") == "uncertainty":
            # Build full Uncertainty model (with var_head), then load hub weights into backbone/aggregation
            model = build_model_mode(args)
            hub_sd = hub_model.state_dict()
            missing, unexpected = model.load_state_dict(hub_sd, strict=False)
            logger.info("Loaded pretrained hub weights into Uncertainty model (strict=False).")
            if missing:
                logger.debug("Keys left at init (expected for var_head/final_l2): %s", missing)
            if unexpected:
                logger.debug("Unexpected hub keys (ignored): %s", unexpected)
        else:
            model = GeneralModelWrapper(hub_model)
    else:
        raise ValueError(f"Unknown method: {method}")

    resume_path = args.get("resume_model")
    loaded_keys = set()
    if resume_path is not None:
        # When var_init: don't load var_head from checkpoint, then force re-init
        skip_var_head = bool(args.get("var_init"))
        state_dict_key = args.get("ckpt_state_dict_key", "model_state_dict")
        loaded_keys = _load_weights(
            model, resume_path, state_dict_key=state_dict_key, skip_var_head=skip_var_head
        )
        if skip_var_head:
            root = getattr(model, "module", model)
            if hasattr(root, "var_head"):
                act = args.get("variance_activation", "softplus") or "softplus"
                _stable_var_init(root.var_head, act)
                logger.info("Variance head re-initialized (var_init) for stable start ~0.1.")
    elif method == "salad":
        logger.info(
            "No --resume_model provided. DINOv2 backbone uses torch.hub weights; "
            "SALAD aggregator is randomly initialized."
        )
    elif method != "cosplace_pretrained":
        if str(args.get("backbone", "")).startswith("DINOv2"):
            logger.info("No --resume_model provided. Model uses DINOv2 pretrained weights.")
        else:
            logger.info("No --resume_model provided. Model uses default weights (ImageNet).")

    if str(args.get("method", "")).lower() == "salad" and args.get("var_head_agg"):
        _seed_salad_prenorm_head(model, loaded_keys)

    if args.get("freeze_model"):
        trainable = model.freeze_base()
        if not trainable:
            raise ValueError("--freeze_model: no trainable parameters remain. Nothing to train.")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.debug(f"Base model frozen, training {len(trainable)} params: {trainable_names}")

    return model
