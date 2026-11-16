#!/usr/bin/env bash
# Train KappaPlace on one CosPlace backbone.
# Joint training uses two-phase early stopping: validation recall, then
# ECE@1, @5, and @10. The recall checkpoint is best_model_recall.pth.
#
#   bash scripts/cosplace_train.sh
#   BACKBONE=ViT bash scripts/cosplace_train.sh
#   BACKBONE=DINOv2-B bash scripts/cosplace_train.sh
#
# Post-training (frozen backbone, uncertainty head only).
# RESUME_MODEL must be a checkpoint saved by this trainer: the vMF targets are
# its CosFace prototypes. Early stopping uses validation ECE@1, and the kept
# weights are best_model_ece_recall_01.pth.
#   FREEZE=1 RESUME_MODEL=path/to/retrieval.pth bash scripts/cosplace_train.sh
#
# One iteration: DRY_RUN=1 bash scripts/cosplace_train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="${BACKBONE:-ResNet50}"
case "${BACKBONE}" in
  ResNet50)
    DIM=512
    SIZE=512
    ;;
  ViT)
    DIM=768
    SIZE=224
    ;;
  DINOv2-B)
    DIM=768
    SIZE=224
    ;;
  *)
    echo "BACKBONE must be ResNet50, ViT, or DINOv2-B, got ${BACKBONE}" >&2
    exit 1
    ;;
esac

ARGS=(
  --config configs/datasets.json
  --method cosplace
  --backbone "${BACKBONE}"
  --descriptors_dimension "${DIM}"
  --image_size "${SIZE}"
  --datasets sf_xl
  --datasets_type train,val
  --groups_num 8
  --model_mode uncertainty
  --uncertainty_loss vmf
  --variance_activation softplus
  --var_head_agg
  --var_head_linear 1
)

if [[ "${FREEZE:-0}" == "1" ]]; then
  if [[ -z "${RESUME_MODEL:-}" ]]; then
    echo "FREEZE=1 requires RESUME_MODEL=path/to/retrieval.pth" >&2
    exit 1
  fi
  # Recall does not move once the retrieval model is frozen, so stop on ECE@1.
  ARGS+=(
    --freeze_model
    --freeze_batchnorm
    --losses uncertainty
    --resume_model "${RESUME_MODEL}"
    --load_classifiers "${RESUME_MODEL}"
    --early_stop_metric ece_recall
    --use_labels
  )
else
  # Phase 1 locks recall. Phase 2 then early-stops on ECE@1, @5, and @10.
  ARGS+=(
    --losses ce,uncertainty
    --phased_early_stop
    --early_stop_metric ece_recall,ece_recall_05,ece_recall_10
    --use_labels
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry_run --num_workers 0)
fi

python train.py "${ARGS[@]}"
