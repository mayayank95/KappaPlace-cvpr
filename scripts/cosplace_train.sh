#!/usr/bin/env bash
# Train KappaPlace on one CosPlace backbone.
#
#   bash scripts/cosplace_train.sh
#   BACKBONE=ViT bash scripts/cosplace_train.sh
#   BACKBONE=DINOv2-B bash scripts/cosplace_train.sh
#
# Post-training (frozen backbone, uncertainty head only):
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
  ARGS+=(--freeze_model --freeze_batchnorm --losses uncertainty)
  if [[ -n "${RESUME_MODEL:-}" ]]; then
    ARGS+=(--resume_model "${RESUME_MODEL}" --load_classifiers "${RESUME_MODEL}")
  fi
else
  ARGS+=(--losses ce,uncertainty)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry_run --num_workers 0)
fi

python train.py "${ARGS[@]}"
