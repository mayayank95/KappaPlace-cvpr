#!/usr/bin/env bash
# Evaluate a CosPlace KappaPlace checkpoint.
# BACKBONE must match training: ResNet50, ViT, or DINOv2-B.
#
#   bash scripts/cosplace_eval.sh path/to/best_model.pth
#   BACKBONE=ViT bash scripts/cosplace_eval.sh path/to/best_model.pth
#   BACKBONE=DINOv2-B DATASETS=pitts30k bash scripts/cosplace_eval.sh path/to/best_model.pth
#
# Every benchmark uses --use_labels (the default), including amstertime,
# nordland, and spedtest. Positives come from UTM coordinates in the path.

set -euo pipefail
cd "$(dirname "$0")/.."

RESUME_MODEL="${RESUME_MODEL:-${1:-}}"
if [[ -z "${RESUME_MODEL}" ]]; then
  echo "Usage: bash scripts/cosplace_eval.sh path/to/best_model.pth" >&2
  exit 1
fi

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

DATASETS="${DATASETS:-pitts30k}"
ARGS=(
  --config configs/datasets.json
  --resume_model "${RESUME_MODEL}"
  --method cosplace
  --backbone "${BACKBONE}"
  --descriptors_dimension "${DIM}"
  --image_size "${SIZE}"
  --model_mode uncertainty
  --uncertainty_loss vmf
  --variance_activation softplus
  --var_head_agg
  --var_head_linear 1
  --datasets "${DATASETS}"
  --datasets_type test
)

if [[ "${USE_LABELS:-1}" == "1" ]]; then
  ARGS+=(--use_labels)
fi

python eval.py "${ARGS[@]}"
