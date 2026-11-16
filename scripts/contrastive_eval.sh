#!/usr/bin/env bash
# Evaluate a checkpoint trained in the Contrastive repository.
#
#   bash scripts/contrastive_eval.sh mixvpr path/to/mixvpr_kappa.pth
#   bash scripts/contrastive_eval.sh salad path/to/salad_kappa.pth
#
# SALAD uses the linear descriptor head: Linear(544, 1) and Softplus on the
# pre-L2 token and cluster sum.
#
# DATASETS defaults to every entry in the config. use_labels is on for all of
# them, including amstertime, nordland, and spedtest. Positives come from UTM
# coordinates in the image path.

set -euo pipefail
cd "$(dirname "$0")/.."

METHOD="${1:-}"
RESUME_MODEL="${RESUME_MODEL:-${2:-}}"
if [[ -z "${METHOD}" || -z "${RESUME_MODEL}" ]]; then
  echo "Usage: bash scripts/contrastive_eval.sh {mixvpr|salad} path/to/checkpoint.pth" >&2
  exit 1
fi

case "${METHOD}" in
  mixvpr) CONFIG=configs/datasets_mixvpr.json ;;
  salad) CONFIG=configs/datasets_salad.json ;;
  *)
    echo "METHOD must be mixvpr or salad, got ${METHOD}" >&2
    exit 1
    ;;
esac

ARGS=(
  --config "${CONFIG}"
  --resume_model "${RESUME_MODEL}"
)
if [[ -n "${DATASETS:-}" ]]; then
  ARGS+=(--datasets "${DATASETS}")
fi

python eval.py "${ARGS[@]}"
