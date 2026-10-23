#!/usr/bin/env bash
# Train and evaluate one DINOv2 configuration for one seed.
# Usage: RAW_ROOT=/path/to/utah2018 bash scripts/run_experiment.sh <e0..e7> <seed>
set -euo pipefail

EXPERIMENT="${1:?experiment id, e.g. e2}"
SEED="${2:?seed, e.g. 42}"
RAW_ROOT="${RAW_ROOT:?set RAW_ROOT to the raw Utah 2018 NRRD directory}"
DATA_ROOT="${DATA_ROOT:-data/utah2018_run2}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${EXPERIMENT}/seed${SEED}}"
LOSS_IMPL="${LOSS_IMPL:-fp32}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "$DATA_ROOT/dataset_provenance.json" ]]; then
  python -u src/prepare_utah_run2.py --raw_root "$RAW_ROOT" --splits_root splits \
    --output_root "$DATA_ROOT"
fi

python -u src/train_e5_e6.py --experiment "$EXPERIMENT" --seed "$SEED" \
  --data_root "$DATA_ROOT" --raw_root "$RAW_ROOT" --output_dir "$OUTPUT_DIR" \
  --batch_size 4 --num_workers 4 --epochs 50 --patience 10 \
  --loss_impl "$LOSS_IMPL" --resume
