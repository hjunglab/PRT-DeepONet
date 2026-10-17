#!/usr/bin/env bash
# Velocity pipeline.   bash velocity/make_all.sh [figures|extract|train|all]   (default: figures)
#   figures : 4_figures/ from the shipped data/ (no LB labels, no GPU)
#   extract : 1_preprocess -> 3_extract -> 4_figures with the released weights (needs PRT_LB)
#   train   : 1_preprocess -> 2_train (pressure U-Net; velocity variant, seeds 42 2 3 4 5, lambda sweep)
#   all     : train, then extract with the newly trained weights (WEIGHTS=$PRT_WORK/velocity/weights)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-figures}
run() { echo "+ python $*"; python "$@"; }
preprocess() { run velocity/1_preprocess/geometry_features.py; }
train() {
  export PRESSURE_UNET=${PRT_WORK:-work}/velocity/weights/pressure_unet.pt
  run velocity/1_preprocess/pressure_labels.py; run velocity/2_train/train_pressure_unet.py
  for s in 42 2 3 4 5; do for l in 0 1 3 5 10 30 100 300 500; do LAMBDA=$l SEED=$s run velocity/2_train/train_velocity.py; done; done
}
extract() {
  run velocity/3_extract/evaluate.py; SPLIT=val run velocity/3_extract/evaluate.py; run velocity/3_extract/predicted_velocity.py; run velocity/3_extract/interpolation_unseen_re.py
  run velocity/3_extract/physics_diagnostics.py; run velocity/3_extract/extract_figure_data.py; run velocity/3_extract/timing.py
}
figures() { for f in velocity/4_figures/*.py; do run "$f"; done; }
case $STAGE in
  figures) figures ;;
  extract) preprocess; extract; figures ;;
  train)   preprocess; train ;;
  all)     preprocess; train; export WEIGHTS=${PRT_WORK:-work}/velocity/weights; extract; figures ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
