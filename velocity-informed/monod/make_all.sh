#!/usr/bin/env bash
# Monod pipeline.   bash monod/make_all.sh [figures|extract|train|all]   (default: figures)
#   figures : 4_figures/ from the shipped data/ (no LB labels, no GPU)
#   extract : 1_preprocess -> 3_extract -> 4_figures with the released weights (needs PRT_LB and the velocity
#             preprocessing + predicted velocity: bash velocity/make_all.sh extract)
#   train   : 1_preprocess -> 2_train (concentration variant, seeds 42 2 3 4 5)
#   all     : train, then extract with the newly trained weights (WEIGHTS=$PRT_WORK/monod/weights)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-figures}
run() { echo "+ python $*"; python "$@"; }
preprocess() { run monod/1_preprocess/geometry_features.py; run monod/1_preprocess/velocity_input.py; run monod/1_preprocess/targets.py; }
train() { for s in 42 2 3 4 5; do SEED=$s run monod/2_train/train_monod.py; done; }
extract() { run monod/3_extract/evaluate.py; run monod/3_extract/extract_figure_data.py; run monod/3_extract/timing.py; }
figures() { for f in monod/4_figures/[ft]*.py; do run "$f"; done; }
case $STAGE in
  figures) figures ;;
  extract) preprocess; extract; figures ;;
  train)   preprocess; train ;;
  all)     preprocess; train; export WEIGHTS=${PRT_WORK:-work}/monod/weights; extract; figures ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
