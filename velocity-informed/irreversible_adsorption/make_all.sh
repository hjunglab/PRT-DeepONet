#!/usr/bin/env bash
# Adsorption pipeline.   bash irreversible_adsorption/make_all.sh [figures|extract|train|all]   (default: figures)
#   figures : 4_figures/ from the shipped data/ (no LB labels, no GPU)
#   extract : 1_preprocess -> 3_extract -> 4_figures with the released weights (needs PRT_LB and the velocity
#             preprocessing + predicted velocity: bash velocity/make_all.sh extract)
#   train   : 1_preprocess -> 2_train (concentration variant, seeds 42 2 3 4 5)
#   all     : train, then extract with the newly trained weights (WEIGHTS=$PRT_WORK/irreversible_adsorption/weights)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-figures}
run() { echo "+ python $*"; python "$@"; }
preprocess() { run irreversible_adsorption/1_preprocess/geometry_features.py; run irreversible_adsorption/1_preprocess/velocity_input.py; }
train() { for s in 42 2 3 4 5; do SEED=$s run irreversible_adsorption/2_train/train_adsorption.py; done; }
extract() { run irreversible_adsorption/3_extract/evaluate.py; SPLIT=val run irreversible_adsorption/3_extract/evaluate.py; run irreversible_adsorption/3_extract/extract_figure_data.py; run irreversible_adsorption/3_extract/timing.py; }
figures() { for f in irreversible_adsorption/4_figures/[ft]*.py; do run "$f"; done; }
case $STAGE in
  figures) figures ;;
  extract) preprocess; extract; figures ;;
  train)   preprocess; train ;;
  all)     preprocess; train; export WEIGHTS=${PRT_WORK:-work}/irreversible_adsorption/weights; extract; figures ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
