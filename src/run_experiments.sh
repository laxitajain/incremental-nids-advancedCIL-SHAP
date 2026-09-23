#!/bin/bash
# run_experiments.sh
# Runs all approaches on IoT-NID to TON_IoT incremental learning scenario

set -e

RESULTS_DIR="../results"
EXP_PREFIX="src_iot-nidd_dst_ton-iot"
DATASETS="iot_nidd ton_iot"
ARGS="--results-path $RESULTS_DIR --datasets $DATASETS --fields PL IAT DIR WIN --num-pkts 10 --batch-size 64 --nepochs 10 --network Lopez17CNN --seed 1"

echo "=== Running Baseline Approaches ==="

echo "1. Scratch"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach scratch --save-models

echo "2. FT"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach jointft --save-models

echo "3. FT-Mem"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach jointft --num-exemplars 100 --save-models

echo "4. BiC"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach bic --num-exemplars 100 --save-models

echo "=== Running New Advanced CIL Approaches ==="

echo "5. EWC"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach ewc --ewc-lambda 5000 --save-models

echo "6. LwF"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach lwf --lwf-lambda 1.0 --save-models

echo "7. DER"
python3 main.py --exp-name ${EXP_PREFIX} $ARGS --approach der --num-exemplars 100 --der-alpha 0.5 --save-models

echo "=== Computing Metrics ==="
python3 compute_metrics.py --exp-name scratch --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name jointft --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name jointft-mem --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name bic-mem --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name ewc --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name lwf --results-path $RESULTS_DIR --yes
python3 compute_metrics.py --exp-name der-mem --results-path $RESULTS_DIR --yes

echo "=== Running SHAP-Guided Experiments ==="

echo "8. Compute SHAP Weights from Scratch model"
python3 xai/shap_weighting.py --model-path ${RESULTS_DIR}/${EXP_PREFIX}_scratch/models/ --data-path ../data/uniform_label/iot-nidd_dwn10p.parquet --output-path ${RESULTS_DIR}/shap_weights.npy --num-samples 500

echo "9. EWC with SHAP feature weighting"
python3 main.py --exp-name ${EXP_PREFIX}_shap_weighted $ARGS --approach ewc --ewc-lambda 5000 --shap-weights ${RESULTS_DIR}/shap_weights.npy --save-models
python3 compute_metrics.py --exp-name ewc_shap_weighted --results-path $RESULTS_DIR --yes

echo "10. DER with SHAP-guided exemplar selection"
python3 main.py --exp-name ${EXP_PREFIX}_shap_exemplars $ARGS --approach der --num-exemplars 100 --der-alpha 0.5 --exemplar-selection shap --save-models
python3 compute_metrics.py --exp-name der-mem_shap_exemplars --results-path $RESULTS_DIR --yes

echo "All experiments finished! Run compare_approaches.py to view results."
