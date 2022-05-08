#!/bin/bash
# This script evaluate a QA model on the MLQA dataset.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Get to root directory, one level before scripts folder
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# ENV_DIR=${SCRIPT_DIR}/env
# source ${ENV_DIR}/bin/activate
# conda activate all-purpose-gpu

MODEL_DIR=${ROOT_DIR}/notebooks/squad-qa-minilmv2-XLMTokeinizer-8
CONTEXT_LANG=$1
QUESTION_LANG=$2
TEST_SET=$3

EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate/${TEST_SET}
mkdir -p ${EVALUATE_DIR}


TRANSFORMERS_DIR=${ROOT_DIR}/notebooks/Baseline
MLQA_DIR=${ROOT_DIR}/MLQA/Data/test

if [[ ${TEST_SET} == "mlqa" ]]; then
    # Select the test file from the MLQA corpus
    TEST_FILE=${MLQA_DIR}/test-context-${CONTEXT_LANG}-question-${QUESTION_LANG}.json
elif [[ ${TEST_SET} == "xquad" ]]; then
    # Select the test file from the XSQUAD datasets
    TEST_FILE=${SCRIPT_DIR}/corpora/XQUAD/xquad.${CONTEXT_LANG}.json
else
    TEST_FILE=${TEST_SET}
fi

# Add the train file path and pass it to the script to avoid errors
# TRAIN_FILE=${SCRIPT_DIR}/corpora/squad_es/$(basename ${MODEL_DIR})
# python ${TRANSFORMERS_DIR}/run_qa.py \
#          --model_name_or_path ${MODEL_DIR} \
#         #  --train_file ${TRAIN_FILE} \
#          --do_predict \
#          --output_dir ${MODEL_DIR}
#          --predict_file ${TEST_FILE} \
#          --overwrite_cache \
#          --n_best_size 5 \
python ${TRANSFORMERS_DIR}/run_qa.py \
    --model_name_or_path ${MODEL_DIR}  \
    # --train_file ${TRAIN_FILE} \
    # --validation_file /home/jcanete/ft-spanish-models/datasets/QA/MLQA/mlqa-dev.json \
    --test_file ${TEST_FILE}  \
    # --max_seq_length 512 \
    # --pad_to_max_length False \
    --output_dir ${MODEL_DIR} \
    # --do_eval \
    --do_predict \
    # --per_device_eval_batch_size 64 \
    # --logging_dir /data/jcanete/evaluation/mlqa \
    --seed 42 \
    # --cache_dir /data/jcanete/cache \
    # --use_auth_token True \
    --fp16 \

PREDICTION_FILE=${MODEL_DIR}/predictions_.json
EVALUATION_FILE=${EVALUATE_DIR}/$(basename ${MODEL_DIR})_eval

# Evaluate with the MLQA original evaluation script that is an
# extension of the official SQUAD evaluation script
python ${MLQA_DIR}/mlqa_evaluation_v1.py \
       ${TEST_FILE} \
       ${PREDICTION_FILE} \
       ${CONTEXT_LANG} \
       >> ${EVALUATION_FILE}