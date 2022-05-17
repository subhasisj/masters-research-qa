#!/bin/bash
# This script evaluate a QA model on the MLQA dataset.
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Get to root directory, one level before scripts folder
# ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# ENV_DIR=${SCRIPT_DIR}/env
# source ${ENV_DIR}/bin/activate
# conda activate all-purpose-gpu

# MODEL_DIR=${ROOT_DIR}/notebooks/squad-qa-minilmv2-XLMTokeinizer-8
# CONTEXT_LANG=$1
# QUESTION_LANG=$2
# TEST_SET=$3

# EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate/${TEST_SET}
# mkdir -p ${EVALUATE_DIR}


# TRANSFORMERS_DIR=${ROOT_DIR}/notebooks/Baseline
# MLQA_DIR=${ROOT_DIR}/MLQA/Data/test



python squad-distill.py --model_type bert --train_file ../SQuAD/translate-train/squad.translate.train.en-zh.json --model_name_or_path "subhasisj/zh-TAPT-MLM-MiniLM" \
--tokenizer_name  "xlm-roberta-base" \
--do_train \
--output_dir ./models_with_distillation \
--teacher_type xlm \
--teacher_name_or_path bhavikardeshna/xlm-roberta-base-chinese \
--predict_file ../SQuAD/translate-dev/squad.translate.dev.en-zh.json \
--num_train_epochs 5.0 \
--per_gpu_train_batch_size 4 \
--learning_rate 3e-5 \
# --no_cuda False \
# --save_total_limit 1 \


    # # --train_file ${TRAIN_FILE} \
    # # --validation_file /home/jcanete/ft-spanish-models/datasets/QA/MLQA/mlqa-dev.json \
    # --test_file ${TEST_FILE}  \
    # # --max_seq_length 512 \
    # # --pad_to_max_length False \
    # --output_dir ${MODEL_DIR} \
    # # --do_eval \
    # --do_predict \
    # # --per_device_eval_batch_size 64 \
    # # --logging_dir /data/jcanete/evaluation/mlqa \
    # --seed 42 \
    # # --cache_dir /data/jcanete/cache \
    # # --use_auth_token True \
    # --fp16 \