#!/bin/bash

# Define constants
MODEL_NAME=$1
DATASET=$2
ITER_NUM=$3
CUDA_DEVICE=$4
BATCH_SIZE=$5
LOGFILE="output_${DATASET}_${MODEL_NAME}.log"

# Clear the existing log file
> $LOGFILE

# First command
echo "Running command 1 for MODEL_NAME=${MODEL_NAME}, DATASET=${DATASET}" >> $LOGFILE
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE && poetry run python main.py \
    --model hf-causal \
    --model_args pretrained=../results/checkpoint_run_name_run_${DATASET}__model_${MODEL_NAME}/run_${DATASET}_${MODEL_NAME}_${ITER_NUM}_hf,dtype="float16" \
    --tasks arc_easy,arc_challenge \
    --batch_size=$BATCH_SIZE \
    --num_fewshot=25 \
    --device cuda >> $LOGFILE 2>&1

# Additional commands
echo "Running command 2 for MODEL_NAME=${MODEL_NAME}, DATASET=${DATASET}" >> $LOGFILE
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE && poetry run python main.py \
    --model hf-causal \
    --model_args pretrained=../results/checkpoint_run_name_run_${DATASET}__model_${MODEL_NAME}/run_${DATASET}_${MODEL_NAME}_${ITER_NUM}_hf,dtype="float16" \
    --tasks hellaswag \
    --batch_size=$BATCH_SIZE \
    --num_fewshot=10 \
    --device cuda >> $LOGFILE 2>&1

echo "Running command 3 for MODEL_NAME=${MODEL_NAME}, DATASET=${DATASET}" >> $LOGFILE
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE && poetry run python main.py \
    --model hf-causal \
    --model_args pretrained=../results/checkpoint_run_name_run_${DATASET}__model_${MODEL_NAME}/run_${DATASET}_${MODEL_NAME}_${ITER_NUM}_hf,dtype="float16" \
    --tasks winogrande \
    --batch_size=$BATCH_SIZE \
    --num_fewshot=0 \
    --device cuda >> $LOGFILE 2>&1
