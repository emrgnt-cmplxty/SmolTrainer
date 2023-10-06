#!/bin/bash

####################################################################################################
# EXAMPLE USE
# Move script to home directory and make executable
# cp run_train.sh .  && chmod +x run_train.sh
# Run training with gpu 0, pythia-410m, sciphi-python-textbook, block_size 1024
# ex - ./run_train.sh 0 pythia-410m sciphi-python-textbook 1024
####################################################################################################

# Define constants

# Default value for block_size
BLOCK_SIZE=${4:-1024}

# Define datasets
declare -a DATASETS=("sciphi-python-textbook" "sciphi-textbooks-are-all-you-need" "programming-books-llama" "open-orca" "sciphi_combine" "vikp_sciphi_combine" "vikp_sciphi_orca_combine" "tiny-textbooks")
# Corresponding CUDA devices
declare -a DEVICES=("$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8")

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    CUDA_DEVICE="${DEVICES[$i]}"
    echo "Running for DATASET=${DATASET} on CUDA_DEVICE=${CUDA_DEVICE}"
    
    nohup sh -c "
      export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  && poetry run python smol_trainer/runner.py \
        --model-name=$2 \
        --dataset=$DATASET \
        --batch-size=8 \
        --block-size=$BLOCK_SIZE \
        --eval-iters=250 \
        --compile=True \
        --device=cuda \
        --eval-interval=1000 \
        --max_checkpoints=10 \
        --wandb-log \
        --run-name=run_$DATASET" \
      > output_$DATASET.log 2>&1 &
done
