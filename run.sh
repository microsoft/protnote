#!/bin/bash

# After updating, make script executable with: chmod +x run.sh
# Run script with: ./run.sh

# Load previous best model to continue training
python main.py \
    --name 'Weighted Sampling 100 Epochs' \
    --full-path-name FULL_DATA_PATH \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --use-wandb

# Optimize gradient clipping norm
# python main.py \
#     --name "FL{CLIP: 10, EPOCH: 30, ALPHA: 0.9}" \
#     --full-path-name FULL_DATA_PATH \
#     --train-path-name TRAIN_DATA_PATH \
#     --validation-path-name VAL_DATA_PATH \
#     --use-wandb 

# Sleep for 10 seconds to allow everything to cleanly exit
# sleep 10

# python main.py \
#     --name 'FL{CLIP: 10, EPOCH: 10, ALPHA: 0.9}' \
#     --full-path-name FULL_DATA_PATH \
#     --train-path-name TRAIN_DATA_PATH \
#     --validation-path-name VAL_DATA_PATH \
#     --override FOCAL_LOSS_ALPHA 0.9 \
#     --use-wandb 
    # --test-paths-names TEST_DATA_PATH \

# Example for reference:
# python main.py \
#     --name 'gradient_clipping{CLIP_VALUE:10}' \
#     --full-path-name "FULL_DATA_PATH" \
#     --train-path-name TRAIN_DATA_PATH \
#     --validation-path-name VAL_DATA_PATH \
#     --test-paths-names TEST_DATA_PATH \
#     --override CLIP_VALUE 10 NUM_EPOCHS 50 \
#     --use-wandb 

