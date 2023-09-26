#!/bin/bash

# After updating, make script executable with: chmod +x run_multiple.sh
# Run script with: ./run_multiple.sh

# Optimize positive weight
python main.py \
    --name 'pos_weight_0_4' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override POSITIVE_WEIGHT 0.4 \
    --use-wandb 


python main.py \
    --name 'pos_weight_0_45' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override POSITIVE_WEIGHT 0.45 \
    --use-wandb 


python main.py \
    --name 'pos_weight_0_5' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override POSITIVE_WEIGHT 0.5 \
    --use-wandb 


python main.py \
    --name 'pos_weight_0_55' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override POSITIVE_WEIGHT 0.55 \
    --use-wandb 

python main.py \
    --name 'pos_weight_0_6' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override POSITIVE_WEIGHT 0.6 \
    --use-wandb 


# Try different batch sizes
python main.py \
    --name 'batch_size_16' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override BATCH_SIZE 16 \
    --use-wandb

python main.py \
    --name 'batch_size_32' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override BATCH_SIZE 32 \
    --use-wandb

python main.py \
    --name 'batch_size_64' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override BATCH_SIZE 64 \
    --use-wandb

python main.py \
    --name 'batch_size_128' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override BATCH_SIZE 128 \
    --use-wandb

# Run for many epochs
python main.py \
    --name 'epochs_10' \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --override EPOCHS 10 \
    --use-wandb