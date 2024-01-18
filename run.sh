#!/bin/bash

# Make script executable with: chmod +x run.sh
# Run script with: ./run.sh

python main.py \
    --name '{A100}{big mlp}{condensed labels}{weighted sampling 0.5}{last token}' \
    --full-path-name FULL_DATA_PATH \
    --train-path-name TRAIN_DATA_PATH \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --use-wandb

# Sleep for 10 seconds to allow everything to cleanly exit
sleep 10