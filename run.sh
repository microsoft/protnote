#!/bin/bash

# Make script executable with: chmod +x run.sh
# Run script with: ./run.sh

python main.py \
    --name 'Test Unseen Sequences' \
    --test-paths-names UNSEEN_DATA_PATH_ZERO_SHOT \
    --override EXTRACT_VOCABULARIES_FROM null \
    --load-model '/home/ncorley/proteins/ProteinFunctions/data/models/zero_shot/2024-03-05_16-43-59_{zero shot}{medium MLP}{names}{mean}{sequence augmentations 0.1}{label noise 40}.pt'

# Sleep for 10 seconds to allow everything to cleanly exit
sleep 10
# --override EXTRACT_VOCABULARIES_FROM null OUTPUT_MLP_HIDDEN_DIM_SCALE_FACTOR 3 OUTPUT_MLP_NUM_LAYERS 3 PROJECTION_HEAD_NUM_LAYERS 4 PROJECTION_HEAD_HIDDEN_DIM_SCALE_FACTOR 3 \