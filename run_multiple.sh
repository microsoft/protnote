#!/bin/bash

# After updating, make script executable with: chmod +x run_multiple.sh
# Run script with: ./run_multiple.sh

python train.py --name 'base_run' --use-wandb
python train.py --name 'let_label_embedding_matrix_train' --train-label-embedding-matrix --use-wandb
python train.py --name 'let_sequence_embedding_matrix_train' --train-sequence-embedding-matrix --use-wandb
python train.py --name 'let_both_matrices_train' --train-sequence-embedding-matrix --train-label-embedding-matrix --use-wandb