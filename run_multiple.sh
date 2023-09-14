#!/bin/bash

# After updating, make script executable with: chmod +x run_multiple.sh
# Run script with: ./run_multiple.sh

# Runs to optimize temperature (using all the labels)
python train.py --name 'train_label_embedding_t0.25_batch_only' --override TEMPERATURE 0.1 --use-wandb
python train.py --name 'train_label_embedding_t0.1_batch_only' --override TEMPERATURE 0.09 --use-wandb
python train.py --name 'train_label_embedding_t0.075_batch_only' --override TEMPERATURE 0.08 --use-wandb
python train.py --name 'train_label_embedding_t0.075_batch_only' --override TEMPERATURE 0.07 --use-wandb
python train.py --name 'train_label_embedding_t0.075_batch_only' --override TEMPERATURE 0.06 --use-wandb

# python train.py --name 'base_run_t3' --override TEMPERATURE 0.05 --use-wandb
# python train.py --name 'base_run_t4' --override TEMPERATURE 0.01 --use-wandb
# python train.py --name 'let_label_embedding_train_t1' --override TEMPERATURE 1 --use-wandb
# python train.py --name 'let_label_embedding_train_t1' --override TEMPERATURE 0.5 --use-wandb
# python train.py --name 'let_label_embedding_train_t1' --override TEMPERATURE 0.05 --use-wandb
# python train.py --name 'let_label_embedding_train_t1' --override TEMPERATURE 0.01 --use-wandb

# # Runs to optimize temperature (batch labels only)
# python train.py --name 'base_run_t1_batch_labels_only' --override TEMPERATURE 1 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'base_run_t2_batch_labels_only' --override TEMPERATURE 0.5 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'base_run_t3_batch_labels_only' --override TEMPERATURE 0.05 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'base_run_t4_batch_labels_only' --override TEMPERATURE 0.01 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_label_embedding_train_t1_batch_labels_only' --override TEMPERATURE 1 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_label_embedding_train_t2_batch_labels_only' --override TEMPERATURE 0.5 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_label_embedding_train_t3_batch_labels_only' --override TEMPERATURE 0.05 USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_label_embedding_train_t4_batch_labels_only' --override TEMPERATURE 0.01 USE_BATCH_LABELS_ONLY True --use-wandb

# # Runs with default config (using all the labels)
# python train.py --name 'base_run' --use-wandb
# python train.py --name 'let_label_embedding_matrix_train' --override TRAIN_LABEL_EMBEDDING_MATRIX True --use-wandb
# python train.py --name 'let_sequence_embedding_matrix_train' --override TRAIN_SEQUENCE_EMBEDDING_MATRIX True TRAIN_LABEL_EMBEDDING_MATRIX True --use-wandb
# python train.py --name 'let_both_matrices_train' --override TRAIN_LABEL_EMBEDDING_MATRIX True --use-wandb
# # python train.py --name 'let_label_encoder_train' --override TRAIN_LABEL_ENCODER True --use-wandb

# # Runs with default config (batch labels only)
# python train.py --name 'base_run_batch_labels_only' --override USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_label_embedding_matrix_train_batch_labels_only' --override TRAIN_LABEL_EMBEDDING_MATRIX True USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_sequence_embedding_matrix_train_batch_labels_only' --override TRAIN_SEQUENCE_EMBEDDING_MATRIX True TRAIN_LABEL_EMBEDDING_MATRIX True USE_BATCH_LABELS_ONLY True --use-wandb
# python train.py --name 'let_both_matrices_train_batch_labels_only' --override TRAIN_LABEL_EMBEDDING_MATRIX True USE_BATCH_LABELS_ONLY True --use-wandb
