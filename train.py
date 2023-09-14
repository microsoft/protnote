import logging
from src.utils.data import (
    read_pickle,
    load_model_weights,
    seed_everything,
    create_ordered_tensor,
    get_tokenized_labels_dataloader,
)
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.utils.evaluation import EvalMetrics
from src.utils.models import count_parameters_by_layer
from src.utils.configs import get_setup
import numpy as np
import torch
import wandb
import os
import datetime
import argparse
import json
import random
import time
from src.utils.models import (
    load_model_and_tokenizer,
    tokenize_inputs,
    get_embeddings_from_tokens,
)
from torch.utils.data import DataLoader, TensorDataset

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train and/or Test the ProTCL model.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "test", "both"],
    default="both",
    help="Specify the mode: 'train', 'test', or 'both'. Default is 'both'.",
)
parser.add_argument(
    "--use-wandb",
    action="store_true",
    default=False,
    help="Use Weights & Biases for logging. Default is False.",
)
parser.add_argument(
    "--load-model",
    type=str,
    default=None,
    help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.",
)
parser.add_argument(
    "--name",
    type=str,
    default="ProTCL",
    help="Name of the W&B run. If not provided, a name will be generated.",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/base_config.yaml",
    help="(Relative) path to the configuration file.",
)
parser.add_argument(
    "--override", nargs="*", help="Override parameters in key-value pairs."
)

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config, run_name=args.name, overrides=args.override
)

# Create datasets
train_dataset, val_dataset, test_dataset = ProteinDataset.create_multiple_datasets(
    paths_list
)

# Initialize new run
logger.info(
    f"################## {timestamp} RUNNING train.py ##################")

# Initialize W&B, if using
if args.use_wandb:
    wandb.init(
        project="protein-functions",
        name=f"{args.name}_{timestamp}",
        config={**params, **vars(args)},
    )

# Log the configuration and arguments
logger.info(f"Configuration: {config}")
logger.info(f"Arguments: {args}")

# Define data loaders
train_loader, val_loader, test_loader = create_multiple_loaders(
    [train_dataset, val_dataset, test_dataset],
    [
        params["TRAIN_BATCH_SIZE"],
        params["VALIDATION_BATCH_SIZE"],
        params["TEST_BATCH_SIZE"],
    ],
    num_workers=params["NUM_WORKERS"],
    pin_memory=True,
)

# Load sequence embeddings
sequence_embedding_matrix = sequence_encoder = None
if not params["TRAIN_SEQUENCE_ENCODER"]:
    # TODO: Rather than loading from file, create from ProteInfer itself (slower at startup, but more flexible)
    sequence_embedding_matrix = create_ordered_tensor(
        paths["SEQUENCE_EMBEDDING_PATH"],
        train_dataset.sequence_id2int,
        params["PROTEIN_EMBEDDING_DIM"],
        device,
    )
    logger.info("Loaded sequence embeddings.")

# Load label embeddings or label encoder
label_embedding_matrix = label_encoder = tokenized_labels_dataloader = None
if not params["TRAIN_LABEL_ENCODER"]:
    # TODO: Rather than loading from file, create from the model itself (slower at startup, but more flexible)
    label_embedding_matrix = create_ordered_tensor(
        paths["LABEL_EMBEDDING_PATH"],
        train_dataset.label2int,
        params["LABEL_EMBEDDING_DIM"],
        device,
    )
    logger.info("Loaded label embeddings.")
# Otherwise, load the pre-tokenized labels
else:
    tokenized_labels_dataloader = get_tokenized_labels_dataloader(
        go_descriptions_path=paths["GO_DESCRIPTIONS_PATH"],
        llm_checkpoint_path=params["PUBMEDBERT_CHECKPOINT"],
        train_label_encoder=params["TRAIN_LABEL_ENCODER"],
        label_vocabulary=train_dataset.label_vocabulary,
        label2int_mapping=train_dataset.label2int,
        batch_size=params["LABEL_BATCH_SIZE"],
        device=device,
    )

# Seed everything so we don't go crazy
seed_everything(params["SEED"], device)
# Initialize the models

# TODO: Initialize ProteInfer and PubMedBERT here as well as the ensemble (ProTCL), which should take the other two as optional arguments

model = ProTCL(
    protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
    label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
    latent_dim=params["LATENT_EMBEDDING_DIM"],
    temperature=params["TEMPERATURE"],
    label_encoder=label_encoder,
    tokenized_labels_dataloader=tokenized_labels_dataloader,
    sequence_encoder=sequence_encoder,
    sequence_embedding_matrix=sequence_embedding_matrix,
    label_embedding_matrix=label_embedding_matrix,
    train_projection_head=params["TRAIN_PROJECTION_HEAD"],
    train_label_embeddings=params["TRAIN_LABEL_EMBEDDING_MATRIX"],
    train_sequence_embeddings=params["TRAIN_SEQUENCE_EMBEDDING_MATRIX"],
    train_label_encoder=params["TRAIN_LABEL_ENCODER"],
    train_sequence_encoder=params["TRAIN_SEQUENCE_ENCODER"],
).to(device)

# Initialize trainer class to handle model training, validation, and testing
Trainer = ProTCLTrainer(
    model=model,
    device=device,
    config=config,
    logger=logger,
    timestamp=timestamp,
    run_name=args.name,
    use_wandb=args.use_wandb,
)

# Log the number of parameters by layer
count_parameters_by_layer(model)

# Load the model weights if --load-model argument is provided
if args.load_model:
    load_model_weights(model, os.path.join(ROOT_PATH, args.load_model))
    logger.info(f"Loading model weights from {args.load_model}...")

####### TRAINING AND VALIDATION LOOPS #######
if args.mode in ["train", "both"]:
    # Train function
    Trainer.train(train_loader=train_loader, val_loader=val_loader)
else:
    logger.info("Skipping training...")

####### TESTING LOOP #######
if args.mode in ["test", "both"]:
    logger.info("Starting testing...")
    best_val_th = params["DECISION_TH"]

    # If no decision threshold is provided, find the optimal threshold on the validation set
    if params["DECISION_TH"] is None:
        logger.info("Decision threshold not provided.")

        best_val_th, best_val_score = Trainer.find_optimal_threshold(
            data_loader=val_loader, optimization_metric_name="f1_micro"
        )
    # Evaluate model on test set
    eval_metrics = EvalMetrics(
        num_labels=params["NUM_LABELS"], threshold=best_val_th, device=device
    ).get_metric_collection(type="all")

    final_metrics = Trainer.evaluate(
        data_loader=test_loader, eval_metrics=eval_metrics, testing=True
    )
    # Convert all metrics to float
    final_metrics = {
        k: (v.item() if isinstance(v, torch.Tensor) else v)
        for k, v in final_metrics.items()
    }

    logger.info(json.dumps(final_metrics, indent=4))
    logger.info("Testing complete.")

# Close the W&B run
if args.use_wandb:
    # TODO: Check to ensure W&B is logging these test metrics correctly
    wandb.log(final_metrics)
    wandb.finish()

# Clear GPU cache
torch.cuda.empty_cache()

logger.info("################## train.py COMPLETE ##################")
