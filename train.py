#### NEW TRAIN ####

# Argments (command line:)
# Path to config file (YAML) (Default: base_config.yaml)
# Name for the run (Default: Base)
# Whether to use W&B (Default: False)
# Mode (train, test, both) (Default: Both)


# Parse everything
# Set up logger
# Set up W&B
# Define data loaders
# Load embeddings
# Seeds
# Define model

import ipdb
import logging
from src.utils.data import read_pickle, load_model_weights, read_yaml, load_gz_json, load_embeddings
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.utils.evaluation import EvalMetrics
from src. utils.models import count_parameters_by_layer
import numpy as np
import torch
import wandb
import os
import datetime
import argparse
import json
import random
import time

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train and/or Test the ProTCL model.")
parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                    help="Specify the mode: 'train', 'test', or 'both'. Default is 'both'.")
parser.add_argument('--use-wandb', action='store_true', default=False,
                    help="Use Weights & Biases for logging. Default is False.")
parser.add_argument('--load-model', type=str, default=None,
                    help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.")
parser.add_argument('--name', type=str, default="ProTCL",
                    help="Name of the W&B run. If not provided, a name will be generated.")
# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

# Get the root path from the environment variable; default to current directory if ROOT_PATH is not set
ROOT_PATH = os.environ.get('ROOT_PATH', '.')

# Load the configuration file
config = read_yaml(ROOT_PATH + '/config.yaml')
params = config['params']
paths = {key: os.path.join(ROOT_PATH, value)
         for key, value in config['relative_paths'].items()}

# Set the timezone for the entire Python environment
os.environ['TZ'] = 'US/Pacific'
time.tzset()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z")

# Initialize logging
log_dir = os.path.join(ROOT_PATH, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
full_log_path = os.path.join(log_dir, f'train_{args.name}.log')
logging.basicConfig(filename=full_log_path, filemode='w',
                    format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S %Z')

logger = logging.getLogger()

# Initialize new run
logger.info(
    f"################## {timestamp} RUNNING train.py ##################")

# Initialize W&B, if using
if args.use_wandb:
    wandb.init(project="protein-functions",
               name=f"{args.name}_{timestamp}", config={**params, **vars(args)})

# Log the configuration and arguments
logger.info(f"Configuration: {config}")
logger.info(f"Arguments: {args}")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load datasets from config file paths; the same vocabulary is used for all datasets
common_paths = {
    'amino_acid_vocabulary_path': paths['AMINO_ACID_VOCAB_PATH'],
    'label_vocabulary_path': paths['GO_LABEL_VOCAB_PATH'],
    'sequence_id_vocabulary_path': paths['SEQUENCE_ID_VOCAB_PATH'],
    'sequence_id_map_path': paths['SEQUENCE_ID_MAP_PATH'],
}
paths_list = [{**common_paths, 'data_path': paths[key]}
              for key in ['TRAIN_DATA_PATH', 'VAL_DATA_PATH', 'TEST_DATA_PATH']]
train_dataset, val_dataset, test_dataset = ProteinDataset.create_multiple_datasets(
    paths_list)

# Define data loaders
train_loader, val_loader, test_loader = create_multiple_loaders(
    [train_dataset, val_dataset, test_dataset],
    [params['TRAIN_BATCH_SIZE'], params['VALIDATION_BATCH_SIZE'],
        params['TEST_BATCH_SIZE']],
    num_workers=params['NUM_WORKERS'],
    pin_memory=True
)

# Load map from alphanumeric sequence ID's to integer sequence ID's
sequence_id_map = read_pickle(paths['SEQUENCE_ID_MAP_PATH'])

# Load sequence embeddings
sequence_embedding_matrix = None
if not params['TRAIN_SEQUENCE_ENCODER']:
    sequence_embedding_matrix = load_embeddings(
        paths['SEQUENCE_EMBEDDING_PATH'],
        sequence_id_map,
        params['PROTEIN_EMBEDDING_DIM'],
        device
    )
    logger.info("Loaded sequence embeddings.")

# Load label embeddings
label_embedding_matrix = None
if not params['TRAIN_LABEL_ENCODER']:
    label_embedding_matrix = load_embeddings(
        paths['LABEL_EMBEDDING_PATH'],
        train_dataset.label2int,
        params['LABEL_EMBEDDING_DIM'],
        device
    )
    logger.info("Loaded label embeddings.")

# Seed everything so we don't go crazy
random.seed(params['SEED'])
np.random.seed(params['SEED'])
torch.manual_seed(params['SEED'])
if device == 'cuda':
    torch.cuda.manual_seed_all(params['SEED'])

# Initialize the model
model = ProTCL(protein_embedding_dim=params['PROTEIN_EMBEDDING_DIM'],
               label_embedding_dim=params['LABEL_EMBEDDING_DIM'],
               latent_dim=params['LATENT_EMBEDDING_DIM'],
               temperature=params['TEMPERATURE'],
               sequence_embedding_matrix=sequence_embedding_matrix,
               train_label_embeddings=params['TRAIN_LABEL_ENCODER'] or params['TRAIN_LABEL_EMBEDDING_MATRIX'],
               label_embedding_matrix=label_embedding_matrix,
               train_sequence_embeddings=params['TRAIN_SEQUENCE_ENCODER'] or params['TRAIN_SEQUENCE_EMBEDDING_MATRIX'],
               ).to(device)


Trainer = ProTCLTrainer(model=model,
                              device=device,
                              config=config,
                              logger=logger,
                              timestamp=timestamp,
                              run_name=args.name,
                              use_wandb=args.use_wandb)

# Log the number of parameters by layer
count_parameters_by_layer(model)

# Load the model weights if --load-model argument is provided
if args.load_model:
    load_model_weights(model, os.path.join(ROOT_PATH, args.load_model))
    logger.info(f"Loading model weights from {args.load_model}...")

if args.mode in ['train', 'both']:

    # Train function
    Trainer.train(train_loader=train_loader,
                  val_loader=val_loader)
else:
    logger.info("Skipping training...")


####### TESTING LOOP #######
if args.mode in ['test', 'both']:
    logger.info("Starting testing...")
    best_val_th = params['DECISION_TH']

    if params['DECISION_TH'] is None:
        logger.info("Decision threshold not provided.")
        
        best_val_th, best_val_score = Trainer.find_optimal_threshold(data_loader=val_loader,
                                                                     average=params['METRICS_AVERAGE'],
                                                                     optimization_metric_name='f1'
                                                                    )
    # Evaluate model on test set
    eval_metrics = EvalMetrics(num_labels=train_dataset.label_vocabulary_size,
                               threshold=best_val_th,
                               average=params['METRICS_AVERAGE'],
                               device=device)

    final_metrics = Trainer.evaluate(data_loader=test_loader,
                                     eval_metrics=eval_metrics)
    logger.info(json.dumps(final_metrics, indent=4))
    logger.info("Testing complete.")
# Close the W&B run
if args.use_wandb:
    wandb.log(final_metrics)
    wandb.finish()

# Clear GPU cache
torch.cuda.empty_cache()

logger.info("################## train.py COMPLETE ##################")
