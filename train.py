import logging
from src.utils.data import read_pickle, load_model_weights, read_yaml, create_ordered_tensor
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
from src.utils.models import load_PubMedBERT

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
parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                    help="(Relative) path to the configuration file.")
parser.add_argument('--override', nargs='*',
                    help='Override parameters in key-value pairs.')

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

# Ensure there's an even number of override arguments
if args.override and len(args.override) % 2 != 0:
    raise ValueError("Overrides must be provided as key-value pairs.")

# Get the root path from the environment variable; default to current directory if ROOT_PATH is not set
ROOT_PATH = os.environ.get('ROOT_PATH', '.')

# Load the configuration file
config = read_yaml(os.path.join(ROOT_PATH, args.config))

# Process the overrides if provided
if args.override:
    if len(args.override) % 2 != 0:
        raise ValueError("Overrides must be provided as key-value pairs.")

    # Convert the list to a dictionary
    overrides = {args.override[i]: args.override[i+1]
                 for i in range(0, len(args.override), 2)}

    # Update the config with the overrides
    for key, value in overrides.items():
        # Convert value to appropriate type if necessary (e.g., float, int)
        # Here, we're assuming that the provided keys exist in the 'params' section of the config
        if key in config['params']:
            config['params'][key] = type(config['params'][key])(value)
        else:
            raise KeyError(
                f"Key '{key}' not found in the 'params' section of the config.")

# Extract the parameters and paths from the (possibly overidden) config file
params = config['params']
paths = {key: os.path.join(ROOT_PATH, value)
         for key, value in config['relative_paths'].items()}

# Set the timezone for the entire Python environment
os.environ['TZ'] = 'US/Pacific'
time.tzset()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()

# Initialize logging
# TODO: Find a way to give W&B access to the log file
log_dir = os.path.join(ROOT_PATH, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
full_log_path = os.path.join(log_dir, f'{timestamp}_train_{args.name}.log')
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
sequence_embedding_matrix = sequence_encoder = None
if not params['TRAIN_SEQUENCE_ENCODER']:
    sequence_embedding_matrix = create_ordered_tensor(
        paths['SEQUENCE_EMBEDDING_PATH'],
        sequence_id_map,
        params['PROTEIN_EMBEDDING_DIM'],
        device
    )
    logger.info("Loaded sequence embeddings.")

# Load label embeddings or label encoder
label_embedding_matrix = label_encoder = tokenized_labels = None
if not params['TRAIN_LABEL_ENCODER']:
    label_embedding_matrix = create_ordered_tensor(
        paths['LABEL_EMBEDDING_PATH'],
        train_dataset.label2int,
        params['LABEL_EMBEDDING_DIM'],
        device
    )
    logger.info("Loaded label embeddings.")
# Otherwise, load the pre-tokenized labels
else:
    # Load the label vocabulary
    annotations = read_pickle(
        '/home/ncorley/protein/ProteinFunctions/data/annotations/go_annotations_2019_07_01.pkl')

    # Filter the annotations df to be only the labels in label_vocab. In annotations, the go id is the index
    annotations = annotations[annotations.index.isin(
        train_dataset.label_vocabulary)]

    # Add a new column 'numeric_id' to the dataframe based on the id_map
    annotations['numeric_id'] = annotations.index.map(train_dataset.label2int)

    # Sort the dataframe by 'numeric_id'
    annotations_sorted = annotations.sort_values(by='numeric_id')

    # Extract the "label" column as a list
    sorted_labels = annotations_sorted['label'].tolist()

    # Initialize label encoder and tokenizer
    # TODO: Pass label encoder an argument to ProTCL rather than re-instantiating
    label_tokenizer, label_encoder = load_PubMedBERT(
        trainable=params['TRAIN_LABEL_ENCODER'] or params['TRAIN_LABEL_EMBEDDING_MATRIX'])
    label_encoder = label_encoder.to(device)

    # Tokenize all labels in the dataframe in a batched manner
    # The batch is ordered according to the order of the labels in the label vocabulary
    tokenized_labels = label_tokenizer(
        sorted_labels, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move the tensors to GPU if available
    tokenized_labels = {name: tensor.to(device)
                        for name, tensor in tokenized_labels.items()}


# Seed everything so we don't go crazy
random.seed(params['SEED'])
np.random.seed(params['SEED'])
torch.manual_seed(params['SEED'])
if device == 'cuda':
    torch.cuda.manual_seed_all(params['SEED'])

# Initialize the models

# TODO: Initialize ProteInfer and PubMedBERT here as well as the ensemble (ProTCL), which should take the other two as optional arguments

model = ProTCL(protein_embedding_dim=params['PROTEIN_EMBEDDING_DIM'],
               label_embedding_dim=params['LABEL_EMBEDDING_DIM'],
               latent_dim=params['LATENT_EMBEDDING_DIM'],
               temperature=params['TEMPERATURE'],
               label_encoder=label_encoder,
               tokenized_labels=tokenized_labels,
               sequence_encoder=sequence_encoder,
               sequence_embedding_matrix=sequence_embedding_matrix,
               label_embedding_matrix=label_embedding_matrix,
               # TODO: Can simplify based on whether we pass an encoder or an embedding matrix
               train_label_embeddings=params['TRAIN_LABEL_ENCODER'] or params['TRAIN_LABEL_EMBEDDING_MATRIX'],
               train_sequence_embeddings=params['TRAIN_SEQUENCE_ENCODER'] or params['TRAIN_SEQUENCE_EMBEDDING_MATRIX'],
               ).to(device)

# Initialize trainer class to handle model training, validation, and testing
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

####### TRAINING AND VALIDATION LOOPS #######
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

    # If no decision threshold is provided, find the optimal threshold on the validation set
    if params['DECISION_TH'] is None:
        logger.info("Decision threshold not provided.")

        best_val_th, best_val_score = Trainer.find_optimal_threshold(data_loader=val_loader,
                                                                     average=params['METRICS_AVERAGE'],
                                                                     optimization_metric_name='f1_macro'
                                                                     )
    # Evaluate model on test set
    eval_metrics = EvalMetrics(num_labels=train_dataset.label_vocabulary_size,
                               threshold=best_val_th,
                               device=device).get_metric_collection(type='all')
    


    final_metrics = Trainer.evaluate(data_loader=test_loader,
                                     eval_metrics=eval_metrics,
                                     testing=True)
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
