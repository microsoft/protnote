import ipdb
import logging
from src.utils.data import read_pickle, load_model_weights, read_yaml, load_gz_json, load_embeddings
from src.utils.losses import contrastive_loss
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCL import ProTCL
from src.utils.evaluation import EvalMetrics
from src.utils.proteinfer import normalize_confidences
from src. utils.models import count_parameters_by_layer
from torchmetrics.classification import F1Score
from typing import Literal
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
parser.add_argument('--train-label-encoder', action='store_true',
                    default=False, help="Train the label encoder. Default is False.")
parser.add_argument('--train-sequence-encoder', action='store_true',
                    default=False, help="Train the sequence encoder. Default is False.")
parser.add_argument('--train-label-embedding-matrix', action='store_true',
                    default=False, help="Train the label nn.Embedding layer. Default is False.")
parser.add_argument('--train-sequence-embedding-matrix', action='store_true',
                    default=False, help="Train the sequence nn.Embedding layer. Default is False.")
parser.add_argument('--use-wandb', action='store_true', default=False,
                    help="Use Weights & Biases for logging. Default is False.")
parser.add_argument('--load-model', type=str, default=None,
                    help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.")
parser.add_argument('--name', type=str, default=None,
                    help="Name of the W&B run. If not provided, a name will be generated.")
#TODO: Make Optimization metric and normalize probabilities part of arguments
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
log_filename = f'{timestamp}_{args.name}.log' if args.name else 'train.log'
full_log_path = os.path.join(log_dir, log_filename)
logging.basicConfig(filename=full_log_path, filemode='w',
                    format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S %Z')


# Initialize new run
logging.info(
    f"################## {timestamp} RUNNING train.py ##################")

# Initialize W&B, if using
if args.use_wandb:
    wandb.init(project="protein-functions",
               name=f"{args.name if args.name else 'ProTCL'}_{timestamp}", config={**params, **vars(args)})

# Log the configuration and arguments
logging.info(f"Configuration: {config}")
logging.info(f"Arguments: {args}")

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
if not args.train_sequence_encoder:
    sequence_embedding_matrix = load_embeddings(
        paths['SEQUENCE_EMBEDDING_PATH'],
        sequence_id_map,
        params['PROTEIN_EMBEDDING_DIM'],
        device
    )
    logging.info("Loaded sequence embeddings.")

# Load label embeddings
label_embedding_matrix = None
if not args.train_label_encoder:
    label_embedding_matrix = load_embeddings(
        paths['LABEL_EMBEDDING_PATH'],
        train_dataset.label2int,
        params['LABEL_EMBEDDING_DIM'],
        device
    )
    logging.info("Loaded label embeddings.")

# Seed everything so we don't go crazy
random.seed(params['SEED'])
np.random.seed(params['SEED'])
torch.manual_seed(params['SEED'])
if device == 'cuda':
    torch.cuda.manual_seed_all(params['SEED'])

# Evaluate function

def evaluation_step(batch,
                    model: torch.nn.Module,
                    train_sequence_encoder: bool,
                    device:str,
                    use_batch_labels_only: bool = True)->tuple:
    
    """Perform a single evaluation step.

    :param batch: _description_
    :type batch: _type_
    :param model: pytorch model
    :type model: torch.nn.Module
    :param train_sequence_encoder: whether to train the sequence encoder
    :type train_sequence_encoder: bool
    :param device: decide for training on cpu or gpu
    :type device: str
    :param use_batch_labels_only: whether to use only the labels present in the batch for calculating loss, defaults to True
    :type use_batch_labels_only: bool, optional
    :return: batch loss, logits and labels
    :rtype: tuple
    """    
    
    # Unpack the validation batch
    sequence_ids, sequence_onehots, label_multihots, sequence_lengths = batch

    # Move sequences and labels to GPU, if available
    labels = label_multihots.to(device)
    sequences = sequence_onehots.to(
        device) if train_sequence_encoder else sequence_ids.to(device)

    # Forward pass
    P_e, L_e = model(sequences,
                        labels if use_batch_labels_only else torch.ones_like(
                            labels)
                        )

    # Compute validation loss for the batch
    loss = contrastive_loss(P_e,
                            L_e,
                            model.t,
                            labels[:, torch.any(
                                labels, dim=0)] if use_batch_labels_only else labels
                            )
    # Compute temperature normalized cosine similarities
    logits = torch.mm(P_e,
                        L_e.t()) / model.t
    
    return loss.item(), logits, labels
    
def find_optimal_threshold(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           device:str,
                           train_sequence_encoder: bool,
                           num_labels:int,
                           average:Literal['micro','macro','weighted'],
                           optimization_metric_name: str,
                           normalize_probabilities: bool = False,
                           label_normalizer: dict = None,
                           label_vocabulary: list = None):
    
    model.eval()

    best_th = 0.0
    best_score = 0.0

    # Reset eval metrics just in case
    with torch.no_grad():
        all_probabilities = []
        all_labels = []
        for batch in data_loader:
            loss,logits,labels = evaluation_step(batch=batch,
                                                 model=model,
                                                 train_sequence_encoder=train_sequence_encoder,
                                                 device=device,
                                                 use_batch_labels_only = False)

            # Apply sigmoid to get the probabilities for multi-label classification
            probabilities = torch.sigmoid(logits)

            if normalize_probabilities:
                
                # TODO: Using original normalize_confidences implemented with numpy,
                # but this is slow. Should be able to do this with torch tensors.
                probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                                                    label_vocab=label_vocabulary,
                                                                    applicable_label_dict=label_normalizer),
                                                                    device=probabilities.device)
            all_probabilities.append(probabilities)
            all_labels.append(labels)
            
        all_probabilities = torch.cat(all_probabilities)
        all_labels = torch.cat(all_labels)

    for th in np.arange(0.1, 1, 0.01):
        eval_metrics = EvalMetrics(num_labels=num_labels,average=average,threshold=th,device=device)
        optimization_metric = getattr(eval_metrics,optimization_metric_name)

        optimization_metric(all_probabilities, all_labels)
        score = optimization_metric.compute()
        if score > best_score:
            best_score = score
            best_th = th
        print('TH:', th, 'F1:', score)
    logging.info(f'Best Val F1: {best_score.item()}, Best Val TH: {best_th}')

    return best_th, best_score

def evaluate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             device: str,
             train_sequence_encoder: bool,
             use_batch_labels_only: bool = True,
             eval_metrics: EvalMetrics = None,
             normalize_probabilities: bool = False,
             label_normalizer: dict = None,
             label_vocabulary: list = None) -> dict:
    """Evaluate the model on the given data loader.

    :param model: pytorch model
    :type model: torch.nn.Module
    :param data_loader: pytorch data loader
    :type data_loader: torch.utils.data.DataLoader
    :param device: decide for training on cpu or gpu
    :type device: str
    :param use_batch_labels_only: whether to use only the labels present in the batch for calculating loss, defaults to True
    :type use_batch_labels_only: bool, optional
    :param train_sequence_encoder: whether to train the sequence encoder
    :type train_sequence_encoder: bool
    :param eval_metrics: an eval metrics class to calculate metrics like F1 score, defaults to None
    :type eval_metrics: EvalMetrics, optional
    :param normalize_probabilities: whether to normalize probabilities to respect go hierarchy, defaults to False
    :type normalize_probabilities: bool, optional
    :param label_normalizer: dictionary with go hierarchy, defaults to None. Only used if normalize_probabilities is True
    :type label_normalizer: dict, optional
    :param label_vocabulary: go label vocabulary list, defaults to None. Only used if normalize_probabilities is True
    :type label_vocabulary: list, optional
    :return: dictionary with evaluation metrics. Always return avg_loss and if eval_metrics is not None, it will return the metrics from eval_metrics.compute()
    :rtype: dict
    """
    if use_batch_labels_only & (eval_metrics is not None):
        raise ValueError(
            "Cannot use batch labels only and eval metrics at the same time.")

    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            loss,logits,labels = evaluation_step(batch=batch,
                                                 model=model,
                                                 train_sequence_encoder=train_sequence_encoder,
                                                 device=device,
                                                 use_batch_labels_only=use_batch_labels_only)
            if eval_metrics is not None:
                # Apply sigmoid to get the probabilities for multi-label classification
                probabilities = torch.sigmoid(logits)

                if normalize_probabilities:
                    # TODO: Using original normalize_confidences implemented with numpy,
                    # but this is slow. Should be able to do this with torch tensors.
                    probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                                                        label_vocab=label_vocabulary,
                                                                        applicable_label_dict=label_normalizer),
                                                                        device=probabilities.device)
                # Update eval metrics
                eval_metrics(probabilities, labels)

            # Accumulate loss
            total_loss += loss

        # Compute average validation loss
        avg_loss = total_loss / len(data_loader)
        final_metrics = eval_metrics.compute() if eval_metrics is not None else {}
        final_metrics.update({"avg_loss": avg_loss})

    return final_metrics


def train(model,
          num_epochs: int,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: str,
          use_batch_labels_only: bool,
          train_sequence_encoder: bool,
          validation_frequency: int,
          output_model_dir: str
          ):
    """_summary_

    :param model: _description_
    :type model: _type_
    :param num_epochs: _description_
    :type num_epochs: int
    :param train_loader: _description_
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: _description_
    :type val_loader: torch.utils.data.DataLoader
    :param optimizer: _description_
    :type optimizer: torch.optim.Optimizer
    :param device: _description_
    :type device: str
    :param use_batch_labels_only: _description_
    :type use_batch_labels_only: bool
    :param train_sequence_encoder: _description_
    :type train_sequence_encoder: bool
    :param validation_frequency: _description_
    :type validation_frequency: int
    :param output_model_dir: _description_
    :type output_model_dir: str
    """    

    model.train()
    # Watch the model
    if args.use_wandb:
        wandb.watch(model)

    # Keep track of batches for logging
    batch_count = 0

    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        ####### TRAINING LOOP #######
        for train_batch in train_loader:
            # Unpack batch
            sequence_ids, sequence_onehots, label_multihots, sequence_lengths = train_batch

            # Move sequences and labels to GPU, if available
            labels = label_multihots.to(device)
            sequences = sequence_onehots.to(
                device) if args.train_sequence_encoder else sequence_ids.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            P_e, L_e = model(sequences, labels)

            # Compute target, if only using the labels that are present in the batch
            target = labels[:, torch.any(labels, dim=0)]

            # Compute loss
            loss = contrastive_loss(P_e, L_e, model.t, target)

            # Log metrics to W&B
            if args.use_wandb:
                wandb.log({"training_loss": loss.item()})

            # Backward pass
            loss.backward()
            optimizer.step()

            # Increment batch count
            batch_count += 1

            # Run validation and log progress every 20 batches
            if batch_count % validation_frequency == 0:

                ####### VALIDATION LOOP #######
                logging.info("Running validation...")

                val_metrics = evaluate(model=model,
                                       data_loader=val_loader,
                                       device=device,
                                       train_sequence_encoder=train_sequence_encoder,
                                       use_batch_labels_only=use_batch_labels_only
                                       )

                logging.info(val_metrics)
                logging.info(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Training Loss: {loss.item()}, Validation Loss: {val_metrics['avg_loss']}")

                if args.use_wandb:
                    wandb.log({"validation_loss": val_metrics['avg_loss']})

                # Save the model if it has the best validation loss so far
                if val_metrics['avg_loss'] < best_val_loss:
                    logging.info(
                        f"New best validation loss: {val_metrics['avg_loss']}. Saving model...")
                    best_val_loss = val_metrics['avg_loss']

                    # Save model to OUTPUT_MODEL_DIR. Create path if it doesn't exist.
                    if not os.path.exists(paths['OUTPUT_MODEL_DIR']):
                        os.makedirs(paths['OUTPUT_MODEL_DIR'])

                    model_name = args.name if args.name else f"best_ProTCL.pt"
                    model_path = os.path.join(
                        output_model_dir, f"{timestamp}_{model_name}.pt")
                    torch.save(model.state_dict(), model_path)
                    logging.info(f"Saved model to {model_path}.")

                    if args.use_wandb:
                        wandb.save(f"{timestamp}_best_ProTCL.pt")


# Initialize the model
model = ProTCL(protein_embedding_dim=params['PROTEIN_EMBEDDING_DIM'],
               label_embedding_dim=params['LABEL_EMBEDDING_DIM'],
               latent_dim=params['LATENT_EMBEDDING_DIM'],
               temperature=params['TEMPERATURE'],
               sequence_embedding_matrix=sequence_embedding_matrix,
               train_label_embeddings=args.train_label_encoder or args.train_label_embedding_matrix,
               label_embedding_matrix=label_embedding_matrix,
               train_sequence_embeddings=args.train_sequence_encoder or args.train_sequence_embedding_matrix,
               ).to(device)

# Log the number of parameters by layer
count_parameters_by_layer(model)

# Load the model weights if --load-model argument is provided
if args.load_model:
    load_model_weights(model, os.path.join(ROOT_PATH, args.load_model))
    logging.info(f"Loading model weights from {args.load_model}...")

if args.mode in ['train', 'both']:
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['LEARNING_RATE'])

    # Log that training is starting
    logging.info("Starting training...")

    # Train function
    train(model,
          num_epochs=params['NUM_EPOCHS'],
          train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          device=device,
          train_sequence_encoder=args.train_sequence_encoder,
          use_batch_labels_only=True,
          validation_frequency=params['VALIDATION_FREQUENCY'],
          output_model_dir=paths['OUTPUT_MODEL_DIR']
          )

else:
    logging.info("Skipping training...")


####### TESTING LOOP #######
if args.mode in ['test', 'both']:

    logging.info("Starting testing...")
    #label_normalizer = load_gz_json(paths['PARENTHOOD_LIB_PATH'])

    if params['DECISION_TH'] is None:
        logging.info("Decision threshold not provided. Finding optimal threshold on validation set...")
        best_val_th,best_val_score = find_optimal_threshold(model=model,
                                                            data_loader=val_loader,
                                                            device=device,
                                                            train_sequence_encoder=args.train_sequence_encoder,
                                                            num_labels=train_dataset.label_vocabulary_size,
                                                            average=params['METRICS_AVERAGE'],
                                                            optimization_metric_name='f1',
                                                            )

        logging.info(f"Best validation threshold: {best_val_th}, Best validation score: {best_val_score}")
        params['DECISION_TH'] = best_val_th

    # Evaluate model on test set
    eval_metrics = EvalMetrics(num_labels=train_dataset.label_vocabulary_size,
                               threshold=params['DECISION_TH'],
                               average=params['METRICS_AVERAGE'],
                               device=device)

    final_metrics = evaluate(model=model,
                             data_loader=test_loader,
                             eval_metrics=eval_metrics,
                             device=device,
                             use_batch_labels_only=False,
                             train_sequence_encoder=args.train_sequence_encoder)

    logging.info(json.dumps(final_metrics, indent=4))

    logging.info("Testing complete.")

# Close the W&B run
if args.use_wandb:
    wandb.log(final_metrics)
    wandb.finish()

# Clear GPU cache
torch.cuda.empty_cache()

logging.info("################## train.py COMPLETE ##################")
