import logging
from src.utils.data import read_pickle, load_model_weights, read_yaml, load_gz_json, load_embeddings
from src.utils.losses import contrastive_loss
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCL import ProTCL
from src.utils.proteinfer import normalize_confidences
from src.utils.evaluation import EvalMetrics
from src. utils.models import count_parameters
import torch
import wandb
import os
import datetime
import argparse
import json
import pytz

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train and/or Test the ProTCL model.")
parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                    help="Specify the mode: 'train', 'test', or 'both'. Default is 'both'.")
parser.add_argument('--train-label-encoder', action='store_true',
                    default=False, help="Train the label encoder. Default is False.")
parser.add_argument('--train-sequence-encoder', action='store_true',
                    default=False, help="Train the sequence encoder. Default is False.")
parser.add_argument('--test-model', action='store_true',
                    default=True, help="Perform a test pass. Default is True.")
parser.add_argument('--use-wandb', action='store_true', default=False,
                    help="Use Weights & Biases for logging. Default is False.")
args = parser.parse_args()

# Get the root path from the environment variable; default to current directory if ROOT_PATH is not set
ROOT_PATH = os.environ.get('ROOT_PATH', '.')

# Load the configuration file
config = read_yaml(ROOT_PATH + '/config.yaml')
params = config['params']
paths = config['relative_paths']

# Initialize logging
logging.basicConfig(filename='train.log', filemode='w',
                    format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S %Z')

# Initialize new run
timestamp = datetime.datetime.now(pytz.timezone(
    'US/Pacific')).strftime("%Y-%m-%d_%H-%M-%S %Z")
logging.info(
    f"################## {timestamp} RUNNING train.py ##################")

# Initialize W&B, if using
if args.use_wandb:
    wandb.init(project="protein-functions",
               name=f"baseline-model_{timestamp}", config=params)

# Log the configuration and arguments
logging.info(f"Configuration: {config}")
logging.info(f"Arguments: {args}")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load datasets from config file paths; the same vocabulary is used for all datasets
common_paths = {
    'amino_acid_vocabulary_path': os.path.join(ROOT_PATH, paths['AMINO_ACID_VOCAB_PATH']),
    'label_vocabulary_path': os.path.join(ROOT_PATH, paths['GO_LABEL_VOCAB_PATH']),
    'sequence_id_vocabulary_path': os.path.join(ROOT_PATH, paths['SEQUENCE_ID_VOCAB_PATH']),
    'sequence_id_map_path': os.path.join(ROOT_PATH, paths['SEQUENCE_ID_MAP_PATH']),
}
paths_list = [
    {
        **common_paths,
        'data_path': os.path.join(ROOT_PATH, data_path)
    }
    for data_path in [
        paths['TRAIN_DATA_PATH'],
        paths['VAL_DATA_PATH'],
        paths['TEST_DATA_PATH']
    ]
]
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
sequence_id_map = read_pickle(os.path.join(
    ROOT_PATH, paths['SEQUENCE_ID_MAP_PATH']))

# Load sequence embeddings
if not args.train_sequence_encoder:
    sequence_embedding_matrix = load_embeddings(
        os.path.join(ROOT_PATH, paths['SEQUENCE_EMBEDDING_PATH']),
        sequence_id_map,
        params['PROTEIN_EMBEDDING_DIM'],
        device
    )
    logging.info("Loaded sequence embeddings.")

# Load label embeddings
if not args.train_label_encoder:
    label_embedding_matrix = load_embeddings(
        os.path.join(ROOT_PATH, paths['LABEL_EMBEDDING_PATH']),
        train_dataset.label2int,
        params['LABEL_EMBEDDING_DIM'],
        device
    )
    logging.info("Loaded label embeddings.")

# Initialize the model
model = ProTCL(protein_embedding_dim=params['PROTEIN_EMBEDDING_DIM'],
               label_embedding_dim=params['LABEL_EMBEDDING_DIM'],
               latent_dim=params['LATENT_EMBEDDING_DIM'],
               temperature=params['TEMPERATURE'],
               sequence_embedding_matrix=sequence_embedding_matrix,
               train_label_encoder=args.train_label_encoder,
               label_embedding_matrix=label_embedding_matrix,
               train_sequence_encoder=args.train_sequence_encoder
               ).to(device)

# Log the number of parameters
total_params, trainable_params = count_parameters(model)
logging.info(f"Total Parameters: {total_params}")
logging.info(f"Trainable Parameters: {trainable_params}")

# Load the model weights if LOAD_MODEL_PATH is provided
if paths['STATE_DICT_PATH'] is not None:
    load_model_weights(model, os.path.join(
        ROOT_PATH, paths['STATE_DICT_PATH']))
    logging.info(
        f"Loading model weights from {paths['STATE_DICT_PATH']}...")

if args.mode in ['train', 'both']:
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['LEARNING_RATE'])

    # Log that training is starting
    logging.info("Starting training...")

    # Watch the model
    if args.use_wandb:
        wandb.watch(model)

    # Keep track of batches for logging
    batch_count = 0

    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')

    for epoch in range(params['NUM_EPOCHS']):
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
            if batch_count % params['VALIDATION_FREQUENCY'] == 0:

                ####### VALIDATION LOOP #######
                logging.info("Running validation...")
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        # Unpack the validation batch
                        sequence_ids, sequence_onehots, label_multihots, sequence_lengths = val_batch

                        # Move sequences and labels to GPU, if available
                        labels = label_multihots.to(device)
                        sequences = sequence_onehots.to(
                            device) if args.train_sequence_encoder else sequence_ids.to(device)

                        # Forward pass
                        P_e, L_e = model(sequences, labels)

                        # Compute target, if only using the labels that are present in the batch
                        target = labels[:, torch.any(labels, dim=0)]

                        # Compute validation loss for the batch
                        val_loss = contrastive_loss(
                            P_e, L_e, model.t, target)

                        # Accumulate the total validation loss
                        total_val_loss += val_loss.item()

                # Compute average validation loss
                avg_val_loss = total_val_loss / len(val_loader)
                logging.info(
                    f"Epoch {epoch+1}/{params['NUM_EPOCHS']}, Batch {batch_count}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}")

                if args.use_wandb:
                    wandb.log({"validation_loss": avg_val_loss})

                # Save the model if it has the best validation loss so far
                if avg_val_loss < best_val_loss:
                    logging.info(
                        f"New best validation loss: {avg_val_loss}. Saving model...")
                    best_val_loss = avg_val_loss
                    # Save model to OUTPUT_MODEL_DIR
                    model_path = os.path.join(
                        paths['OUTPUT_MODEL_DIR'], f"{timestamp}_best_ProTCL.pt")
                    torch.save(model.state_dict(), model_path)

                    if args.use_wandb:
                        wandb.save(f"{timestamp}_best_ProTCL.pt")

                # Set model back to training mode
                model.train()
else:
    logging.info("Skipping training...")

# label_normalizer = load_gz_json(paths['PARENTHOOD_LIB_PATH'])

####### TESTING LOOP #######
if args.mode in ['test', 'both']:

    logging.info("Starting testing...")
    model.eval()

    # Initialize metrics
    total_test_loss = 0

    eval_metrics = EvalMetrics(NUM_LABELS=train_dataset.label_vocabulary_size,
                               threshold=params['DECISION_TH'], average=params['METRICS_AVERAGE'], device=device)

    with torch.no_grad():
        for test_batch in test_loader:
            # Unpack the test batch
            sequence_ids, sequence_onehots, label_multihots, sequence_lengths = test_batch

            # Move sequences and labels to GPU, if available
            labels = label_multihots.to(device)
            sequences = sequence_onehots.to(
                device) if args.train_sequence_encoder else sequence_ids.to(device)

            # Forward pass
            P_e, L_e = model(sequences, labels)

            # Compute target, if only using the labels that are present in the batch
            target = labels[:, torch.any(labels, dim=0)]

            # Compute test loss for the batch
            test_loss = contrastive_loss(P_e, L_e, model.t, target)

            # Accumulate the total test loss
            total_test_loss += test_loss.item()

            # Compute cosine similarities for zero-shot classification
            logits = torch.mm(P_e, L_e.t()) * torch.exp(model.t)

            # Apply sigmoid to get the probabilities for multi-label classification
            probabilities = torch.sigmoid(logits)

            # TODO: Using original normalize_confidences implemented with numpy,
            # but this is slow. Should be able to do this with torch tensors.
            probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                                               label_vocab=train_dataset.label_vocabulary,
                                                               applicable_label_dict=label_normalizer), device=probabilities.device)
            eval_metrics(probabilities, labels)

    # Compute average test loss
    avg_test_loss = total_test_loss / len(test_loader)
    final_metrics = eval_metrics.compute()
    final_metrics.update({"test_loss": avg_test_loss})

    logging.info(json.dumps(final_metrics, indent=4))

    logging.info("Testing complete.")

# Close the W&B run
if args.use_wandb:
    wandb.log(final_metrics)
    wandb.finish()

# Clear GPU cache
torch.cuda.empty_cache()

logging.info("################## train.py COMPLETE ##################")
