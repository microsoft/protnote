import logging
from src.utils.data import read_json, load_embeddings, load_model_weights
from src.utils.losses import contrastive_loss
from src.data.collators import collate_variable_sequence_length
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCL import ProTCL
from src.utils.proteinfer import normalize_confidences
from src.utils.data import read_json,load_gz_json
from src.utils.evaluation import EvalMetrics
import torch
import wandb
import os
import datetime
from torchmetrics.classification import BinaryPrecision, BinaryRecall
import argparse
import json

# Argument parser setup
parser = argparse.ArgumentParser(description="Train the ProTCL model.")
parser.add_argument('--train', action='store_true',
                    default=True, help="Perform a training pass. Default is true.")
parser.add_argument('--train-label-encoder', action='store_true',
                    default=False, help="Train the label encoder. Default is False.")
parser.add_argument('--train-sequence-encoder', action='store_true',
                    default=False, help="Train the sequence encoder. Default is False.")
parser.add_argument('--test-model', action='store_true',
                    default=True, help="Perform a test pass. Default is True.")
parser.add_argument('--use-wandb', action='store_true', default=False,
                    help="Use Weights & Biases for logging. Default is False.")
args = parser.parse_args()

# Get the root path from the environment variable
# Default to current directory if ROOT_PATH is not set
ROOT_PATH = os.environ.get('ROOT_PATH', '.')

# Load the configuration file
config = read_json(os.path.join(ROOT_PATH, 'config.json'))
params = config['params']
paths = config['relative_paths']

# Initialize logging
logging.basicConfig(filename='train.log', filemode='w',
                    format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S %Z')

# Initialize new run
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z")
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
train_dataset, val_dataset, test_dataset = ProteinDataset\
    .create_multiple_datasets(data_paths=[paths['TRAIN_DATA_PATH'], paths['VAL_DATA_PATH'], paths['TEST_DATA_PATH']],
                              sequence_vocabulary_path=paths['AMINO_ACID_VOCAB_PATH'],
                              label_vocabulary_path=paths['GO_LABEL_VOCAB_PATH'],
                              allowed_annotations_path=paths['ALLOWED_ANNOTATIONS_PATH'],
                              train_sequence_encoder=args.train_sequence_encoder)

# Define data loaders
train_loader, val_loader, test_loader = create_multiple_loaders(
    [train_dataset, val_dataset, test_dataset],
    [params['TRAIN_BATCH_SIZE'], params['VALIDATION_BATCH_SIZE'],
        params['TEST_BATCH_SIZE']],
    num_workers=params['NUM_WORKERS'],
    train_sequence_encoder=args.train_sequence_encoder,
)

# Load embeddings
if not args.train_sequence_encoder:
    sequence_to_embeddings_dict = load_embeddings(
        paths['SEQUENCE_EMBEDDING_PATH'])
    logging.info("Loaded sequence embeddings.")
if not args.train_label_encoder:
    label_embeddings = load_embeddings(paths['LABEL_EMBEDDING_PATH'])
    logging.info("Loaded label embeddings.")

# Move embeddings to GPU
label_embeddings = label_embeddings.to(device)
sequence_to_embeddings_dict = {seq: embedding.to(
    device) for seq, embedding in sequence_to_embeddings_dict.items()}

# Initialize the model
model = ProTCL(protein_embedding_dim=params['PROTEIN_EMBEDDING_DIM'],
               label_embedding_dim=params['LABEL_EMBEDDING_DIM'],
               latent_dim=params['LATENT_EMBEDDING_DIM'],
               temperature=params['TEMPERATURE'],
               sequence_to_embeddings_dict=sequence_to_embeddings_dict,
               ordered_label_embeddings=label_embeddings).to(device)

# Load the model weights if LOAD_MODEL_PATH is provided
if paths['STATE_DICT_PATH'] is not None:
    load_model_weights(model, paths['STATE_DICT_PATH'])
    logging.info(
        f"Loading model weights from {paths['STATE_DICT_PATH']}...")

if args.train:
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

    # Define which loader to use
    loader = train_loader

    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')

    for epoch in range(params['NUM_EPOCHS']):

        ####### TRAINING LOOP #######
        for train_batch in loader:
            # Unpack batch
            sequences, sequence_lengths, labels, target = train_batch

            # Move sequences and labels to GPU, if available
            # TODO: Use sequence ID's so we can move to GPU
            # sequences = sequences.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            P_e, L_e = model(sequences, labels)

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
                        sequences, sequence_lengths, labels, target = val_batch

                        # Move sequences and labels to GPU, if available
                        # sequences = sequences.to(device)
                        labels = labels.to(device)

                        # Forward pass
                        P_e, L_e = model(sequences, labels)

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
                        paths['OUTPUT_MODEL_DIR'], f"{timestamp}_best_ProTCL.pth")
                    torch.save(model.state_dict(), model_path)

                    if args.use_wandb:
                        wandb.save(f"{timestamp}_best_ProTCL.pth")

                # Set model back to training mode
                model.train()
else:
    logging.info("Skipping training...")


label_normalizer = load_gz_json(paths['PARENTHOOD_LIB_PATH'])

####### TESTING LOOP #######
if args.test_model:

    logging.info("Starting testing...")
    model.eval()

    # Initialize metrics
    total_test_loss = 0

    eval_metrics = EvalMetrics(NUM_LABELS=train_dataset.label_vocabulary_size,threshold=params['DECISION_TH'],average=params['METRICS_AVERAGE'],device=device)

    with torch.no_grad():
        for test_batch in test_loader:
            # Unpack the test batch
            sequences, sequence_lengths, labels, target = test_batch

            # Move sequences and labels to GPU, if available
            # sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            P_e, L_e = model(sequences, labels)

            # Compute test loss for the batch
            test_loss = contrastive_loss(P_e, L_e, model.t, target)

            # Accumulate the total test loss
            total_test_loss += test_loss.item()

            # Compute cosine similarities for zero-shot classification
            logits = torch.mm(P_e, L_e.t()) * torch.exp(model.t)

            # Apply sigmoid to get the probabilities for multi-label classification
            probabilities = torch.sigmoid(logits)
            
            #TODO: Using original normalize_confidences implemented with numpy,
            #but this is slow. Should be able to do this with torch tensors.
            probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                label_vocab=train_dataset.label_vocabulary,
                                applicable_label_dict=label_normalizer),device=probabilities.device)
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
