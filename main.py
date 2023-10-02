import logging
from src.utils.data import (
    load_model_weights,
    seed_everything,
)
from src.utils.main_utils import get_or_generate_vocabularies,  get_or_generate_label_embeddings, get_or_generate_sequence_embeddings
from src.data.datasets import ProteinDataset, calculate_pos_weight, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.evaluation import EvalMetrics, save_evaluation_results
from src.utils.models import count_parameters_by_layer
from src.utils.configs import get_setup
from torch.utils.data import DataLoader
import torch
import wandb
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from src.data.collators import collate_variable_sequence_length

"""
 Sample usage: python main.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH --test-paths-names TEST_DATA_PATH TEST_DATA_PATH 
 here we pass the same test set twice as an example.
"""

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Train and/or Test the ProTCL model.")
parser.add_argument(
    "--train-path-name",
    type=str,
    default=None,
    help="Specify the desired train path name to train the model using names from config file. If not provided, model will not be trained. If provided, must also provide --val-path.",
)

parser.add_argument(
    "--validation-path-name",
    type=str,
    default=None,
    help="Specify the desired val path name to validate the model during training using names from config file. If not provided, model will not be trained. If provided, must also provide --train-path.",
)

parser.add_argument(
    "--full-path-name",
    type=str,
    default=None,
    help="Specify the desired full path name to define the vocabularies. Defaults to the full path name in the config file.",
)

parser.add_argument(
    "--test-paths-names",
    nargs="+",
    type=str,
    default=None,
    help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.",
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
    "--override", nargs="*", help="Override config parameters in key-value pairs."
)

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

# TODO: This could be more elegant with parser.add_subparsers()
# Ensure the full data path is provided
if args.full_path_name is None:
    parser.error(
        "You must provide the full path name to define the vocabularies using --full-path-name."
    )

# Raise error if only one of train or val path is provided

if (args.train_path_name is not None) & (args.validation_path_name is None):
    parser.error(
        "If providing --train-path-name you must provide --val-path-name."
    )

# Raise error if none of the paths are provided
if args.test_paths_names is None and \
   (args.train_path_name is None or args.validation_path_name is None):
    parser.error("You must provide one of the following options:\n"
                 "--test-path-names\n"
                 "--train-path-name and --validation-path-name together\n"
                 "All three options\nPlease provide the required option(s) and try again.")

# Raise error if no train path is provided and no model is loaded
if (
    (args.train_path_name is None)
    and (args.load_model is None)
):
    parser.error(
        "You must provide --load-model if no --train-path-names is provided")


(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=args.override,
    train_path_name=args.train_path_name,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
)

# Initialize W&B, if using
if args.use_wandb:
    wandb.init(
        project="protein-functions",
        name=f"{args.name}_{timestamp}",
        config={**params, **vars(args)},
        entity="microsoft-research-incubation"
    )
    # Log the wandb link
    logger.info(f"W&B link: {wandb.run.get_url()}")

# Log the params
logger.info(json.dumps(params, indent=4))

# Initialize label tokenizer
label_tokenizer = AutoTokenizer.from_pretrained(
    params['LABEL_ENCODER_CHECKPOINT'])

# Load or generate the vocabularies
vocabularies = get_or_generate_vocabularies(
    paths[args.full_path_name], paths["VOCABULARIES_DIR"], logger)

# Create datasets
datasets = ProteinDataset.create_multiple_datasets(
    paths_list,
    label_tokenizer=label_tokenizer,
    vocabularies=vocabularies,
    subset_fractions={
        "train": params["TRAIN_SUBSET_FRACTION"],
        "validation": params["VALIDATION_SUBSET_FRACTION"],
        "test": params["TEST_SUBSET_FRACTION"]}
)

# Seed everything so we don't go crazy
seed_everything(params["SEED"], device)

# Initialize new run
logger.info(
    f"################## {timestamp} RUNNING main.py ##################")

# Define label sample sizes for train, validation, and test loaders
label_sample_sizes = {
    "train": params["TRAIN_LABEL_SAMPLE_SIZE"],
    "validation": params["VALIDATION_LABEL_SAMPLE_SIZE"],
    "test": None  # No sampling for the test set
}

# Define data loaders
loaders = create_multiple_loaders(
    datasets,
    params,
    label_sample_sizes=label_sample_sizes,
    num_workers=params["NUM_WORKERS"],
)

# Mappings. A bit hacky, but it works
# TODO: Is there a cleaner way of doing this? It feels awkward that we define "label2int" based on only train (even though it's the same)
# Maybe we load label2int from the JSON vocabulary upfront and pass that to the dataset?
label2int = datasets[list(datasets.keys())[0]][0].label2int
int2label = datasets[list(datasets.keys())[0]][0].int2label
label_annotation_map = datasets[list(datasets.keys())[
    0]][0].label_annotation_map

# Load label encoder
label_encoder = AutoModel.from_pretrained(params['LABEL_ENCODER_CHECKPOINT'])

# Generate all label embeddings upfront, if not training the label encoder
label_embedding_matrix = None
if not params["TRAIN_LABEL_ENCODER"]:
    # Create a list of text labels
    sorted_labels = sorted(
        vocabularies["GO_label_vocab"], key=lambda x: label2int[x])
    label_annotations = [label_annotation_map[label_id]
                         for label_id in sorted_labels]
    label_encoder = label_encoder.to(device)
    label_embedding_matrix = get_or_generate_label_embeddings(
        paths,
        device,
        label_annotations,
        label_tokenizer,
        label_encoder,
        logger,
        label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT"]
    )

    # Move the label encoder to CPU
    label_encoder = label_encoder.cpu()

# Initialize ProteInfer
sequence_encoder = ProteInfer.from_pretrained(
    weights_path=paths["PROTEINFER_WEIGHTS_PATH"],
    num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
    input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
    output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
    kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
    activation=torch.nn.ReLU,
    dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
    num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
    bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
)

# Generate all sequence embeddings upfront, if not training the sequence encoder
sequence_embedding_dict = None
if not params["TRAIN_SEQUENCE_ENCODER"]:
    sequence_embedding_dict = get_or_generate_sequence_embeddings(
        paths,
        device,
        sequence_encoder,
        datasets,
        params,
        logger,
    )

# Loop through all the datasets and set the label and sequence embedding matrices
for dataset in datasets.values():
    for subset in dataset:
        if not params["TRAIN_LABEL_ENCODER"]:
            subset.set_label_embedding_matrix(label_embedding_matrix.cpu())
        if not params["TRAIN_SEQUENCE_ENCODER"]:
            subset.set_sequence_embedding_dict(sequence_embedding_dict)

model = ProTCL(
    # Parameters
    protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
    label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
    latent_dim=params["LATENT_EMBEDDING_DIM"],

    # Encoders
    label_encoder=label_encoder,
    sequence_encoder=sequence_encoder,

    # Output Layer
    output_dim=params["OUTPUT_DIM"],
    output_num_layers=params["OUTPUT_NUM_LAYERS"],

    # Training options
    train_label_encoder=params["TRAIN_LABEL_ENCODER"],
    train_sequence_encoder=params["TRAIN_SEQUENCE_ENCODER"],
).to(device)

# Calculate pos_weight based on the training set
if (params["POS_WEIGHT"] is None) & (args.train_path_name is not None):
    logger.info("Calculating pos_weight...")
    pos_weight = calculate_pos_weight(datasets["train"][0].data,
                                      datasets["train"][0].get_label_vocabulary_size()
                                      ).to(device)
    logger.info(f"Calculated pos_weight= {pos_weight.item()}")
elif (params["POS_WEIGHT"] is not None):
    pos_weight = torch.tensor(params["POS_WEIGHT"]).to(device)
else:
    raise ValueError(
        "POS_WEIGHT is not provided and no training set is provided to calculate it.")

# Initialize trainer class to handle model training, validation, and testing
Trainer = ProTCLTrainer(
    model=model,
    device=device,
    config={"params": params, "paths": paths},
    vocabularies=vocabularies,
    logger=logger,
    timestamp=timestamp,
    run_name=args.name,
    use_wandb=args.use_wandb,
    pos_weight=pos_weight
)

# Log the number of parameters by layer
count_parameters_by_layer(model)

# Load the model weights if --load-model argument is provided
if args.load_model:
    load_model_weights(model, os.path.join(ROOT_PATH, args.load_model))
    logger.info(f"Loading model weights from {args.load_model}...")

####### TRAINING AND VALIDATION LOOPS #######
if args.train_path_name is not None:
    # Train function
    Trainer.train(train_loader=loaders["train"][0],
                  val_loader=loaders["validation"][0])
else:
    logger.info("Skipping training...")

####### TESTING LOOP #######
all_test_results = []
all_test_metrics = []

if not params["DECISION_TH"] and not args.validation_path_name:
    raise ValueError(
        "DECISION_TH is not provided and no validation set is provided to calculate it.")

best_val_th = params["DECISION_TH"]
# Setup for validation
if args.validation_path_name:
    val_loader = DataLoader(
        datasets["validation"][0],
        batch_size=params["TEST_BATCH_SIZE"],
        shuffle=False,
        collate_fn=collate_variable_sequence_length,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )

    # If no decision threshold is provided, find the optimal threshold on the validation set
    if not params["DECISION_TH"]:
        logger.info("Decision threshold not provided.")
        best_val_th, _ = Trainer.find_optimal_threshold(
            data_loader=val_loader,
            optimization_metric_name="f1_micro",
        )

    logger.info("====Testing on validation set====")
    eval_metrics = EvalMetrics(
        num_labels=len(vocabularies["GO_label_vocab"]), threshold=best_val_th, device=device
    ).get_metric_collection(type="all")
    validation_metrics, validation_results = Trainer.evaluate(
        data_loader=val_loader, eval_metrics=eval_metrics
    )

    # save_evaluation_results(results=validation_results,
    #                         label_vocabulary=label_vocabulary,
    #                         run_name=args.name,
    #                         output_dir=os.path.join(
    #                             ROOT_PATH, paths["RESULTS_DIR"])
    #                         )

    logger.info(json.dumps(validation_metrics, indent=4))
    logger.info("Final validation complete.")

# Setup for testing
if args.test_paths_names:
    logger.info("Starting testing...")
    for idx, test_loader in enumerate(loaders["test"]):
        logger.info(f"====Testing on test set #{idx}====")
        # If best_val_th is not defined, alert an error to either provide a decision threshold or a validation datapath
        eval_metrics = EvalMetrics(
            num_labels=len(vocabularies["GO_label_vocab"]), threshold=best_val_th, device=device
        ).get_metric_collection(type="all")
        test_metrics, test_results = Trainer.evaluate(
            data_loader=test_loader, eval_metrics=eval_metrics
        )

        all_test_results.append(test_results)
        all_test_metrics.append(test_metrics)
        logger.info(json.dumps(test_metrics, indent=4))
        logger.info("Testing complete.")

# Close the W&B run
if args.use_wandb and all_test_metrics:
    wandb.log(all_test_metrics[-1])
    wandb.finish()

# Clear GPU cache
torch.cuda.empty_cache()

logger.info("################## train.py COMPLETE ##################")
