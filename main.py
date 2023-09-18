import logging
from src.utils.data import (
    load_model_weights,
    seed_everything,
    create_ordered_tensor,
    get_tokenized_labels_dataloader,
)
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.evaluation import EvalMetrics, save_evaluation_results
from src.utils.models import count_parameters_by_layer
from src.utils.configs import get_setup
from src.utils.proteinfer import transfer_tf_weights_to_torch
import torch
import wandb
import os
import datetime
import argparse
import json


"""
 sample usage: python train.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH --test-paths-names TEST_DATA_PATH TEST_DATA_PATH 
 here we pass the same test set twice as an example.
"""

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train and/or Test the ProTCL model.")
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
    "--override", nargs="*", help="Override parameters in key-value pairs."
)

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

# TODO: Need to
# TODO: This could be more elegant with parser.add_subparsers()
# Raise error if only one of train or val path is provided
if (args.train_path_name is None) ^ (args.validation_path_name is None):
    parser.error(
        "You must provide both --train-path-name and --val-path-name, or neither."
    )

# Raise error if none of the paths are provided
if (
    (args.train_path_name is None)
    and (args.validation_path_name is None)
    and (args.test_paths_names is None)
):
    parser.error(
        "You must provide --test-path-names, or --train-path-name, --val-path-name together together, or all three."
    )

# Raise error if only test path is provided and no model is loaded
if (
    (args.train_path_name is None)
    and (args.validation_path_name is None)
    and (args.test_paths_names is not None)
    and (args.load_model is None)
):
    parser.error("You must provide --load-model if you only provide --test-path-names.")

(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=args.override,
    train_path_name=args.train_path_name,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
)

# Create datasets
datasets = ProteinDataset.create_multiple_datasets(paths_list)

# Initialize new run
logger.info(f"################## {timestamp} RUNNING train.py ##################")

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
loaders = create_multiple_loaders(
    datasets=datasets,
    params=params,
    num_workers=params["NUM_WORKERS"],
    pin_memory=True,
)

# mappings. A bit hacky, but it works
sequence_id2int = datasets[list(datasets.keys())[0]][0].sequence_id2int
label2int = datasets[list(datasets.keys())[0]][0].label2int
label_vocabulary = datasets[list(datasets.keys())[0]][0].label_vocabulary


# Load sequence embeddings
sequence_embedding_matrix = None
if not params["TRAIN_SEQUENCE_ENCODER"]:
    # TODO: Rather than loading from file, create from ProteInfer itself (slower at startup, but more flexible)
    sequence_embedding_matrix = create_ordered_tensor(
        paths["SEQUENCE_EMBEDDING_PATH"],
        sequence_id2int,
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
        label2int,
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
        label_vocabulary=sequence_id2int,
        label2int_mapping=label2int,
        batch_size=params["LABEL_BATCH_SIZE"],
        device=device,
    )

# Seed everything so we don't go crazy
seed_everything(params["SEED"], device)
# Initialize the models


# Initialize 
sequence_encoder = ProteInfer.from_pretrained(
    weights_path=config["relative_paths"]["PROTEINFER_WEIGHTS_PATH"],
    num_labels=params["NUM_LABELS"],
    input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
    output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
    kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
    activation=torch.nn.ReLU,
    dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
    num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
    bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
)


# TODO: Initialize ProteInfer and PubMedBERT here as well as the ensemble (ProTCL), which should take the other two as optional arguments

model = ProTCL(
    protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
    label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
    latent_dim=params["LATENT_EMBEDDING_DIM"],
    temperature=params["TEMPERATURE"],
    label_encoder=label_encoder,
    tokenized_labels_dataloader=tokenized_labels_dataloader,
    sequence_encoder=sequence_encoder,
    label_embedding_matrix=label_embedding_matrix,
    train_projection_head=params["TRAIN_PROJECTION_HEAD"],
    train_label_embeddings=params["TRAIN_LABEL_EMBEDDING_MATRIX"],
    train_label_encoder=params["TRAIN_LABEL_ENCODER"]
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
if args.train_path_name is not None:
    # Train function
    Trainer.train(train_loader=loaders["train"][0], val_loader=loaders["validation"][0])
else:
    logger.info("Skipping training...")

####### TESTING LOOP #######
all_test_results = []
all_test_metrics = []
if args.test_paths_names is not None:
    logger.info("Starting testing...")
    best_val_th = params["DECISION_TH"]

    # If no decision threshold is provided, find the optimal threshold on the validation set
    if params["DECISION_TH"] is None:
        if args.validation_path_name is not None:
            logger.info("Decision threshold not provided.")

            best_val_th, best_val_score = Trainer.find_optimal_threshold(
                data_loader=loaders["validation"][0],
                optimization_metric_name="f1_micro",
            )
        else:
            raise ValueError(
                "Decision threshold not provided and no validation set provided to find optimal."
            )

    for idx, test_loader in enumerate(loaders["test"]):
        logger.info(f"====Testing on test set #{idx}====")
        # Evaluate model on test set
        eval_metrics = EvalMetrics(
            num_labels=params["NUM_LABELS"], threshold=best_val_th, device=device
        ).get_metric_collection(type="all")

        final_metrics, test_results = Trainer.evaluate(
            data_loader=test_loader, eval_metrics=eval_metrics, testing=True
        )

        save_evaluation_results(results=test_results,
                                   label_vocabulary = label_vocabulary,
                                   run_name = args.name,
                                   output_dir=os.path.join(ROOT_PATH,paths["RESULTS_DIR"])
                                   )


        # Convert all metrics to float
        final_metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in final_metrics.items()
        }

        all_test_results.append(test_results)
        all_test_metrics.append(final_metrics)

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
