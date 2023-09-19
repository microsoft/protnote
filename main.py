import logging
from src.utils.data import (
    load_model_weights,
    seed_everything,
    create_ordered_tensor,
    read_pickle,
)
from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.evaluation import EvalMetrics, save_evaluation_results
from src.utils.models import count_parameters_by_layer, get_embeddings
from src.utils.configs import get_setup
import torch
import wandb
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModel

"""
 Sample usage: python main.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH --test-paths-names TEST_DATA_PATH TEST_DATA_PATH 
 here we pass the same test set twice as an example.
"""

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
parser.add_argument(
    "--log-to-console",
    action="store_true",
    default=False,
    help="Log outputs to console instead of the default logfile.",
)

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

# TODO: This could be more elegant with parser.add_subparsers()
# Raise error if only one of train or val path is provided
if (args.train_path_name is None) ^ (args.validation_path_name is None):
    parser.error(
        "You must provide both --train-path-name and --val-path-name, or neither."
    )

# Raise error if none of the paths are provided
if args.test_paths_names is None and \
   (args.train_path_name is None or args.validation_path_name is None):
    parser.error("You must provide one of the following options:\n"
                 "--test-path-names\n"
                 "--train-path-name and --validation-path-name together\n"
                 "All three options\nPlease provide the required option(s) and try again.")

# Raise error if only test path is provided and no model is loaded
if (
    (args.train_path_name is None)
    and (args.validation_path_name is None)
    and (args.test_paths_names is not None)
    and (args.load_model is None)
):
    parser.error(
        "You must provide --load-model if you only provide --test-path-names.")

(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=args.override,
    log_to_console=args.log_to_console,
    train_path_name=args.train_path_name,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
)

# Create datasets
datasets = ProteinDataset.create_multiple_datasets(paths_list)

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
loaders = create_multiple_loaders(
    datasets=datasets,
    params=params,
    num_workers=params["NUM_WORKERS"],
    pin_memory=True,
)

# mappings. A bit hacky, but it works
# TODO: Is there a cleaner way of doing this? It feels awkward that we define "label2int" based on only train (even though it's the same)
# Maybe we load label2int from the JSON vocabulary upfront and pass that to the dataset?
label2int = datasets[list(datasets.keys())[0]][0].label2int
int2label = datasets[list(datasets.keys())[0]][0].int2label
label_vocabulary = datasets[list(datasets.keys())[0]][0].label_vocabulary

# TODO: Get rid of this one (sequence ids should be dynamically loaded in the future; no caching)
sequence_id2int = datasets[list(datasets.keys())[0]][0].sequence_id2int

# Create a map from GO alphanumeric id to numberic id
int_label_id_map = {k: int(v[3:]) for k, v in int2label.items()}

# Load the map from numeric label id to text label
label_annotation_map = {key: value['label'] for key, value in read_pickle(
    paths['GO_ANNOTATIONS_PATH']).to_dict(orient='index').items()}

# Load the tokenizer and model
label_tokenizer = AutoTokenizer.from_pretrained(
    params['LABEL_ENCODER_CHECKPOINT'])
label_encoder = AutoModel.from_pretrained("microsoft/biogpt").to(device)

# Generate all label embeddings upfront, if not training the encoder
label_embedding_matrix = None
if not params["TRAIN_LABEL_ENCODER"]:
    # Create a list of text labels
    label_annotations = [label_annotation_map[int(label_id[3:])]
                         for label_id in label_vocabulary]

    # Tokenize the labels in batches
    logger.info("Tokenizing all labels and getting embeddings...")
    with torch.no_grad():
        label_embedding_matrix = get_embeddings(
            label_annotations, label_tokenizer, label_encoder, batch_size=params[
                "LABEL_BATCH_SIZE_LIMIT"]
        )

# Seed everything so we don't go crazy
seed_everything(params["SEED"], device)

# Initialize ProteInfer
sequence_encoder = ProteInfer.from_pretrained(
    weights_path=paths["PROTEINFER_WEIGHTS_PATH"],
    num_labels=params["NUM_LABELS"],
    input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
    output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
    kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
    activation=torch.nn.ReLU,
    dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
    num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
    bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
)

model = ProTCL(
    # Parameters
    protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
    label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
    latent_dim=params["LATENT_EMBEDDING_DIM"],
    temperature=params["TEMPERATURE"],
    # Label encoder
    label_encoder=label_encoder,
    label_tokenizer=label_tokenizer,
    label_annotation_map=label_annotation_map,
    int_label_id_map=int_label_id_map,
    # Sequence encoder
    sequence_encoder=sequence_encoder,
    # Training options
    train_projection_head=params["TRAIN_PROJECTION_HEAD"],
    train_label_encoder=params["TRAIN_LABEL_ENCODER"],
    train_sequence_encoder=params["TRAIN_SEQUENCE_ENCODER"],
    label_embedding_matrix=label_embedding_matrix,
).to(device)

# Initialize trainer class to handle model training, validation, and testing
Trainer = ProTCLTrainer(
    model=model,
    device=device,
    config={"params": params, "paths": paths},
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
    Trainer.train(train_loader=loaders["train"]
                  [0], val_loader=loaders["validation"][0])
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
                                label_vocabulary=label_vocabulary,
                                run_name=args.name,
                                output_dir=os.path.join(
                                    ROOT_PATH, paths["RESULTS_DIR"])
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
