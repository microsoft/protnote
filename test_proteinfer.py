from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.utils.main_utils import get_or_generate_vocabularies
from src.utils.configs import get_setup
from src.models.protein_encoders import ProteInfer
from src.utils.proteinfer import transfer_tf_weights_to_torch
from src.utils.evaluation import EvalMetrics
from torchmetrics.classification import F1Score
from src.utils.data import read_json, load_gz_json
from src.utils.proteinfer import normalize_confidences
import torch
import numpy as np
from tqdm import tqdm
import logging
import argparse
import os
import json

"""
sample usage: python test_proteinfer.py --validation-path-name VAL_DATA_PATH --test-paths-names TEST_DATA_PATH --decision-th 0.06
"""

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train and/or Test the ProTCL model.")

parser.add_argument(
    "--validation-path-name",
    type=str,
    required=True,
    help="Specify the desired val path name to validate the model during training using names from config file. If not provided, model will not be trained. If provided, must also provide --train-path.",
)

parser.add_argument(
    "--test-paths-names",
    nargs="+",
    type=str,
    required=True,
    help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.",
)

parser.add_argument(
    "--full-path-name",
    type=str,
    default=None,
    help="Specify the desired full path name to define the vocabularies. Defaults to the full path name in the config file.",
)

parser.add_argument(
    "--optimization-metric-name",
    type=str,
    default="f1_micro",
    help="Specify the desired metric to optimize for. Default is f1_micro.",
)

parser.add_argument(
    "--decision-th",
    type=float,
    default=None,
    help="Specify the desired decision threshold. If not provided, the decision threshold will be optimized on the validation set.",
)

parser.add_argument(
    "--name",
    type=str,
    default="ProteInfer",
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

parser.add_argument(
    "--normalize-probabilities",
    action="store_true",
    help="Normalize probabilities using the label vocabulary and applicable label dictionary."
)

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()

def to_device(device, *args):
    return [item.to(device) if isinstance(item,torch.Tensor) else None for item in args]

(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=args.override,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
)

# Load or generate the vocabularies
vocabularies = get_or_generate_vocabularies(
    paths[args.full_path_name], paths["VOCABULARIES_DIR"], logger)

# Create datasets
datasets = ProteinDataset.create_multiple_datasets(paths_list,
                                                   vocabularies=vocabularies)

# Initialize new run
logger.info(f"################## {timestamp} RUNNING train.py ##################")

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


model = ProteInfer.from_pretrained(
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

model.to(device)
model = model.eval()

label_normalizer = load_gz_json(paths["PARENTHOOD_LIB_PATH"])

# Initialize EvalMetrics
eval_metrics = EvalMetrics(device=device)

best_th = args.decision_th

if best_th is None:
    assert (
        args.validation_path_name is not None
    ), "Must provide validation path name to optimize decision threshold."
    val_probas = []
    val_labels = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(loaders["validation"][0]), total=len(loaders["validation"][0])
        ):
            sequence_onehots, sequence_embeddings, sequence_lengths, sequence_ids, label_multihots, tokenized_labels, label_embeddings = (
                batch["sequence_onehots"],
                batch["sequence_embeddings"],
                batch["sequence_lengths"],
                batch["sequence_ids"],
                batch["label_multihots"],
                batch["tokenized_labels"],
                batch["label_embeddings"]
            )

            if batch_idx==0:
                print(label_multihots.shape)

            sequence_onehots, sequence_lengths, label_multihots = to_device(device,
                sequence_onehots, sequence_lengths, label_multihots)

            logits = model(sequence_onehots, sequence_lengths)
            probabilities = torch.sigmoid(logits)

            if args.normalize_probabilities:
                probabilities = torch.tensor(
                    normalize_confidences(
                        predictions=probabilities.detach().cpu().numpy(),
                        label_vocab=vocabularies["GO_LABEL_VOCAB_PATH"],
                        applicable_label_dict=label_normalizer,
                    ),
                    device=probabilities.device,
                )

            val_probas.append(probabilities)
            val_labels.append(label_multihots)

    val_probas = torch.cat(val_probas)
    val_labels = torch.cat(val_labels)
    best_th = 0.0
    best_f1 = 0.0

    for th in np.arange(0.01, 1, 0.01):
        optimization_metric = eval_metrics\
            .get_metric_by_name(name=args.optimization_metric_name,
                                num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
                                threshold=th)

        optimization_metric(val_probas, val_labels)
        val_f1_score = optimization_metric.compute()
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
            best_th = th
        print("TH:", th, "F1:", val_f1_score)
    print("Best Val F1:", best_f1, "Best Val TH:", best_th)

val_metrics = eval_metrics\
            .get_metric_collection(type='all',
                                   threshold=best_th,
                                   num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"]
                                   )

with torch.no_grad():
    for batch_idx, batch in tqdm(
        enumerate(loaders["validation"][0]), total=len(loaders["validation"][0])
    ):
        sequence_onehots, sequence_embeddings, sequence_lengths, sequence_ids, label_multihots, tokenized_labels, label_embeddings = (
            batch["sequence_onehots"],
            batch["sequence_embeddings"],
            batch["sequence_lengths"],
            batch["sequence_ids"],
            batch["label_multihots"],
            batch["tokenized_labels"],
            batch["label_embeddings"]
        )

        sequence_onehots, sequence_lengths, label_multihots = to_device(device,
            sequence_onehots, sequence_lengths, label_multihots)

        logits = model(sequence_onehots, sequence_lengths)
        probabilities = torch.sigmoid(logits)

        if args.normalize_probabilities:
            probabilities = torch.tensor(
                normalize_confidences(
                    predictions=probabilities.detach().cpu().numpy(),
                    label_vocab=vocabularies["GO_LABEL_VOCAB_PATH"],
                    applicable_label_dict=label_normalizer,
                ),
                device=probabilities.device,
            )
        val_metrics(probabilities, label_multihots)

print(f"{args.optimization_metric_name} optimized val metrics")
print(val_metrics.compute())


test_metrics = eval_metrics\
            .get_metric_collection(type='all',
                                   threshold=best_th,
                                   num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"]
                                   )
all_labels = []
all_probas = []
all_seqids = []
with torch.no_grad():
    for batch_idx, batch in tqdm(
        enumerate(loaders["test"][0]), total=len(loaders["test"][0])
    ):

        sequence_onehots, sequence_lengths, sequence_ids, label_multihots = (
            batch["sequence_onehots"],
            batch["sequence_lengths"],
            batch["sequence_ids"],
            batch["label_multihots"]
        )

        sequence_onehots, sequence_lengths, label_multihots = to_device(device,
            sequence_onehots, sequence_lengths, label_multihots)
            
        logits = model(sequence_onehots, sequence_lengths)
        probabilities = torch.sigmoid(logits)
        if args.normalize_probabilities:
            probabilities = torch.tensor(
                normalize_confidences(
                    predictions=probabilities.detach().cpu().numpy(),
                    label_vocab=vocabularies["GO_LABEL_VOCAB_PATH"],
                    applicable_label_dict=label_normalizer,
                ),
                device=probabilities.device,
            )

        test_metrics(probabilities, label_multihots)

        all_labels.append(label_multihots)
        all_probas.append(probabilities)
        all_seqids.append(sequence_ids)

    print(f"{args.optimization_metric_name} optimized test metrics")
    print(test_metrics.compute())

    all_labels = torch.cat(all_labels)
    all_probas = torch.cat(all_probas)
    all_seqids = torch.cat(all_seqids)

    # np.save(PROTEINFER_RESULTS_DIR+'labels.npy',all_labels.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'probas.npy',all_probas.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'seqids.npy',all_seqids.detach().cpu().numpy())

torch.cuda.empty_cache()
