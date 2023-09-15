from src.data.datasets import ProteinDataset, create_multiple_loaders
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
    "--optimization-metric-name",
    type=str,
    default="f1_macro",
    help="Specify the desired metric to optimize for. Default is f1_macro.",
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

# TODO: Add an option to serialize and save config with a name corresponding to the model save path

# TODO: Make Optimization metric and normalize probabilities part of arguments
args = parser.parse_args()


(config, params, paths, paths_list, timestamp, logger, device, ROOT_PATH) = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=args.override,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
)

# Create datasets
datasets = ProteinDataset.create_multiple_datasets(paths_list)

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


model = ProteInfer(
    num_labels=params["NUM_LABELS"],
    input_channels=20,
    output_channels=1100,
    kernel_size=9,
    activation=torch.nn.ReLU,
    dilation_base=3,
    num_resnet_blocks=5,
    bottleneck_factor=0.5,
)

transfer_tf_weights_to_torch(model, paths["PROTEINFER_WEIGHTS_PATH"])
model.to(device)
model = model.eval()

vocab = read_json(paths["GO_LABEL_VOCAB_PATH"])
label_normalizer = load_gz_json(paths["PARENTHOOD_LIB_PATH"])

best_th = args.decision_th
if best_th is None:
    assert (
        args.validation_path_name is not None
    ), "Must provide validation path name to optimize decision threshold."
    val_probas = []
    val_labels = []

    with torch.no_grad():
        for batch_idx, (sequence_ids, sequences, labels, sequence_lengths) in tqdm(
            enumerate(loaders["validation"][0]), total=len(loaders["validation"][0])
        ):
            sequence_ids, sequences, labels, sequence_lengths = (
                sequence_ids.to(device),
                sequences.to(device),
                labels.to(device),
                sequence_lengths.to(device),
            )

            logits = model(sequences, sequence_lengths)
            probabilities = torch.sigmoid(logits)
            probabilities = torch.tensor(
                normalize_confidences(
                    predictions=probabilities.detach().cpu().numpy(),
                    label_vocab=vocab,
                    applicable_label_dict=label_normalizer,
                ),
                device=probabilities.device,
            )

            val_probas.append(probabilities)
            val_labels.append(labels)

    val_probas = torch.cat(val_probas)
    val_labels = torch.cat(val_labels)
    best_th = 0.0
    best_f1 = 0.0

    for th in np.arange(0.01, 1, 0.01):
        eval_metrics = EvalMetrics(
            num_labels=params["NUM_LABELS"], threshold=th, device=device
        ).get_metric_collection(type="all")

        optimization_metric = getattr(eval_metrics, args.optimization_metric_name)

        optimization_metric(val_probas, val_labels)
        val_f1_score = optimization_metric.compute()
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
            best_th = th
        print("TH:", th, "F1:", val_f1_score)
    print("Best Val F1:", best_f1, "Best Val TH:", best_th)

eval_metrics = EvalMetrics(
    num_labels=params["NUM_LABELS"], threshold=best_th, device=device
).get_metric_collection(type="all")

all_labels = []
all_probas = []
all_seqids = []
with torch.no_grad():
    for batch_idx, (sequence_ids, sequences, labels, sequence_lengths) in tqdm(
        enumerate(loaders["test"][0]), total=len(loaders["test"][0])
    ):
        sequence_ids, sequences, labels, sequence_lengths = (
            sequence_ids.to(device),
            sequences.to(device),
            labels.to(device),
            sequence_lengths.to(device),
        )

        logits = model(sequences, sequence_lengths)
        probabilities = torch.sigmoid(logits)
        probabilities = torch.tensor(
            normalize_confidences(
                predictions=probabilities.detach().cpu().numpy(),
                label_vocab=vocab,
                applicable_label_dict=label_normalizer,
            ),
            device=probabilities.device,
        )

        eval_metrics(probabilities, labels)

        all_labels.append(labels)
        all_probas.append(probabilities)
        all_seqids.append(sequence_ids)

    final_metrics = eval_metrics.compute()
    print("Final Metrics:", final_metrics)

    all_labels = torch.cat(all_labels)
    all_probas = torch.cat(all_probas)
    all_seqids = torch.cat(all_seqids)

    # np.save(PROTEINFER_RESULTS_DIR+'labels.npy',all_labels.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'probas.npy',all_probas.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'seqids.npy',all_seqids.detach().cpu().numpy())

torch.cuda.empty_cache()
