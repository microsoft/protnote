from src.data.datasets import ProteinDataset, create_multiple_loaders
from src.utils.configs import get_setup
from src.models.protein_encoders import ProteInfer
from src.utils.evaluation import EvalMetrics, save_evaluation_results
from src.utils.data import read_json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
from src.utils.losses import FocalLoss
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
from torch.cuda.amp import autocast
from src.utils.data import generate_vocabularies


"""
sample usage: python test_proteinfer.py --validation-path-name VAL_DATA_PATH --full-path-name FULL_DATA_PATH
"""

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Argument parser setup
parser = argparse.ArgumentParser(description="Train and/or Test the ProtNote model.")


parser.add_argument("--train-path-name", type=str)

parser.add_argument("--validation-path-name", type=str)

parser.add_argument(
    "--test-paths-names",
    nargs="+",
    type=str,
    help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.",
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

parser.add_argument("--threshold", type=float, default=0.5)

parser.add_argument(
    "--proteinfer-weights",
    type=str,
    default="GO",
    help="Which model weights to use: GO or EC",
)
parser.add_argument(
    "--model-weights-id",
    type=int,
    default=None,
    help="The model id if any. If not specified will pick the same one always.",
)
parser.add_argument(
    "--override", nargs="*", help="Override config parameters in key-value pairs."
)

parser.add_argument(
    "--save-prediction-results",
    action="store_true",
    default=False,
    help="Save predictions and ground truth dataframe for validation and/or test",
)

parser.add_argument(
    "--only-inference",
    action="store_true",
    default=False,
    help="Whether to only predict without testing and computing metrics",
)

parser.add_argument(
    "--only-represented-labels",
    action="store_true",
    default=False,
    help="Whether to only predict labels that are represented in the dataset",
)

parser.add_argument(
    "--annotations-path-name",
    type=str,
    default="GO_ANNOTATIONS_PATH",
    help="Name of the annotation path. Defaults to GO.",
)

parser.add_argument(
    "--base-label-embedding-name",
    type=str,
    default="GO_BASE_LABEL_EMBEDDING_PATH",
    help="Name of the base label embedding path. Defaults to GO.",
)


# TODO: Add an option to serialize and save config with a name corresponding to the model save path

args = parser.parse_args()


def to_device(device, *args):
    return [
        item.to(device) if isinstance(item, torch.Tensor) else None for item in args
    ]


if args.override:
    args.override += ["WEIGHTED_SAMPLING", "False", "TEST_BATCH_SIZE", 4]
else:
    args.override = ["WEIGHTED_SAMPLING", "False", "TEST_BATCH_SIZE", 4]


task = args.annotations_path_name.split("_")[0]
config = get_setup(
    config_path=args.config,
    run_name=args.name,
    train_path_name=args.train_path_name,
    val_path_name=args.validation_path_name,
    test_paths_names=args.test_paths_names,
    annotations_path_name=args.annotations_path_name,
    base_label_embedding_name=args.base_label_embedding_name,
    amlt=False,
    is_master=True,
    overrides=args.override + ["EXTRACT_VOCABULARIES_FROM", "null"]
    if args.only_represented_labels
    else args.override,
)
params, paths, timestamp, logger = (
    config["params"],
    config["paths"],
    config["timestamp"],
    config["logger"],
)


# Create datasets
train_dataset = (
    ProteinDataset(
        data_paths=config["dataset_paths"]["train"][0],
        config=config,
        logger=logger,
        require_label_idxs=params["GRID_SAMPLER"],
        label_tokenizer=None,
    )
    if args.train_path_name is not None
    else None
)

validation_dataset = (
    ProteinDataset(
        data_paths=config["dataset_paths"]["validation"][0],
        config=config,
        logger=logger,
        require_label_idxs=False,  # Label indices are not required for validation.
        label_tokenizer=None,
    )
    if args.validation_path_name is not None
    else None
)

test_dataset = (
    ProteinDataset(
        data_paths=config["dataset_paths"]["test"][0],
        config=config,
        logger=logger,
        require_label_idxs=False,  # Label indices are not required for testing
        label_tokenizer=None,
    )
    if args.test_paths_names is not None
    else None
)

# Add datasets to a dictionary
# TODO: This does not support multiple datasets. But I think we should remove that support anyway. Too complicated.
datasets = {
    "train": [train_dataset],
    "validation": [validation_dataset],
    "test": [test_dataset],
}

# Remove empty datasets. May happen in cases like only validating a model.
datasets = {k: v for k, v in datasets.items() if v[0] is not None}

# Define label sample sizes for train, validation, and test loaders
label_sample_sizes = {
    "train": params["TRAIN_LABEL_SAMPLE_SIZE"],
    "validation": params["VALIDATION_LABEL_SAMPLE_SIZE"],
    "test": None,  # No sampling for the test set
}

# Initialize new run
logger.info(f"################## {timestamp} RUNNING train.py ##################")


# Define data loaders
loaders = create_multiple_loaders(
    datasets=datasets, params=params, num_workers=params["NUM_WORKERS"], pin_memory=True
)

model_weights = paths[f"PROTEINFER_{args.proteinfer_weights}_WEIGHTS_PATH"]
if args.model_weights_id is not None:
    model_weights = model_weights.replace(".pkl", f"{args.model_weights_id}.pkl")

model = ProteInfer.from_pretrained(
    weights_path=model_weights,
    num_labels=config["embed_sequences_params"][
        f"PROTEINFER_NUM_{args.proteinfer_weights}_LABELS"
    ],
    input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
    output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
    kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
    activation=torch.nn.ReLU,
    dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
    num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
    bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = model.eval()

label_normalizer = read_json(paths["PARENTHOOD_LIB_PATH"])


# Initialize EvalMetrics
eval_metrics = EvalMetrics(device=device)
label_sample_sizes = {
    k: (v if v is not None else len(datasets[k][0].label_vocabulary))
    for k, v in label_sample_sizes.items()
    if k in datasets.keys()
}

full_data_path = (
    "FULL_DATA_PATH" if args.proteinfer_weights == "GO" else "FULL_EC_DATA_PATH"
)
PROTEINFER_VOCABULARY = generate_vocabularies(
    file_path=config["paths"][full_data_path]
)["label_vocab"]

for loader_name, loader in loaders.items():
    print(loader_name, len(loader[0].dataset.label_vocabulary))

    represented_labels = [
        label in loader[0].dataset.label_vocabulary for label in PROTEINFER_VOCABULARY
    ]

    test_metrics = eval_metrics.get_metric_collection_with_regex(
        pattern="f1_m.*",
        threshold=args.threshold,
        num_labels=label_sample_sizes["test"]
        if (params["IN_BATCH_SAMPLING"] or params["GRID_SAMPLER"]) is False
        else None,
    )

    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    focal_loss = FocalLoss(
        gamma=config["params"]["FOCAL_LOSS_GAMMA"],
        alpha=config["params"]["FOCAL_LOSS_ALPHA"],
    )
    total_bce_loss = 0
    total_focal_loss = 0
    test_results = defaultdict(list)

    mAP_micro = BinaryAUPRC(device="cpu")
    mAP_macro = MultilabelAUPRC(device="cpu", num_labels=label_sample_sizes["test"])

    with torch.no_grad(), autocast(enabled=True):
        for batch_idx, batch in tqdm(enumerate(loader[0]), total=len(loader[0])):
            # Unpack the validation or testing batch
            (
                sequence_onehots,
                sequence_lengths,
                sequence_ids,
                label_multihots,
                label_embeddings,
            ) = (
                batch["sequence_onehots"],
                batch["sequence_lengths"],
                batch["sequence_ids"],
                batch["label_multihots"],
                batch["label_embeddings"],
            )
            sequence_onehots, sequence_lengths, label_multihots = to_device(
                device, sequence_onehots, sequence_lengths, label_multihots
            )

            logits = model(sequence_onehots, sequence_lengths)
            if args.only_represented_labels:
                logits = logits[:, represented_labels]

            probabilities = torch.sigmoid(logits)

            if not args.only_inference:
                test_metrics(probabilities, label_multihots)

                if loader_name in ["validation", "test"]:
                    mAP_micro.update(
                        probabilities.cpu().flatten(), label_multihots.cpu().flatten()
                    )
                    mAP_macro.update(probabilities.cpu(), label_multihots.cpu())

                total_bce_loss += bce_loss(logits, label_multihots.float())
                total_focal_loss += focal_loss(logits, label_multihots.float())

            if args.save_prediction_results:
                test_results["sequence_ids"].append(sequence_ids)
                test_results["logits"].append(logits.cpu())
                test_results["labels"].append(label_multihots.cpu())

        if not args.only_inference:
            test_metrics = test_metrics.compute()
            test_metrics.update({"bce_loss": total_bce_loss / len(loader[0])})
            test_metrics.update({"focal_loss": total_focal_loss / len(loader[0])})

            if loader_name in ["validation", "test"]:
                test_metrics.update(
                    {"map_micro": mAP_micro.compute(), "map_macro": mAP_macro.compute()}
                )

        print("\n\n", "=" * 20)
        print(f"##{loader_name}##")
        print(test_metrics)
        print("=" * 20, "\n\n")

        if args.save_prediction_results:
            for key in test_results.keys():
                if key == "sequence_ids":
                    test_results[key] = np.array(
                        [j for i in test_results["sequence_ids"] for j in i]
                    )
                else:
                    test_results[key] = torch.cat(test_results[key]).numpy()
            print("saving resuts...")
            save_evaluation_results(
                results=test_results,
                label_vocabulary=loader[0].dataset.label_vocabulary,
                run_name=f"{task}_{args.name}" + str(args.model_weights_id)
                if args.model_weights_id is not None
                else "",
                output_dir=config["paths"]["RESULTS_DIR"],
                data_split_name=loader_name,
                save_as_h5=True,
            )
            print("Done saving resuts...")
torch.cuda.empty_cache()
