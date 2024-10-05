import os
import torch
import argparse
import os
import sys
import logging
from typing import Literal
from pprint import pprint
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
import pandas as pd
import subprocess
from protnote.utils.evaluation import EvalMetrics
from protnote.utils.data import generate_vocabularies, read_yaml
from protnote.utils.configs import load_config, construct_absolute_paths
from test_models import TEST_COMMANDS, MODEL_PATH_TOKEN, MODEL_NAME_TOKEN

#TODO: This could be more general by eliminating unnecessary dependencies and passing the embedding file names as arguments
def main(
    annotation_type: str,
    label_embedding_model: Literal["E5", "BioGPT"],
    test_name: str,
    model_name: str,
    cache: bool,
):
    
    # Load the configuration and project root
    config, project_root = load_config()
    results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]
    full_data_path = config["paths"]['data_paths']['FULL_DATA_PATH']
    proteinfer_predictions_path = results_dir / f"test_logits_{annotation_type}_{test_name}_proteinfer.h5"
    protnote_labels_path = results_dir / f"test_1_labels_{test_name}_{model_name}.h5"
    output_file = results_dir / f"test_logits_{annotation_type}_{test_name}_{label_embedding_model}_baseline.h5"
    

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-4s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    assert test_name in TEST_COMMANDS.keys(), f"{test_name} does not exist"

    if label_embedding_model == "E5":
        label_embeddings = "2024_E5_multiling_inst_frozen_label_embeddings_mean"

    elif label_embedding_model == "BioGPT":
        label_embeddings = "2024_BioGPT_frozen_label_embeddings_mean"

    base_embeddings_path = project_root / "data" / "embeddings" / f"{label_embeddings}.pt"
    base_embeddings_idx_path = project_root / "data" / "embeddings" / f"{label_embeddings}_index.pt"

    if (not os.path.exists(proteinfer_predictions_path)) | (not cache):
        if annotation_type == "GO":
            logger.info("Running inference with Proteinfer...")
            subprocess.run(
                f"python bin/test_proteinfer.py --test-paths-names {test_name} --only-inference --override EXTRACT_VOCABULARIES_FROM null --save-prediction-results --name {test_name}_proteinfer --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH",
                shell=True,
            )
        elif annotation_type == "EC":
            subprocess.run(
                f"python bin/test_proteinfer.py --test-paths-names {test_name} --only-inference --override EXTRACT_VOCABULARIES_FROM null --save-prediction-results --name {test_name}_proteinfer --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH --annotations-path-name EC_ANNOTATIONS_PATH",
                shell=True,
            )
    else:
        logger.info("Found existing proteinfer predictions... caching")

    if (
        not os.path.exists(protnote_labels_path)
    ) | (not cache):
        logger.info("Getting label dataframe in a hacky way...")
        #TODO: This can be avoided by creating the label dataframe and performing the preprocessing
        # instead of running the model
        subprocess.run(
            TEST_COMMANDS[test_name]
            .replace(MODEL_PATH_TOKEN, f"{model_name}.pt")
            .replace(MODEL_NAME_TOKEN, model_name) + " --save-prediction-results",
            shell=True,
        )

    zero_shot_pinf_logits = pd.read_hdf(proteinfer_predictions_path,key="logits_df").astype("float32")
    zero_shot_labels = pd.read_hdf(protnote_labels_path, key="labels_df")
    vocabularies = generate_vocabularies(file_path=str(full_data_path))
    zero_shot_pinf_logits.columns = vocabularies["label_vocab"]

    embeddings = torch.load(base_embeddings_path)
    embeddings_idx = torch.load(base_embeddings_idx_path)

    # Select embeddings based on name / short definition
    embedding_mask = embeddings_idx["description_type"] == "name"
    embeddings_idx = embeddings_idx[embedding_mask].reset_index(drop=True)
    embeddings = embeddings[embedding_mask]

    # Create embeddings matrix of known proteinfer GO Term definitions
    train_embeddings_mask = embeddings_idx["id"].isin(vocabularies["label_vocab"])
    train_embeddings_idx = embeddings_idx[train_embeddings_mask].reset_index(drop=True)
    train_embeddings = embeddings[train_embeddings_mask]

    if annotation_type == "GO":
        # Create embedding matrix of the new/unknown GO Term definitions
        zero_shot_embeddings_mask = embeddings_idx["id"].isin(zero_shot_labels.columns)
        zero_shot_embeddings_idx = embeddings_idx[
            zero_shot_embeddings_mask
        ].reset_index(drop=True)
        zero_shot_embeddings = embeddings[zero_shot_embeddings_mask]

    if annotation_type == "EC":
        if label_embedding_model == "BioGPT":
            label_embeddings_new = "ecv1_BioGPT_frozen_label_embeddings_mean"
        elif label_embedding_model == "E5":
            label_embeddings_new = "ecv1_E5_multiling_inst_frozen_label_embeddings_mean"

        new_embeddings_path = project_root / "data" / "embeddings" / f"{label_embeddings_new}.pt"
        new_embeddings_idx_path = project_root / "data" / "embeddings" / f"{label_embeddings_new}_index.pt"

        embeddings_new = torch.load(new_embeddings_path)
        embeddings_idx_new = torch.load(new_embeddings_idx_path)

        # Create embedding matrix of the new/unknown GO Term definitions
        embedding_mask_new = embeddings_idx_new["description_type"] == "name"
        embeddings_idx_new = embeddings_idx_new[embedding_mask_new].reset_index(
            drop=True
        )
        embeddings_new = embeddings_new[embedding_mask_new]

        zero_shot_embeddings_mask = embeddings_idx_new["id"].isin(
            zero_shot_labels.columns
        )
        zero_shot_embeddings_idx = embeddings_idx_new[
            zero_shot_embeddings_mask
        ].reset_index(drop=True)
        zero_shot_embeddings = embeddings_new[zero_shot_embeddings_mask]

    # Create description mapping from seen to new
    label_train_2_zero_shot_similarities = (
        torch.nn.functional.normalize(zero_shot_embeddings)
        @ torch.nn.functional.normalize(train_embeddings).T
    )
    zero_shot_label_mapping = {
        zero_shot_embeddings_idx["id"]
        .iloc[zero_shot_label_idx]: train_embeddings_idx["id"]
        .iloc[train_label_idx.item()]
        for zero_shot_label_idx, train_label_idx in enumerate(
            label_train_2_zero_shot_similarities.max(dim=-1).indices
        )
    }

    # Create baseline predictions by replicating logits of most similar labels
    zero_shot_pinf_baseline_logits = zero_shot_pinf_logits[
        [zero_shot_label_mapping[i] for i in zero_shot_labels.columns]
    ]
    zero_shot_pinf_baseline_logits.columns = zero_shot_labels.columns

    zero_shot_pinf_baseline_logits.to_hdf(output_file,key='logits_df')

    # Evaluate performance of baseline
    eval_metrics = EvalMetrics(device="cuda")
    mAP_micro = BinaryAUPRC(device="cpu")
    mAP_macro = MultilabelAUPRC(device="cpu", num_labels=zero_shot_labels.shape[-1])
    metrics = eval_metrics.get_metric_collection_with_regex(
        pattern="f1_m.*", threshold=0.5, num_labels=zero_shot_labels.shape[-1]
    )

    metrics(
        torch.sigmoid(
            torch.tensor(zero_shot_pinf_baseline_logits.values, device="cuda")
        ),
        torch.tensor(zero_shot_labels.values, device="cuda"),
    )
    mAP_micro.update(
        torch.sigmoid(torch.tensor(zero_shot_pinf_baseline_logits.values)).flatten(),
        torch.tensor(zero_shot_labels.values).flatten(),
    )
    mAP_macro.update(
        torch.sigmoid(torch.tensor(zero_shot_pinf_baseline_logits.values)),
        torch.tensor(zero_shot_labels.values),
    )

    metrics = metrics.compute()
    metrics.update({"map_micro": mAP_micro.compute(), "map_macro": mAP_macro.compute()})
    metrics = {k: v.item() for k, v in metrics.items()}
    pprint(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline inference")
    parser.add_argument("--annotation-type", type=str, default="GO")
    parser.add_argument(
        "--test-name",
        type=str,
        required=True,
        help="The name of the test set to run baseline on",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of the latest zero shot model without the .pt. Its predictions wont be used, it is a hack to obtain processed/clean label df",
    )
    parser.add_argument(
        "--label-embedding-model",
        type=str,
        required=True,
        help="The name of LLM used for text embeddings: E5 or BioGPT",
    )
    parser.add_argument(
        "--cache",
        help="Whether to cache proteinfer predictions if available",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        annotation_type=args.annotation_type,
        test_name=args.test_name,
        model_name=args.model_name,
        label_embedding_model=args.label_embedding_model,
        cache=args.cache,
    )


