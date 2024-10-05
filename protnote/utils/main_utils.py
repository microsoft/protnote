import torch
from tqdm import tqdm
from protnote.data.collators import collate_variable_sequence_length
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd
from functools import partial
import warnings


def validate_arguments(args, parser):
    # Ensure the full data path is provided, or we are using the zero shot model
    if args.full_path_name is None and "zero" not in str(args.train_path_name).lower():
        warnings.warn(
            "The full path name is not provided and the train path name does not contain the word 'zero'. Please ensure this is intentional."
        )

    # Raise error if train is provided without val
    if args.train_path_name is not None:
        if args.validation_path_name is None:
            parser.error(
                "If providing --train-path-name you must provide --val-path-name."
            )

    # Raise error if no train path is provided and no model is loaded
    if (args.train_path_name is None) and (args.model_file is None):
        parser.error(
            "You must provide --load-model if no --train-path-names is provided"
        )

    # Raise error if none of the paths are provided

    if (
        (args.test_paths_names is None)
        & (args.train_path_name is None)
        & (args.validation_path_name is None)
    ):
        parser.error(
            "You must provide one of the following options:\n"
            "--test-path-names --load-model\n"
            "--val-path-names --load-model\n"
            "--train-path-name and --validation-path-name (optional load model)\n"
            "--train-path-name and --validation-path-name --test-path-names (optional load model)\n"
            "All cases with including --full-path-name. Please provide the required option(s) and try again."
        )

    if (args.save_prediction_results) & (
        (args.test_paths_names is None) & (args.validation_path_name is None)
    ):
        parser.error(
            "You must provide --test-path-names and/or --val-path-names to save the results of the validation and/or test sets."
        )


def generate_sequence_embeddings(device, sequence_encoder, datasets, params):
    """Generate sequence embeddings for the given datasets."""
    sequence_encoder = sequence_encoder.to(device)
    sequence_encoder.eval()
    all_datasets = [
        dataset for dataset_list in datasets.values() for dataset in dataset_list
    ]
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
        shuffle=False,
        collate_fn=partial(
            collate_variable_sequence_length, return_label_multihots=False
        ),  # have to use return_label_multihots to ignore multihot concat with zero shot
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )
    # Initialize an empty list to store data
    data_list = []

    for batch in tqdm(combined_loader):
        sequence_onehots, sequence_ids, sequence_lengths = (
            batch["sequence_onehots"].to(device),
            batch["sequence_ids"],
            batch["sequence_lengths"].to(device),
        )
        with torch.no_grad():
            embeddings = sequence_encoder.get_embeddings(
                sequence_onehots, sequence_lengths
            )
            for i, original_id in enumerate(sequence_ids):
                data_list.append((original_id, embeddings[i].cpu().numpy()))

    sequence_encoder.train()
    # Convert the list to a DataFrame
    df = pd.DataFrame(data_list, columns=["ID", "Embedding"]).set_index("ID")
    return df
