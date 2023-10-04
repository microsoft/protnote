import os
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils.data import read_json, read_fasta, read_pickle, save_to_pickle
from src.data.collators import collate_variable_sequence_length
from torch.utils.data import ConcatDataset, DataLoader
from src.utils.models import generate_label_embeddings_from_text


def validate_arguments(args, parser):
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


def prompt_user_for_path(message, default_path=None):
    """Prompt the user for a file path and return it."""
    user_input = input(f"{message} (Default: {default_path}): ").strip()
    return user_input if user_input else default_path


def generate_sequence_embeddings(device, sequence_encoder, datasets, params):
    """Generate sequence embeddings for the given datasets."""
    sequence_encoder = sequence_encoder.to(device)
    all_datasets = datasets["train"] + \
        datasets["validation"] + datasets["test"]
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=params["SEQUENCE_BATCH_SIZE_LIMIT"],
        shuffle=False,
        collate_fn=collate_variable_sequence_length,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )
    sequence_embedding_dict = {}
    for batch in tqdm(combined_loader):
        sequence_onehots, sequence_ids, sequence_lengths = (
            batch["sequence_onehots"].to(device),
            batch["sequence_ids"],
            batch["sequence_lengths"].to(device)
        )
        with torch.no_grad():
            embeddings = sequence_encoder.get_embeddings(
                sequence_onehots, sequence_lengths)
            for i, original_id in enumerate(sequence_ids):
                sequence_embedding_dict[original_id] = embeddings[i].cpu()
    return sequence_embedding_dict


def get_or_generate_label_embeddings(paths, device, label_annotations, label_tokenizer, label_encoder, logger, label_batch_size_limit):
    """Load or generate label embeddings based on the provided paths and parameters."""
    if "LABEL_EMBEDDING_PATH" in paths and os.path.exists(paths["LABEL_EMBEDDING_PATH"]):
        label_embedding_matrix = torch.load(
            paths["LABEL_EMBEDDING_PATH"], map_location=device)
        logger.info(
            f"Loaded label embeddings from {paths['LABEL_EMBEDDING_PATH']}")
    else:
        logger.info("Generating label embeddings...")
        label_embedding_matrix = generate_label_embeddings_from_text(
            label_annotations, label_tokenizer, label_encoder, label_batch_size_limit)
        save_path = prompt_user_for_path(
            "Enter the full file path to save the label embeddings, or hit enter to continue without saving",
            default_path=None
        )
        if save_path:
            torch.save(label_embedding_matrix, save_path)
            logger.info(f"Saved label embeddings to {save_path}")
        else:
            logger.info("Label embeddings not saved.")
    return label_embedding_matrix


def get_or_generate_sequence_embeddings(paths, device, sequence_encoder, datasets, params, logger):
    """Load or generate sequence embeddings based on the provided paths and parameters."""
    if "SEQUENCE_EMBEDDING_PATH" in paths and os.path.exists(paths["SEQUENCE_EMBEDDING_PATH"]):
        sequence_embedding_dict = read_pickle(paths["SEQUENCE_EMBEDDING_PATH"])
        sequence_embedding_dict = {k: torch.tensor(
            v) if isinstance(v, np.ndarray) else v for k, v in sequence_embedding_dict.items()}
        logger.info(
            f"Loaded sequence embeddings from {paths['SEQUENCE_EMBEDDING_PATH']}")
    else:
        logger.info("Generating sequence embeddings...")
        sequence_embedding_dict = generate_sequence_embeddings(
            device, sequence_encoder, datasets, params)
        save_path = prompt_user_for_path(
            "Enter the full file path to save the sequence embeddings, or hit enter to continue without saving",
            default_path=None
        )
        if save_path:
            with open(save_path, 'wb') as f:
                save_to_pickle(sequence_embedding_dict, save_path)
            logger.info(f"Saved sequence embeddings to {save_path}")
        else:
            logger.info("Sequence embeddings not saved.")
    return sequence_embedding_dict


def get_or_generate_vocabularies(full_data_path, vocabularies_dir, logger):
    """Load or generate vocabularies based on the provided paths."""
    all_vocab_types = ['amino_acid_vocab',
                       'GO_label_vocab', 'sequence_id_vocab']
    missing_vocab_types = []
    vocabularies = {}
    for vocab_type in all_vocab_types:
        full_path = os.path.join(vocabularies_dir, f"{vocab_type}.json")
        if os.path.exists(full_path):
            vocabularies[vocab_type] = read_json(full_path)
            logger.info(f"Loaded {vocab_type} vocabulary from {full_path}")
        else:
            missing_vocab_types.append(vocab_type)
    if missing_vocab_types:
        vocabularies.update(generate_vocabularies(
            full_data_path, missing_vocab_types, logger, vocabularies_dir))
    return vocabularies


def generate_vocabularies(data_path, vocab_types, logger, output_dir=None):
    """Generate vocabularies based on the provided data path."""
    data = read_fasta(data_path)
    go_labels, amino_acids, sequence_ids = set(), set(), set()
    for sequence, labels in data:
        sequence_ids.add(labels[0])
        go_labels.update(labels[1:])
        amino_acids.update(list(sequence))

    vocab_dict = {}
    if "GO_label_vocab" in vocab_types:
        vocab_dict["GO_label_vocab"] = sorted(list(go_labels))
    if "amino_acid_vocab" in vocab_types:
        vocab_dict["amino_acid_vocab"] = sorted(list(amino_acids))
    if "sequence_id_vocab" in vocab_types:
        vocab_dict["sequence_id_vocab"] = sorted(list(sequence_ids))

    if output_dir:
        for key, value in vocab_dict.items():
            if value:
                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                    json.dump(value, f)
                    logger.info(
                        f"Saved {len(value)} items as the {key} to {os.path.join(output_dir, key, '.json')}")
    return vocab_dict
