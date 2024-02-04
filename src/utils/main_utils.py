import os
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils.data import read_json, read_fasta, read_pickle, save_to_pickle
from src.data.collators import collate_variable_sequence_length
from torch.utils.data import ConcatDataset, DataLoader
from src.utils.models import generate_label_embeddings_from_text
import pandas as pd
 
 
def validate_arguments(args, parser):
    # Ensure the full data path is provided
    if args.full_path_name is None:
        parser.error(
            "You must provide the full path name to define the vocabularies using --full-path-name."
        )
 
    # Raise error if train is provided without val
    if (args.train_path_name is not None):
        if (args.validation_path_name is None):
            parser.error(
                "If providing --train-path-name you must provide --val-path-name."
            )
 
 
    # Raise error if no train path is provided and no model is loaded
    if (
        (args.train_path_name is None)
        and (args.load_model is None) 
    ):
        parser.error(
            "You must provide --load-model if no --train-path-names is provided")
 
 
    # Raise error if none of the paths are provided
    if (args.test_paths_names is None) & \
        (args.train_path_name is None) & \
        (args.validation_path_name is None):
 
        parser.error("You must provide one of the following options:\n"
                     "--test-path-names --load-model\n"
                     "--val-path-names --load-model\n"
                     "--train-path-name and --validation-path-name (optional load model)\n"
                     "--train-path-name and --validation-path-name --test-path-names (optional load model)\n"
                     "All cases with including --full-path-name. Please provide the required option(s) and try again.")
    if (args.save_prediction_results)\
&((args.test_paths_names is None)\
&(args.validation_path_name is None)):
        parser.error("You must provide --test-path-names and/or --val-path-names to save the results of the validation and/or test sets.")
 
 
def generate_sequence_embeddings(device, sequence_encoder, datasets, params):
    """Generate sequence embeddings for the given datasets."""
    sequence_encoder = sequence_encoder.to(device)
    all_datasets = [dataset for dataset_list in datasets.values() for dataset in dataset_list]
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
        shuffle=False,
        collate_fn=collate_variable_sequence_length,
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )
    # Initialize an empty list to store data
    data_list = []
    c=1
    for batch in tqdm(combined_loader):
        print(c,device)
        sequence_onehots, sequence_ids, sequence_lengths = (
            batch["sequence_onehots"].to(device),
            batch["sequence_ids"],
            batch["sequence_lengths"].to(device)
        )
        with torch.no_grad():
            embeddings = sequence_encoder.get_embeddings(
                sequence_onehots, sequence_lengths)
            for i, original_id in enumerate(sequence_ids):
                data_list.append((original_id, embeddings[i].cpu().numpy()))
        c+=1
 
    # Convert the list to a DataFrame
    df = pd.DataFrame(data_list, columns=["ID", "Embedding"]).set_index("ID")
    return df
 


def get_or_generate_label_embeddings(
    label_annotations,
    label_tokenizer,
    label_encoder,
    pooling_method,
    label_embedding_path=None,
    logger=None,
    batch_size_limit=1000,
    is_master=True,
    append_in_cpu=True
):
    """Load or generate label embeddings based on the provided paths and parameters."""

    assert pooling_method in ['mean','last_token','all'], f"pooling_method = {pooling_method} not supported for caching"

    if os.path.exists(label_embedding_path):
        label_embedding_matrix = torch.load(label_embedding_path)
        if logger is not None and is_master:
            logger.info(
                f"Loaded label embeddings from {label_embedding_path}")
        return label_embedding_matrix
    else:
        if logger is not None:
            logger.info("Generating and saving label embeddings from text...")
 
        # If not master, throw an error
        if is_master:
            label_encoder.eval()
            with torch.no_grad():
                label_embedding_matrix = generate_label_embeddings_from_text(
                    label_annotations,
                    label_tokenizer,
                    label_encoder,
                    pooling_method,
                    batch_size_limit,
                    append_in_cpu
                ).cpu()
            label_encoder.train()

            torch.save(label_embedding_matrix, label_embedding_path)
            if logger is not None:
                logger.info(
                    f"Saved label embeddings to {label_embedding_path}")
            return label_embedding_matrix
 
 
def get_or_generate_sequence_embeddings(paths, device, sequence_encoder, datasets, params, logger):
    """Load or generate sequence embeddings based on the provided paths and parameters."""
    if "SEQUENCE_EMBEDDING_PATH" in paths and os.path.exists(paths["SEQUENCE_EMBEDDING_PATH"]):
        sequence_embedding_df = read_pickle(paths["SEQUENCE_EMBEDDING_PATH"])
        logger.info(
            f"Loaded sequence embeddings from {paths['SEQUENCE_EMBEDDING_PATH']}")
    else:
        logger.info("Generating sequence embeddings...")
        sequence_embedding_df = generate_sequence_embeddings(
            device, sequence_encoder, datasets, params)
        
        # Save the sequence embeddings to paths["SEQUENCE_EMBEDDING_PATH"]
        if paths["SEQUENCE_EMBEDDING_PATH"] is not None:
            torch.save(sequence_embedding_df, paths["SEQUENCE_EMBEDDING_PATH"])
            if logger is not None:
                    logger.info(
                        f"Saved label embeddings to {paths['SEQUENCE_EMBEDDING_PATH']}")

    return sequence_embedding_df
 
 
def get_or_generate_vocabularies(full_data_path, vocabularies_dir, logger,prefix=''):
    """Load or generate vocabularies based on the provided paths."""
    all_vocab_types = ['amino_acid_vocab',
                       'GO_label_vocab',
                       'sequence_id_vocab']
    
    missing_vocab_types = []
    vocabularies = {}
    for vocab_type in all_vocab_types:
        full_path = os.path.join(vocabularies_dir, f"{prefix+vocab_type}.json")
        if os.path.exists(full_path):
            vocabularies[vocab_type] = read_json(full_path)
            logger.info(f"Loaded {vocab_type} vocabulary from {full_path}")
        else:
            missing_vocab_types.append(vocab_type)
    if missing_vocab_types:
        logger.info(
            f"Generating {', '.join(missing_vocab_types)} vocabularies...")
        vocabularies.update(generate_vocabularies(
            full_data_path, missing_vocab_types, logger, vocabularies_dir, prefix=prefix))
    return vocabularies
 
 
def generate_vocabularies(data_path, vocab_types, logger, output_dir=None, prefix=''):
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
                with open(os.path.join(output_dir, f"{prefix+key}.json"), "w") as f:
                    json.dump(value, f)
                    logger.info(
                        f"Saved {len(value)} items as the {key} to {os.path.join(output_dir, key, '.json')}")
    return vocab_dict