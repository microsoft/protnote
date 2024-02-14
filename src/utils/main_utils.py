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
from functools import partial

 
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
    sequence_encoder.eval()
    all_datasets = [dataset for dataset_list in datasets.values() for dataset in dataset_list]
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
        shuffle=False,
        collate_fn=partial(collate_variable_sequence_length,
                           return_label_multihots=False), #have to use return_label_multihots to ignore multihot concat with zero shot
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )
    # Initialize an empty list to store data
    data_list = []
    
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
                data_list.append((original_id, embeddings[i].cpu().numpy()))
    
    sequence_encoder.train()
    # Convert the list to a DataFrame
    df = pd.DataFrame(data_list, columns=["ID", "Embedding"]).set_index("ID")
    return df
 

 
 
def get_or_generate_sequence_embeddings(paths, device, sequence_encoder, datasets, params, logger, is_master=True):
    """Load or generate sequence embeddings based on the provided paths and parameters."""
    if "SEQUENCE_EMBEDDING_PATH" in paths and os.path.exists(paths["SEQUENCE_EMBEDDING_PATH"]):
        sequence_embedding_df = torch.load(paths["SEQUENCE_EMBEDDING_PATH"])
        if logger is not None and is_master:
                logger.info(
                    f"Loaded sequence embeddings from {paths['SEQUENCE_EMBEDDING_PATH']}")
                
        return sequence_embedding_df
    elif logger is not None and is_master:
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
 