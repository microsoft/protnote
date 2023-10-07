from Bio import SeqIO
import json
import pickle
import gzip
import os
import torch
import yaml
import random
import numpy as np
import wget
import hashlib
from collections import OrderedDict


def log_gpu_memory_usage(logger, device_id):
    device = torch.device(f"cuda:{device_id}")
    allocated = torch.cuda.memory_allocated(
        device) / (1024 ** 2)  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved(
        device) / (1024 ** 2)  # Convert bytes to MB
    total_memory = torch.cuda.get_device_properties(
        device).total_memory / (1024 ** 2)  # Convert bytes to MB

    allocated_percent = (allocated / total_memory) * 100
    reserved_percent = (reserved / total_memory) * 100
    combined = allocated + reserved
    combined_percent = (combined / total_memory) * 100
    logger.info("-" * 50)
    logger.info(
        f"Device {device_id} [Name: {torch.cuda.get_device_name(device)}]")
    logger.info(
        f"  Allocated Memory: {allocated:.2f} MB ({allocated_percent:.2f}%)")
    logger.info(
        f"  Reserved Memory: {reserved:.2f} MB ({reserved_percent:.2f}%)")
    logger.info(
        f"  Combined (Allocated + Reserved) Memory: {combined:.2f} MB ({combined_percent:.2f}%)")
    logger.info(f"  Total Device Memory: {total_memory:.2f} MB")
    logger.info("-" * 50)


def convert_float16_to_float32(df):
    float16_cols = df.select_dtypes(include='float16').columns
    df[float16_cols] = df[float16_cols].astype('float32')
    return df


def hash_alphanumeric_sequence_id(s: str):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def read_fasta(data_path: str, sep=" "):
    sequences_with_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        labels = record.description.split(sep)
        sequences_with_labels.append((sequence, labels))
    return sequences_with_labels


def read_yaml(data_path: str):
    with open(data_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def read_json(data_path: str):
    with open(data_path, "r") as file:
        data = json.load(file)
    return data


def write_json(data, data_path: str):
    with open(data_path, "w") as file:
        json.dump(data, file)


def get_vocab_mappings(vocabulary):
    assert len(vocabulary) == len(set(vocabulary)
                                  ), "items in vocabulary must be unique"
    term2int = {term: idx for idx, term in enumerate(vocabulary)}
    int2term = {idx: term for term, idx in term2int.items()}
    return term2int, int2term


def save_to_pickle(item, file_path: str):
    with open(file_path, "wb") as p:
        pickle.dump(item, p)


def read_pickle(file_path: str):
    with open(file_path, "rb") as p:
        item = pickle.load(p)
    return item


def download_and_unzip(url, output_file):
    """
    Download a file from a given link and unzip it.

    Args:
        link (str): The URL to download the file from.
        filename (str): The absolute path to save the downloaded file.
    """
    filename = output_file + '.gz'

    # Download the file from the web
    wget.download(url, filename)

    # Unzip the downloaded file
    with gzip.open(filename, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())

    print(
        f"File {output_file + '.gz'} has been downloaded and unzipped to {output_file}.")


def load_state_dict(model, checkpoint_path):
    """
    Load the model's state dict from the checkpoint.

    This function handles both DDP-wrapped and non-DDP checkpoints.

    :param model: The model into which the checkpoint's state dict should be loaded.
    :param checkpoint_path: Path to the checkpoint file.
    :return: The model with loaded state dict.
    """

    # Load checkpoint's state dict
    state_dict = torch.load(checkpoint_path)

    # Check if the state_dict is NOT from a DDP-wrapped model
    if not list(state_dict.keys())[0].startswith('module.'):
        # Create a new OrderedDict with the "module." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    return model


def seed_everything(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_gz_json(path):
    with open(path, "rb") as f:
        with gzip.GzipFile(fileobj=f, mode="rb") as gzip_file:
            return json.load(gzip_file)
