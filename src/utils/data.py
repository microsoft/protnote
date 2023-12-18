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
import transformers
from collections import OrderedDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def log_gpu_memory_usage(logger, device_id):
    # Initialize NVML and get handle for the device
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)

    # Get memory information using NVML
    info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total
    used_memory = info.used
    memory_percent = used_memory / total_memory * 100

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device_id)

    # Log memory usage information
    logger.info(
        f"GPU memory occupied: {used_memory // 1024 ** 2} MB ({memory_percent:.2f}% of total memory {total_memory // 1024 ** 2} MB). "
        f"Device {device_id} [Name: {torch.cuda.get_device_name(device_id)}]")



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
    zip_name = wget.download(url)
    
    # Move the file to data/swissprot
    os.rename(zip_name, filename)

    # Unzip the downloaded file
    with gzip.open(zip_name, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())

    print(
        f"File {output_file + '.gz'} has been downloaded and unzipped to {output_file}.")


def save_checkpoint(model, optimizer, epoch, best_val_metric, model_path):
    """
    Save model and optimizer states as a checkpoint.

    Args:
    - model (torch.nn.Module): The model whose state we want to save.
    - optimizer (torch.optim.Optimizer): The optimizer whose state we want to save.
    - epoch (int): The current training epoch.
    - model_path (str): The path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_metric': best_val_metric,
    }

    torch.save(checkpoint, model_path)


def load_model(trainer, checkpoint_path, from_checkpoint=False):
    """
    Load the model's state from a given checkpoint.

    This function is designed to handle checkpoints from both Data Distributed Parallel (DDP) wrapped 
    and non-DDP models. If the checkpoint originates from a DDP-wrapped model, the function will adjust 
    the state dictionary keys accordingly before loading.

    Parameters:
    - trainer (object): An instance of the trainer containing the model, optimizer, and other training attributes.
    - checkpoint_path (str): The path to the checkpoint file to be loaded.
    - from_checkpoint (bool, optional): If True, the function will also load the optimizer's state, 
      epoch number, and best validation metric from the checkpoint. Defaults to False.

    Note:
    The function assumes that the model in the trainer object is DDP-wrapped.
    """

    # Load the entire checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['model_state_dict']

    # Check if the state_dict is from a DDP-wrapped model
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove the "module." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Load the state_dict into the model
    trainer.model.module.load_state_dict(state_dict)

    # Load the optimizer state and epoch number if they exist in the checkpoint
    if 'optimizer_state_dict' in checkpoint and from_checkpoint:
        trainer.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
    if 'epoch' in checkpoint and from_checkpoint:
        trainer.epoch = checkpoint['epoch']
    if 'best_val_metric' in checkpoint and from_checkpoint:
        trainer.best_val_metric = checkpoint['best_val_metric']

    # Delete the checkpoint to save memory
    del checkpoint


def seed_everything(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def load_gz_json(path):
    with open(path, "rb") as f:
        with gzip.GzipFile(fileobj=f, mode="rb") as gzip_file:
            return json.load(gzip_file)
