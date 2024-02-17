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
from typing import Union
import transformers
from collections import OrderedDict
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
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
    """
    Reads a FASTA file and returns a list of tuples containing sequences, ids, and labels.
    """
    sequences_with_ids_and_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        components = record.description.split(sep)
        # labels[0] contains the sequence ID, and the rest of the labels are GO terms. 
        sequence_id = components[0]
        labels = components[1:]
        
        # Return a tuple of sequence, sequence_id, and labels
        sequences_with_ids_and_labels.append((sequence, sequence_id, labels))
    return sequences_with_ids_and_labels


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

def generate_vocabularies(file_path: str)->dict:
    """
    Generate vocabularies based on the provided data path.
    path must be .fasta file
    """
    vocabs = {'amino_acid_vocab':set(),
              'GO_label_vocab':set(),
              'sequence_id_vocab':set()
            }
    
    if isinstance(file_path,str):
        data = read_fasta(file_path)
    else:
        raise TypeError("File not supported, vocabularies can only be generated from .fasta files.")

    for sequence, sequence_id, labels in data:
        vocabs['sequence_id_vocab'].add(sequence_id)
        vocabs['GO_label_vocab'].update(labels)
        vocabs['amino_acid_vocab'].update(list(sequence))
    
    for vocab_type in vocabs.keys():
        vocabs[vocab_type] = sorted(list(vocabs[vocab_type]))
 
    return vocabs

def save_to_pickle(item, file_path: str):
    with open(file_path, "wb") as p:
        pickle.dump(item, p)
        

def save_to_fasta(sequence_id_labels_tuples, output_file):
    """
    Save a list of tuples in the form (sequence, [labels]) to a FASTA file.

    :param sequence_label_tuples: List of tuples containing sequences and labels
    :param output_file: Path to the output FASTA file
    """
    records = []
    for _, (sequence, id, labels,) in enumerate(sequence_id_labels_tuples):
        # Create a description from labels, joined by space
        description = " ".join(labels)
       
        record = SeqRecord(Seq(sequence), id=id, description=description)
        records.append(record)

    # Write the SeqRecord objects to a FASTA file
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
        print("Saved FASTA file to " + output_file)

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
