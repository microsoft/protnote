from Bio import SeqIO
import json
import pickle
from collections import defaultdict
import gzip
import os
import torch
import yaml
import random
import numpy as np
import wget
import hashlib
import math
import re
import blosum as bl
from typing import Union,List,Set,Literal
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
              'label_vocab':set(),
              'sequence_id_vocab':set()
            }
    
    if isinstance(file_path,str):
        data = read_fasta(file_path)
    else:
        raise TypeError("File not supported, vocabularies can only be generated from .fasta files.")

    for sequence, sequence_id, labels in data:
        vocabs['sequence_id_vocab'].add(sequence_id)
        vocabs['label_vocab'].update(labels)
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


def ensure_list(value):
    # Case 1: If the value is already a list
    if isinstance(value, list):
        return value
    # Case 2: If the value is NaN
    elif value is math.nan or (isinstance(value, float) and math.isnan(value)):
        return []
    # Case 3: For all other cases (including strings)
    else:
        return [value]

def remove_obsolete_from_string(text):
    pattern= r'(?i)\bobsolete\.?\s*'
    return re.sub(pattern, '', text)



class Blossum62Mutations:
    def __init__(self,amino_acid_vocabulary:Union[Set,List]=None):

        if amino_acid_vocabulary is None:
            self.amino_acid_vocabulary = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                             'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        else:
            self.amino_acid_vocabulary = set(amino_acid_vocabulary)

        # Load the BLOSUM62 matrix and convert to defaultdict using dictionary comprehension
        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(dict, {aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()} for aa1 in blosum62.keys()})

    def get_aa_scores(self,amino_acid: str,mutation_type:Literal['conservative','non-conservative']):

        # Get the substitutions for the amino acid, ensuring only amino acids within the vocabulary are considered
        substitutions = self.blosum62[amino_acid]
        multiplier = -1 if mutation_type=='non-conservative'else 1
        substitutions = {aa: score*multiplier for aa, score in substitutions.items() if aa in self.amino_acid_vocabulary}
        amino_acids, scores = zip(*substitutions.items())
        return amino_acids, scores

    def get_most_extreme_mutation(self,amino_acid:str,mutation_type:Literal['conservative','non-conservative']):
        amino_acids, scores = self.get_aa_scores(amino_acid=amino_acid,
                                                 mutation_type=mutation_type)    
        fun = max if mutation_type == 'conservative' else min
        return amino_acids[scores.index(fun(scores))]

    def corrupt_sequence(self,sequence:str,mutation_type:Literal['conservative','non-conservative'],sample:bool):
        corrupted = ''
        for aa in sequence:
            corrupted+=self.corrupt_amino_acid(amino_acid=aa,mutation_type=mutation_type,sample=sample)
        return corrupted

    def corrupt_amino_acid(self,amino_acid:str,mutation_type:Literal['conservative','non-conservative'],sample:bool):
        if sample:
            return self.sample_aa(amino_acid=amino_acid,mutation_type=mutation_type)
        else:
            return self.get_most_extreme_mutation(amino_acid=amino_acid,mutation_type=mutation_type)

    def corrupt_sequence_at_locations(self,sequence:str,locations:set,mutation_type:Literal['conservative','non-conservative'],sample:bool):
        corrupted = ''
        for loc,aa in enumerate(sequence):
            if loc in locations:
                corrupted+=self.corrupt_amino_acid(amino_acid=aa,
                                                   mutation_type=mutation_type,
                                                   sample=sample)
            else:
                corrupted+=aa
        return corrupted

    def sample_aa(self, amino_acid: str,mutation_type:Literal['conservative','non-conservative']) -> str:
        """
        Sample an amino acid based on the BLOSUM62 substitution matrix, favoring mutations based on mutation_type selected. 
        Args:
            amino_acid (str): The amino acid to find a substitution for.
        Returns:
            str: The substituted amino acid.
        """

        amino_acids, scores = self.get_aa_scores(amino_acid=amino_acid,
                                                 mutation_type=mutation_type)
        
        # Use only non-negative scores
        probabilities = [max(0, score) for score in scores]
        total = sum(probabilities)
        
        # If all scores are negative, do not change the amino acid
        if total == 0:
            return amino_acid
        else:
            # Normalize the scores to sum to 1 and sample from the distribution
            probabilities = [p / total for p in probabilities]
            return random.choices(amino_acids, weights=probabilities, k=1)[0]