from Bio import SeqIO
import json
import numpy as np
from typing import Optional,List
import pickle
import pandas as pd
import gzip

def read_fasta(data_path:str,sep=" "):
    sequences_with_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        labels = record.description.split(sep)
        sequences_with_labels.append((sequence, labels))
    return sequences_with_labels

def read_json(data_path: str):
    with open(data_path,'r') as file:
        data = json.load(file)
    return data

def write_json(data,data_path: str):
    with open(data_path,'w') as file:
        json.dump(data,file)

def get_vocab_mappings(vocabulary):
    assert not any(vocabulary.count(x) > 1 for x in vocabulary), 'items in vocabulary must be unique'
    term2int = {term: idx for idx, term in enumerate(vocabulary)}
    int2term = {idx:term for term,idx in term2int.items()}
    return term2int, int2term


def save_to_pickle(item,file_path: str):
    with open(file_path,'wb') as p:
        pickle.dump(item,p)

def read_pickle(file_path: str):
    with open(file_path,'rb') as p:
        item = pickle.load(p)
    return item

def filter_annotations(sequences_with_labels: list, allowed_annotations: set) -> list:
    """
    Filters out specified annotations from a list of sequences with labels.

    Parameters:
    - sequences_with_labels (list): A list of tuples where each tuple contains a sequence and its associated labels.
    - allowed_annotations (set): Set of annotations that are allowed.

    Returns:
    - List of tuples where each tuple is (sequence, annotations) and each annotation is in the allowed_annotations set.
    """
    # Initialize the filtered data
    filtered_data = []
    for sequence, annotations in sequences_with_labels:
        # Filter the annotations for the current sequence
        filtered_annots = [annot for annot in annotations if annot in allowed_annotations]
        if filtered_annots:
            filtered_data.append((sequence, filtered_annots))
    return filtered_data

def load_gz_json(path):
  with open(path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as gzip_file:
      return json.load(gzip_file)    