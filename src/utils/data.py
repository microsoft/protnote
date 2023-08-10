from Bio import SeqIO
import json
import numpy as np
from typing import Optional,List

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

def ints_to_multihot(int_list: list,num_labels: list )->List:
    multihot = [0]*num_labels
    for i in int_list:
        multihot[i] = 1
    return multihot

def multihot_to_ints(multihot:List)->List:
    return [idx for idx,val in enumerate(multihot) if val==1]

    