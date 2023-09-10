from Bio import SeqIO
import json
import pickle
import gzip
import os
import torch
import yaml


def read_fasta(data_path: str, sep=" "):
    sequences_with_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        labels = record.description.split(sep)
        sequences_with_labels.append((sequence, labels))
    return sequences_with_labels


def read_yaml(data_path: str):
    with open(data_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def read_json(data_path: str):
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data, data_path: str):
    with open(data_path, 'w') as file:
        json.dump(data, file)


def get_vocab_mappings(vocabulary):
    assert len(vocabulary) == len(set(vocabulary)
                                  ), 'items in vocabulary must be unique'
    term2int = {term: idx for idx, term in enumerate(vocabulary)}
    int2term = {idx: term for term, idx in term2int.items()}
    return term2int, int2term


def save_to_pickle(item, file_path: str):
    with open(file_path, 'wb') as p:
        pickle.dump(item, p)


def read_pickle(file_path: str):
    with open(file_path, 'rb') as p:
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
        filtered_annots = [
            annot for annot in annotations if annot in allowed_annotations]
        if filtered_annots:
            filtered_data.append((sequence, filtered_annots))
    return filtered_data


def load_gz_json(path):
    with open(path, 'rb') as f:
        with gzip.GzipFile(fileobj=f, mode='rb') as gzip_file:
            return json.load(gzip_file)


def load_embeddings(embedding_path, id_map, embedding_dim, device):
    """
    Load embeddings from a given path and convert them into a tensor matri for an nn.Embedding layer.

    Args:
    - embedding_path (str): Path to the embeddings file. Must be a dictionary.
    - id_map (dict): Mapping from alphanumeric IDs to numeric IDs.
    - embedding_dim (int): Dimension of the embeddings.
    - device (torch.device): Device to which the tensor should be moved.

    Returns:
    - torch.Tensor: A tensor matrix containing the embeddings.
    """
    embeddings = read_pickle(embedding_path)
    numeric_id_embedding_map = {
        id_map[k]: v for k, v in embeddings.items() if k in id_map
    }
    max_id = max(numeric_id_embedding_map.keys())
    embedding_matrix = torch.zeros(max_id + 1, embedding_dim, device=device)
    for numeric_id, embedding in numeric_id_embedding_map.items():
        tensor_embedding = torch.tensor(embedding, device=device)
        embedding_matrix[numeric_id] = tensor_embedding
    return embedding_matrix


def load_model_weights(model, path):
    """
    Loads PyTorch model weights from a .pt file.
    """
    assert path and os.path.exists(path), f"Model weights not found at {path}."
    model.load_state_dict(torch.load(path))
