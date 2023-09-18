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
from torch.utils.data import DataLoader, TensorDataset


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
            annot for annot in annotations if annot in allowed_annotations
        ]
        if filtered_annots:
            filtered_data.append((sequence, filtered_annots))
    return filtered_data


def load_gz_json(path):
    with open(path, "rb") as f:
        with gzip.GzipFile(fileobj=f, mode="rb") as gzip_file:
            return json.load(gzip_file)


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


def create_ordered_tensor(data_path, id_map, data_dim, device):
    """
    Load data from a given path and organize it into a tensor matrix based on a given ID mapping.

    Args:
    - data_path (str): Path to the data file. Must be a dictionary.
    - id_map (dict): Mapping from original IDs to desired order of numeric IDs.
    - data_dim (int): Dimension of the data.
    - device (torch.device): Device to which the tensor should be moved.

    Returns:
    - torch.Tensor: A tensor matrix containing the data, with each row corresponding to one id.
    """
    data = read_pickle(data_path)
    numeric_id_data_map = {id_map[k]: v for k,
                           v in data.items() if k in id_map}

    # Assert that the maximum id in the map is less than the number of data entries
    assert max(numeric_id_data_map.keys()) < len(
        id_map
    ), f"Maximum numeric ID in the map ({max(numeric_id_data_map.keys())}) is not less than the number of data entries ({len(id_map)})."

    data_matrix = torch.zeros(len(id_map), data_dim, device=device)
    for numeric_id, data_entry in numeric_id_data_map.items():
        tensor_data = torch.tensor(data_entry, device=device)
        data_matrix[numeric_id] = tensor_data
    return data_matrix


def load_model_weights(model, path):
    """
    Loads PyTorch model weights from a .pt file.
    """
    assert path and os.path.exists(path), f"Model weights not found at {path}."
    model.load_state_dict(torch.load(path))


# def get_tokenized_labels_dataloader(
#     go_descriptions_path: str,
#     llm_checkpoint_path: str,
#     train_label_encoder: bool,
#     label_vocabulary: list,
#     label2int_mapping: dict,
#     batch_size: int,
#     device: str,
# ):
#     # Load the go annotations (include free text) from data file
#     annotations = read_pickle(go_descriptions_path)

#     # Filter the annotations df to be only the labels in label_vocab. In annotations, the go id is the index
#     annotations = annotations[annotations.index.isin(label_vocabulary)]

#     # Add a new column 'numeric_id' to the dataframe based on the id_map
#     annotations["numeric_id"] = annotations.index.map(label2int_mapping)

#     # Sort the dataframe by 'numeric_id'
#     annotations_sorted = annotations.sort_values(by="numeric_id")

#     # Extract the "label" column as a list
#     sorted_labels = annotations_sorted["label"].tolist()

#     checkpoint = llm_checkpoint_path

#     # Load the tokenizer and tokenize the labels
#     label_tokenizer = load_HF_tokenizer(checkpoint)
#     model_inputs = tokenize_inputs(label_tokenizer, sorted_labels)

#     # Load the model
#     label_encoder = load_HF_model(
#         checkpoint, freeze_weights=not train_label_encoder
#     )
#     label_encoder = label_encoder.to(device)

#     # Move the tensors to GPU if available
#     model_inputs = {name: tensor.to(device)
#                     for name, tensor in model_inputs.items()}

#     # Create a DataLoader to iterate over the tokenized labels in batches
#     return DataLoader(TensorDataset(*model_inputs.values()), batch_size=batch_size)


def seed_everything(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
