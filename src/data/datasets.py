import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.data import read_fasta, read_json, get_vocab_mappings, read_pickle
from src.utils.models import tokenize_labels, get_label_embeddings
from typing import Dict
import pandas as pd
import logging
from typing import List
from src.data.collators import collate_variable_sequence_length
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
from functools import partial
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from src.utils.main_utils import get_or_generate_label_embeddings


class ProteinDataset(Dataset):
    """
    Dataset class for protein sequences with GO annotations.
    """

    def __init__(
        self,
        data_paths: dict,
        config: dict,
        vocabularies: dict,
        label_tokenizer=None,
        label_encoder=None,
        sequence_encoder=None,
        logger=None,
        subset_fraction: float = 1.0,
        deduplicate: bool = False,
        is_master: bool = True,
    ):
        """
        paths (dict): Dictionary containing paths to the data and vocabularies.
            data_path (str): Path to the FASTA file containing the protein sequences and corresponding GO annotations
            dataset_type (str): One of 'train', 'validation', or 'test'
            go_descriptions_path (str): Path to the pickled file containing the GO term descriptions mapped to GO term IDs
        deduplicate (bool): Whether to remove duplicate sequences (default: False)
        """
        # Error handling: check for missing keys and invalid dataset types
        required_keys = ["data_path", "dataset_type"]
        for key in required_keys:
            if key not in data_paths:
                raise ValueError(
                    f"Missing required key in paths dictionary: {key}")

        assert data_paths["dataset_type"] in [
            "train",
            "validation",
            "test",
        ], "dataset_type must be one of 'train', 'val', or 'test'"

        # Set the dataset type and data path
        self.dataset_type = data_paths["dataset_type"]
        self.data_path = data_paths["data_path"]

        # Set and process the vocabularies
        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["GO_label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]
        self._process_vocab()

        # Initialize class variables
        self.data = read_fasta(data_paths["data_path"])
        self.label_embedding_matrix = self.sequence_embedding_df = None

        # Subset the data if subset_fraction is provided (to improve training speed)
        if subset_fraction < 1.0:
            logging.info(
                f"Subsetting {subset_fraction*100}% of the {self.dataset_type} set..."
            )
            self.data = self.data[:int(subset_fraction * len(self.data))]

        # Deduplicate the data if deduplicate is True
        if deduplicate:
            self._remove_duplicates()

        # Load the map from alphanumeric label id to text label
        self.label_annotation_map = {key: value['label'] for key, value in read_pickle(
            data_paths["go_annotations_path"]).to_dict(orient='index').items()}

        # Create ordered list of labels
        label_text_list = []
        for label_id in self.label_vocabulary:
            label_text_list.append(self.label_annotation_map[label_id])
        self.label_text_list = label_text_list

        # Loop through the label IDs and tokenize the labels if a label tokenizer is provided
        self.tokenized_labels = None
        self.label_tokenizer = None
        if label_tokenizer is not None:
            self.label_tokenizer = label_tokenizer
            self.tokenized_labels = tokenize_labels(
                label_text_list, label_tokenizer)

        # If a label encoder is provided, encode the labels
        self.label_embedding_matrix = None
        self.label_encoder = None
        if label_encoder is not None and not config["params"]["TRAIN_LABEL_ENCODER"]:
            self.label_encoder = label_encoder
            label_embedding_matrix = get_or_generate_label_embeddings(
                label_annotations=self.label_text_list,
                label_tokenizer=label_tokenizer,
                label_encoder=label_encoder,
                label_embedding_path=config["paths"]["LABEL_EMBEDDING_PATH"],
                logger=logger,
                batch_size_limit=config["params"]["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
                is_master=is_master,
            )
            self.label_embedding_matrix = label_embedding_matrix

        # If a sequence encoder is provided, generate sequence embeddings
        # TODO: Implement this here

    # Helper functions for setting embedding dictionaries

    def set_sequence_embedding_df(self, embedding_df: pd.DataFrame):
        self.sequence_embedding_df = embedding_df

    def set_label_embedding_matrix(self, embedding_matrix: torch.Tensor):
        self.label_embedding_matrix = embedding_matrix

    def _remove_duplicates(self):
        """
        Remove duplicate sequences from self.data, keeping only the first instance of each sequence
        Use pandas to improve performance
        """

        # Convert self.data to a DataFrame
        df = pd.DataFrame(self.data, columns=["sequence", "labels"])

        # Drop duplicate rows based on the 'sequence' column, keeping the first instance
        df = df.drop_duplicates(subset="sequence", keep="first")

        # Log the number of duplicate sequences removed
        num_duplicates = len(self.data) - len(df)
        logging.info(
            f"Removing {num_duplicates} duplicate sequences from {self.data_path}...")

        # Convert the DataFrame back to the list of tuples format
        self.data = list(df.itertuples(index=False, name=None))

    # Helper functions for processing and loading vocabularies
    def _process_vocab(self):
        self._process_amino_acid_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_amino_acid_vocab(self):
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(
            self.amino_acid_vocabulary
        )

    def _process_label_vocab(self):
        self.label2int, self.int2label = get_vocab_mappings(
            self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_id2int, self.int2sequence_id = get_vocab_mappings(
            self.sequence_id_vocabulary
        )

    def __len__(self) -> int:
        return len(self.data)

    def process_example(self, sequence: str, labels: list[str]) -> dict:
        sequence_id_alphanumeric, labels = labels[0], labels[1:]

        # Convert the sequence and labels to integers for one-hot encoding
        amino_acid_ints = torch.tensor(
            [self.aminoacid2int[aa] for aa in sequence], dtype=torch.long
        )

        labels_ints = torch.tensor(
            [self.label2int[label] for label in labels], dtype=torch.long
        )

        # Get the length of the sequence
        sequence_length = torch.tensor(len(amino_acid_ints))

        # Get multi-hot encoding of sequence and labels
        sequence_onehots = torch.nn.functional.one_hot(
            amino_acid_ints, num_classes=len(self.amino_acid_vocabulary)
        ).permute(1, 0)
        label_multihots = torch.nn.functional.one_hot(
            labels_ints, num_classes=len(self.label_vocabulary)
        ).sum(dim=0)

        # Set the label embeddings, if provided
        label_embeddings = self.label_embedding_matrix if self.label_embedding_matrix is not None else None

        # Get the sequence embedding, if provided
        sequence_embedding = None
        # TODO: Remove this check
        if self.sequence_embedding_df is not None:
            sequence_embedding = torch.tensor(
                self.sequence_embedding_df.loc[sequence_id_alphanumeric].values)

        # Get the tokenized labels, if provided
        tokenized_labels = self.tokenized_labels if self.tokenized_labels is not None else None

        # Return a dict containing the processed example
        return {
            "sequence_onehots": sequence_onehots,
            "sequence_id": sequence_id_alphanumeric,
            "sequence_embedding": sequence_embedding,
            "sequence_length": sequence_length,
            "label_multihots": label_multihots,
            "tokenized_labels": tokenized_labels,
            "label_embeddings": label_embeddings,
        }

    def __getitem__(self, idx) -> tuple:
        sequence, labels = self.data[idx]
        return self.process_example(sequence, labels)

    @classmethod
    def create_multiple_datasets(
        cls,
        paths_list: List[Dict[str, str]],
        config: dict,
        vocabularies: dict,
        subset_fractions: dict = None,
        label_tokenizer=None,
        label_encoder=None,
        logger=None,
        deduplicate: bool = False,
    ) -> List[Dataset]:
        """
        paths_list (List[Dict[str, str]]): List of dictionaries, each containing paths to the data and vocabularies.
        subset_fractions (dict): Dictionary containing the subset fraction for each dataset type (default: None)
        """
        datasets = defaultdict(list)
        subset_fractions = subset_fractions or {}
        for data_paths in paths_list:
            datasets[data_paths["dataset_type"]].append(
                cls(
                    data_paths,
                    config,
                    vocabularies,
                    label_tokenizer=label_tokenizer,
                    label_encoder=label_encoder,
                    logger=logger,
                    subset_fraction=subset_fractions.get(
                        data_paths["dataset_type"], 1.0),
                    deduplicate=deduplicate
                )
            )
        return datasets


def calculate_pos_weight(data: list, num_labels: int):
    def count_labels(chunk):
        num_positive_labels_chunk = 0
        num_negative_labels_chunk = 0
        for _, labels in chunk:
            labels = labels[1:]
            num_positive = len(labels)
            num_positive_labels_chunk += num_positive
            num_negative_labels_chunk += num_labels - num_positive
        return num_positive_labels_chunk, num_negative_labels_chunk

    chunk_size = len(data) // cpu_count()  # Adjust chunk size if necessary.

    results = Parallel(n_jobs=-1)(
        delayed(count_labels)(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)
    )

    num_positive_labels = sum(res[0] for res in results)
    num_negative_labels = sum(res[1] for res in results)
    pos_weight = torch.tensor((num_negative_labels / num_positive_labels))
    return pos_weight


def calculate_label_weights(data: list):
    def count_labels(chunk):
        label_freq = Counter()
        for _, labels in chunk:
            labels = labels[1:]
            label_freq.update(labels)
        return label_freq

    # Adjust chunk size if necessary.
    chunk_size = max(len(data) // cpu_count(), 1)

    results = Parallel(n_jobs=-1)(
        delayed(count_labels)(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)
    )

    label_freq = Counter()
    for result in results:
        label_freq.update(result)

    # Inverse frequency
    total = sum(label_freq.values())
    label_inv_freq = {k: total/v for k, v in label_freq.items()}
    return label_inv_freq


def set_padding_to_sentinel(
    padded_representations: torch.Tensor,
    sequence_lengths: torch.Tensor,
    sentinel: float,
) -> torch.Tensor:
    """
    Set the padding values in the input tensor to the sentinel value.

    Parameters:
        padded_representations (torch.Tensor): The input tensor of shape (batch_size, dim, max_sequence_length)
        sequence_lengths (torch.Tensor): 1D tensor containing original sequence lengths for each sequence in the batch
        sentinel (float): The value to set the padding to

    Returns:
        torch.Tensor: Tensor with padding values set to sentinel
    """

    # Get the shape of the input tensor
    batch_size, dim, max_sequence_length = padded_representations.shape

    # Get the device of the input tensor
    device = padded_representations.device

    # Create a mask that identifies padding, ensuring it's on the same device
    mask = torch.arange(max_sequence_length, device=device).expand(
        batch_size, max_sequence_length
    ) >= sequence_lengths.unsqueeze(1).to(device)

    # Expand the mask to cover the 'dim' dimension
    mask = mask.unsqueeze(1).expand(-1, dim, -1)

    # Use the mask to set the padding values to sentinel
    padded_representations = torch.where(
        mask, sentinel, padded_representations)

    return padded_representations


def create_multiple_loaders(
    datasets: dict,
    params: dict,
    label_sample_sizes: dict = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> List[DataLoader]:
    loaders = defaultdict(list)
    for dataset_type, dataset_list in datasets.items():
        batch_size_for_type = params[f"{dataset_type.upper()}_BATCH_SIZE"]

        # Get the number of labels to sample for the current dataset type, if provided
        label_sample_size = None
        if label_sample_sizes:
            label_sample_size = label_sample_sizes.get(dataset_type)

        for dataset in dataset_list:
            loader = DataLoader(
                dataset,
                batch_size=batch_size_for_type,
                shuffle=False,
                collate_fn=partial(
                    collate_variable_sequence_length, label_sample_size=label_sample_size),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                sampler=DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True
                ),
            )
            loaders[dataset_type].append(loader)

    return loaders
