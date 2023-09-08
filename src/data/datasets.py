import torch
from torch.utils.data import Dataset
from src.utils.data import read_fasta, read_json, get_vocab_mappings, filter_annotations
from typing import Optional, Text
import pandas as pd
import logging
from functools import partial
from typing import List
from src.data.collators import collate_variable_sequence_length


class ProteinDataset(Dataset):
    """
    Dataset class for protein sequences with GO annotations.
    Reads protein sequences and GO annotations from a FASTA file and a CSV file, respectively.
    """

    def __init__(self, data_path: str, allowed_annotations_path: str, sequence_vocabulary_path: Optional[Text] = None,
                 label_vocabulary_path: Optional[Text] = None, train_sequence_encoder: bool = False, ):
        # Initialize class variables
        self.data_path = data_path
        self.data = read_fasta(data_path)
        self.sequence_vocabulary_path = sequence_vocabulary_path
        self.label_vocabulary_path = label_vocabulary_path
        self.max_seq_len = None
        self.train_sequence_encoder = train_sequence_encoder

        # Load raw data
        raw_data = read_fasta(data_path)

        # Remove all labels that are not present in the source-of-truth (e.g., EC annotations, annotations with "-",
        # obsolete annotations, or remapped annotations)
        allowed_annotations = set(pd.read_csv(
            allowed_annotations_path)['go_id'])
        self.data = filter_annotations(raw_data, allowed_annotations)

        # Generate vocabularies
        # TODO: Explore whether we can add back or remap some of the GO annotations that were changed
        self._process_vocab()
        logging.info(f"Loaded {len(self.data)} sequences from {data_path}.")

    def _process_vocab(self):
        self._process_sequence_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_sequence_vocab(self):

        if self.sequence_vocabulary_path is not None:
            self.sequence_vocabulary = read_json(self.sequence_vocabulary_path)
        else:
            logging.warning(
                "No sequence vocabulary path given; building sequence vocabulary from scratch...")
            self.sequence_vocabulary = set()
            for obs in self.data:
                self.sequence_vocabulary.update(list(obs[0]))
            self.sequence_vocabulary = sorted(list(self.sequence_vocabulary))
        self.sequence_vocabulary_size = len(self.sequence_vocabulary)
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(
            self.sequence_vocabulary)

    def _process_label_vocab(self):

        if self.label_vocabulary_path is not None:
            self.label_vocabulary = read_json(self.label_vocabulary_path)
        else:
            logging.warning(
                "No label vocabulary path given; building label vocabulary from scratch...")
            self.label_vocabulary = set()
            for obs in self.data:
                self.label_vocabulary.update(obs[1][1:])
            self.label_vocabulary = sorted(list(self.label_vocabulary))
        self.label_vocabulary_size = len(self.label_vocabulary)
        self.label2int, self.int2label = get_vocab_mappings(
            self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_ids = set()
        for obs in self.data:
            self.sequence_ids.add(obs[1][0])
        self.sequence_ids = sorted(list(self.sequence_ids))
        self.sequence_id2int = {term: idx for idx,
                                term in enumerate(self.sequence_ids)}
        self.int2sequence_id = {idx: term for term,
                                idx in self.sequence_id2int.items()}

    def get_max_seq_len(self):
        if self.max_seq_len is None:
            self.max_seq_len = max(len(i[0]) for i in self.data)
        return self.max_seq_len

    def __len__(self):
        return len(self.data)

    def process_example(self, sequence, labels):
        sequence_id, labels = labels[0], labels[1:]

        # Get one-hot encoding of sequence and multi-hot encoding of labels labels
        sequence_ints = torch.tensor(
            [self.aminoacid2int[aa] for aa in sequence], dtype=torch.long)
        sequence_length = torch.tensor(len(sequence_ints))
        labels_ints = torch.tensor([self.label2int[l]
                                   for l in labels], dtype=torch.long)
        sequence_onehots = torch.nn.functional.one_hot(
            sequence_ints, num_classes=self.sequence_vocabulary_size).permute(1, 0).float()
        labels_multihot = torch.nn.functional.one_hot(
            labels_ints, num_classes=self.label_vocabulary_size).sum(dim=0).float()

        if self.train_sequence_encoder:
            return sequence_onehots, labels_multihot, sequence_length
        else:
            # TODO: Return sequence_id instead of sequence
            return sequence, labels_multihot, sequence_length, sequence_id

    def __getitem__(self, idx):
        sequence, labels = self.data[idx]
        return self.process_example(sequence, labels)

    @classmethod
    def create_multiple_datasets(cls,
                                 data_paths: list,
                                 allowed_annotations_path: str,
                                 sequence_vocabulary_path: Optional[Text] = None,
                                 label_vocabulary_path: Optional[Text] = None,
                                 train_sequence_encoder: bool = False):
        return [cls(data_path, allowed_annotations_path, sequence_vocabulary_path, label_vocabulary_path, train_sequence_encoder) for data_path in data_paths]


def set_padding_to_sentinel(padded_representations, sequence_lengths, sentinel):
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
        batch_size, max_sequence_length) >= sequence_lengths.unsqueeze(1).to(device)

    # Expand the mask to cover the 'dim' dimension
    mask = mask.unsqueeze(1).expand(-1, dim, -1)

    # Use the mask to set the padding values to sentinel
    padded_representations[mask] = sentinel

    return padded_representations


def create_multiple_loaders(datasets: List[torch.utils.data.Dataset],
                            batch_sizes: List[int],
                            num_workers=int,
                            train_sequence_encoder: bool = False,
                            ) -> List[torch.utils.data.DataLoader]:
    loaders = []
    for dataset, batch_size in zip(datasets, batch_sizes):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_variable_sequence_length,
                train_sequence_encoder=train_sequence_encoder
            ),
            num_workers=num_workers
        )
        loaders.append(loader)
    return loaders
