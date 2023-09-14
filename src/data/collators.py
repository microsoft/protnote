import torch
from typing import List, Tuple


def collate_variable_sequence_length(batch: List[Tuple]):
    """
    Collates batches with variable sequence lengths by padding sequences to the maximum length in the batch.

    Args:
    - batch (list): List of tuples containing (sequence_id_numeric, sequence_onehot, labels_multihot, sequence_length).

    Returns:
    - processed_sequence_ids (torch.Tensor): Tensor of shape (batch_size,) containing the (numeric) sequence IDs.
    - processed_sequence_onehots (torch.Tensor): Tensor of shape (batch_size, sequence_dim, max_length) containing the padded sequences.
    - processed_label_multihots (torch.Tensor): Tensor of shape (batch_size, num_labels) containing the multihot-encoded labels.
    - processed_sequence_lengths (torch.Tensor): Tensor of shape (batch_size,) containing the sequence lengths.
    """

    # Determine the maximum sequence length in the batch
    max_length = max(item[3] for item in batch)

    processed_sequence_ids = []
    processed_sequence_onehots = []
    processed_label_multihots = []
    processed_sequence_lengths = []

    # Loop through the batch
    for sequence_id_numeric, sequence_onehots, label_multihots, sequence_length in batch:
        # Set padding
        padding_length = max_length - sequence_length

        # Get the sequence dimension (e.g., 20 for amino acids)
        sequence_dim = sequence_onehots.shape[0]

        # Pad the sequence to the max_length and append to the processed_sequences list
        processed_sequence_onehots.append(torch.cat(
            (sequence_onehots, torch.zeros((sequence_dim, padding_length))), dim=1))

        # Simply append the sequence lengths, labels, and ids to their respective lists
        processed_sequence_lengths.append(sequence_length)
        processed_label_multihots.append(label_multihots)
        processed_sequence_ids.append(sequence_id_numeric)

    # TODO: Do we really need to return the sequence lengths?
    return torch.stack(processed_sequence_ids), torch.stack(processed_sequence_onehots), torch.stack(processed_label_multihots), torch.stack(processed_sequence_lengths)


### TEMPORAL ###


def proteinfer_collate_variable_sequence_length(batch):
    '''
    Batch should have sequences, labels, and sequence lengths
    '''
    max_length = 0
    for i in batch:
        sequence, labels, sequence_length, sequence_id = i
        max_length = max(max_length, sequence_length)

    processed_sequences = []
    processed_labels = []
    sequence_lengths = []
    sequence_ids = []

    for i in batch:
        sequence, labels, sequence_length, sequence_id = i
        padding_length = max_length - sequence_length
        # TODO: This might be better with a global constant instead
        sequence_embedding_dim = sequence.shape[0]

        processed_sequences.append(torch.cat(
            (sequence, torch.zeros((sequence_embedding_dim, padding_length))), dim=1))
        sequence_lengths.append(sequence_length)
        processed_labels.append(labels)
        sequence_ids.append(sequence_id)

    return torch.stack(processed_sequences), torch.stack(sequence_lengths), torch.stack(processed_labels), torch.stack(sequence_ids)
