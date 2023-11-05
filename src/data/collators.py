import torch
from typing import List, Tuple
from transformers import BatchEncoding


def collate_variable_sequence_length(batch: List[Tuple], label_sample_size=None, shuffle_labels=False):
    """
    Collates batches with variable sequence lengths by padding sequences to the maximum length in the batch.

    Args:
    - batch (list): List of dictionaries containing: {
        "sequence_onehots": torch.Tensor of shape (sequence_dim, sequence_length) containing the one-hot-encoded sequence,
        "sequence_embedding": torch.Tensor of shape (sequence_embedding_dim) containing the sequence embedding,
        "sequence_length": torch.Tensor of shape (1) containing the sequence length,
        "label_multihots": torch.Tensor of shape (num_labels) containing the multihot-encoded labels,
        "tokenized_labels": list of strings containing the tokenized labels,
        "label_embeddings": torch.Tensor of shape (label_embedding_dim) containing the label embedding,
    }

    Returns:
    - processed_sequence_ids (torch.Tensor): Tensor of shape (batch_size,) containing the (numeric) sequence IDs.
    - processed_sequence_onehots (torch.Tensor): Tensor of shape (batch_size, sequence_dim, max_length) containing the padded sequences.
    - processed_label_multihots (torch.Tensor): Tensor of shape (batch_size, num_labels) containing the multihot-encoded labels.
    - processed_sequence_lengths (torch.Tensor): Tensor of shape (batch_size,) containing the sequence lengths.
    """

    # Determine the maximum sequence length in the batch
    max_length = max(item["sequence_length"] for item in batch)

    # Initialize lists to store the processed values
    processed_sequence_onehots = []
    processed_sequence_ids = []
    processed_sequence_embeddings = []
    processed_sequence_lengths = []
    processed_label_multihots = []
    processed_label_embeddings = None
    processed_tokenized_labels = None

    # Sample labels if num_sampled_labels is specified
    sampled_label_indices = None
    if label_sample_size:
        all_labels = batch[0]["label_multihots"].shape[0]
        
        sampled_label_indices = torch.randperm(all_labels)[:label_sample_size] if shuffle_labels else torch.arange(label_sample_size)

    # Apply the sampled labels to the tokenized labels and label embeddings
    tokenized_labels = batch[0]["tokenized_labels"]
    label_embeddings = batch[0]["label_embeddings"]
    if sampled_label_indices is not None:
        # Index input_ids and attention_mask with the sampled labels
        tokenized_labels_input_ids = tokenized_labels["input_ids"][sampled_label_indices]
        tokenized_labels_attention_mask = tokenized_labels["attention_mask"][sampled_label_indices]

        # Create a new BatchEncoding object with the indexed tensors
        processed_tokenized_labels = BatchEncoding({
            "input_ids": tokenized_labels_input_ids,
            "attention_mask": tokenized_labels_attention_mask
        })

        # Create a new tensor of embeddings with only the sampeld labels
        if label_embeddings is not None:
            processed_label_embeddings = label_embeddings[sampled_label_indices]
    # Otherwise, use the original tokenized labels and label embeddings
    else:
        processed_tokenized_labels = tokenized_labels
        processed_label_embeddings = label_embeddings

    # Loop through the batch
    for row in batch:
        # Get the sequence onehots, sequence embedding, sequence length, label multihots, tokenized labels, and label embedding
        sequence_onehots = row["sequence_onehots"]
        sequence_id = row["sequence_id"]
        sequence_embedding = row["sequence_embedding"]
        sequence_length = row["sequence_length"]
        label_multihots = row["label_multihots"]

        # Set padding
        padding_length = max_length - sequence_length

        # Get the sequence dimension (e.g., 20 for amino acids)
        sequence_dim = sequence_onehots.shape[0]

        # Pad the sequence to the max_length and append to the processed_sequences list
        processed_sequence_onehots.append(
            torch.cat(
                (sequence_onehots, torch.zeros((sequence_dim, padding_length))), dim=1
            )
        )

        # Use the sampled labels for each element in the batch
        if sampled_label_indices is not None:
            label_multihots = label_multihots[sampled_label_indices]

        # Append the other values to the processed lists
        if sequence_embedding is not None:
            processed_sequence_embeddings.append(sequence_embedding)
        processed_sequence_lengths.append(sequence_length)
        processed_label_multihots.append(label_multihots)
        processed_sequence_ids.append(sequence_id)

    if len(processed_sequence_embeddings) == len(processed_sequence_onehots):
        processed_sequence_embeddings = torch.stack(
            processed_sequence_embeddings)

    return {
        "sequence_onehots": torch.stack(processed_sequence_onehots),
        "sequence_ids": processed_sequence_ids,
        "sequence_embeddings": processed_sequence_embeddings,
        "sequence_lengths": torch.stack(processed_sequence_lengths),
        "label_multihots": torch.stack(processed_label_multihots),
        "tokenized_labels": processed_tokenized_labels,
        "label_embeddings": processed_label_embeddings,
    }
