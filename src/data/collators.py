import torch


def collate_variable_sequence_length(batch, train_sequence_encoder):
    """
    Collates batches with variable sequence lengths by padding sequences to the maximum length in the batch.

    Args:
    - batch (list): List of tuples containing sequences (either multi-hot or), labels, and sequence lengths.
    - train_sequence_encoder (bool): Indicates whether the sequence encoder is being trained

    Returns:
    - Tensor or list: One-hot encoded sequences OR sequence IDs (depending on train_sequence_encoder).
    - Tensor: Sequence lengths.
    - Tensor: Multi-hot encoded labels.
    - Tensor: Target labels after collapsing.
    """

    # Determine the maximum sequence length in the batch
    max_length = max(item[2] for item in batch)

    processed_sequences = []
    processed_labels = []
    sequence_lengths = []
    sequence_ids = []

    # Loop through the batch
    for sequence, labels, sequence_length, sequence_id in batch:
        # Set padding
        padding_length = max_length - sequence_length

        # If training the sequence encoder, pad the one-hot encoded sequences
        if train_sequence_encoder:
            sequence_embedding_dim = sequence.shape[0]
            # TODO: This might be better with a global constant instead
            processed_sequences.append(torch.cat(
                (sequence, torch.zeros((sequence_embedding_dim, padding_length))), dim=1))
        # Otherwise, just append the sequence
        else:
            processed_sequences.append(sequence)
        sequence_lengths.append(sequence_length)
        processed_labels.append(labels)
        sequence_ids.append(sequence_id)

    # Convert to tensors
    labels_tensor = torch.stack(processed_labels)
    sequence_lengths_tensor = torch.tensor(sequence_lengths)

    # TODO: do something with this tensor
    # sequence_ids = torch.tensor(sequence_ids)

    # TODO: Use sequence ID's rather than full strings

    # Compute collapsed labels and target tensor for the batch
    collapsed_labels = torch.any(labels_tensor, dim=0)
    target_tensor = labels_tensor[:, collapsed_labels]

    if train_sequence_encoder:
        return torch.stack(processed_sequences), sequence_lengths_tensor, labels_tensor, target_tensor
    else:
        # TODO: Return sequence_id instead of sequence
        return processed_sequences, sequence_lengths_tensor, labels_tensor, target_tensor
