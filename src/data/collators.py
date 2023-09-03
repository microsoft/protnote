import torch

def collate_variable_sequence_length(batch):
    '''
    Batch should have sequences, labels, sequence lengths, sequence embeddings, and label embeddings
    '''
    # Get the maximum sequence length and number of labels in the batch
    max_sequence_length = 0
    for i in batch:
        sequence, labels, sequence_length = i
        #  sequence, labels, sequence_length, _, _ = i
        max_sequence_length = max(max_sequence_length, sequence_length)

    # Pad the sequences and labels to the maximum length, and copy over the sequence lengths, labels, and embeddings
    processed_sequences = []
    processed_labels = []
    sequence_lengths = []
    # processed_sequence_embeddings = []
    # processed_label_embeddings = []

    for i in batch:
        sequence, labels, sequence_length = i
        #  sequence, labels, sequence_length, sequence_embedding, label_embeddings = i

        # Get padding length for sequence and labels
        sequence_padding_length = max_sequence_length - sequence_length

        sequence_embedding_dim = sequence.shape[0]  # This might be better with a global constant instead

        # Pad the sequences and labels
        processed_sequences.append(torch.cat((sequence, torch.zeros((sequence_embedding_dim, sequence_padding_length))), dim=1))
        sequence_lengths.append(sequence_length)
        processed_labels.append(labels)
        # processed_sequence_embeddings.append(sequence_embedding)
        # processed_label_embeddings.append(label_embeddings)

    return (torch.stack(processed_sequences),
            torch.stack(sequence_lengths),
            torch.stack(processed_labels))
            # torch.stack(processed_sequence_embeddings),
            # torch.stack(processed_label_embeddings))