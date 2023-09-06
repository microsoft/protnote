import torch

def proteinfer_collate_variable_sequence_length(batch):
    '''
    Batch should have sequences, labels, and sequence lengths
    '''
    max_length = 0
    for i in batch:
        sequence,labels,sequence_length = i
        max_length = max(max_length,sequence_length)
    
    processed_sequences = []
    processed_labels = []
    sequence_lengths = []

    for i in batch:
        sequence,labels,sequence_length = i
        padding_length = max_length - sequence_length
        sequence_embedding_dim= sequence.shape[0]#TODO: This might be better with a global constant instead

        processed_sequences.append(torch.cat((sequence,torch.zeros((sequence_embedding_dim,padding_length))),dim=1))
        sequence_lengths.append(sequence_length)
        processed_labels.append(labels)

    return torch.stack(processed_sequences),torch.stack(sequence_lengths),torch.stack(processed_labels)
