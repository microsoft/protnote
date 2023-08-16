
from src.data.datasets import ProteinDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torchtext

TRAIN_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/train_GO.fasta'
VAL_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/dev_GO.fasta'
TEST_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO.fasta'
AMINO_ACID_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json'
GO_LABEL_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/GO_label_vocab.json'
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
NUM_EPOCHS = 1
PADDING_ID = -1
#UNK_ID = -3

def collate_variable_sequence_length(batch):
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


train_dataset = ProteinDataset(data_path=TRAIN_DATA_PATH,
                               sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                               label_vocabulary_path=GO_LABEL_VOCAB_PATH)

val_dataset = ProteinDataset(data_path=VAL_DATA_PATH,
                               sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                               label_vocabulary_path=GO_LABEL_VOCAB_PATH)

test_dataset = ProteinDataset(data_path=TEST_DATA_PATH,
                               sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                               label_vocabulary_path=GO_LABEL_VOCAB_PATH)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=TRAIN_BATCH_SIZE,
                          shuffle=True,
                          num_workers=2,
                          collate_fn=collate_variable_sequence_length)

val_loader = DataLoader(dataset=val_dataset,
                          batch_size=TEST_BATCH_SIZE,
                          shuffle=False,
                          num_workers=2)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=TEST_BATCH_SIZE,
                          shuffle=False,
                          num_workers=2)

'''# Group similar length text sequences together in batches.
train_dataloader, val_dataloader,test_dataloader = torchtext.data.BucketIterator.splits(
                              # Datasets for iterator to draw data from
                              (train_dataset, val_dataset,test_dataset),
                              # Tuple of train and validation batch sizes.
                              batch_sizes=(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,TEST_BATCH_SIZE),
                              # Function to use for sorting examples.
                              sort_key=lambda x: len(x['text']),
                              # Repeat the iterator for multiple epochs.
                              repeat=True,
                              # Sort all examples in data using `sort_key`.
                              sort=False,
                              # Shuffle data on each epoch run.
                              shuffle=True,
                              # Use `sort_key` to sort examples in each batch.
                              sort_within_batch=True,
                              )
'''


for epoch in range(NUM_EPOCHS):
    for batch_idx, (sequence,sequences_mask,labels) in enumerate(train_loader):
        if batch_idx>=3:
            break
        print("Batch index:",batch_idx,end="\t")
        print("| Batch size:",labels.shape[0],end="\t")
        print("| Sequences shape:",sequence.shape,end="\t")
        print("| Sequences mask shape:",sequences_mask.shape,end="\t")
        print("| Labels shape:",labels.shape,end="\t")