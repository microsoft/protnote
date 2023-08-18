
from src.data.datasets import ProteinDataset
from torch.utils.data import DataLoader
from src.data.collators import collate_variable_sequence_length
import torchmetrics
import torch

#TODO: all of these paths shouldnt be here. Could use config, hydra.
TRAIN_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/train_GO.fasta'
VAL_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/dev_GO.fasta'
TEST_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO.fasta'
AMINO_ACID_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json'
GO_LABEL_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/GO_label_vocab.json'
TEST_BATCH_SIZE = 8


#TODO: Add GPU and prediction function with threshold
def compute_metric(model,dataloader,metric):
    
    model = model.eval()
    for  batch_idx, (sequence,sequences_mask,labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(sequence,sequences_mask)

        predictions = None
        metric(predictions,labels)
    return metric


train_dataset,val_dataset,test_dataset = ProteinDataset\
    .create_multiple_datasets(data_paths=[TEST_DATA_PATH],
                              sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                              label_vocabulary_path=GO_LABEL_VOCAB_PATH)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=TEST_BATCH_SIZE,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=collate_variable_sequence_length)

#For debugging
for epoch in range(1):
    for batch_idx, (sequences,sequence_masks,labels) in enumerate(test_loader):
        if batch_idx>=3:
            break
        print("Batch index:",batch_idx,end="\t")
        print("| Batch size:",labels.shape[0],end="\t")
        print("| Sequences shape:",sequences.shape,end="\t")
        print("| Sequences mask shape:",sequence_masks.shape,end="\t")
        print("| Labels shape:",labels.shape,end="\t")