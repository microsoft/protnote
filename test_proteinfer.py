
from src.data.datasets import ProteinDataset
from torch.utils.data import DataLoader
from src.data.collators import collate_variable_sequence_length
from src.models.protein_encoders import Residual,ProteInfer
from torchmetrics.classification import BinaryPrecision,BinaryRecall
import torch
from tqdm import tqdm
import logging

#TODO: all of these paths shouldnt be here. Could use config, hydra.
TRAIN_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/train_GO.fasta'
VAL_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/dev_GO.fasta'
TEST_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO.fasta'
AMINO_ACID_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json'
GO_LABEL_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/GO_label_vocab.json'
NUM_LABELS = 32102
TEST_BATCH_SIZE = 8
DECISION_TH =  0.99
DEBUG = False

logging.basicConfig(filename='myapp.log', level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device = {device}")


#TODO: Add GPU and prediction function with threshold
def compute_metric(model,dataloader,metric):
    
    model = model.eval()
    for  batch_idx, (sequence,sequences_mask,labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(sequence,sequences_mask)

        predictions = None
        metric(predictions,labels)
    return metric


val_dataset,test_dataset = ProteinDataset\
    .create_multiple_datasets(data_paths=[VAL_DATA_PATH,TEST_DATA_PATH],
                              sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                              label_vocabulary_path=GO_LABEL_VOCAB_PATH)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=TEST_BATCH_SIZE,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=collate_variable_sequence_length)

model = ProteInfer(num_labels=NUM_LABELS,
                      input_channels=20,
                      output_channels=1100,
                      kernel_size=9,
                      activation=torch.nn.ReLU,
                      dilation_base=3,
                      num_resnet_blocks=5,
                      bottleneck_factor=0.5)


model.to(device)

precisions = []
recalls = []
at_least_one_positive_pred = torch.tensor(0,dtype=int).to(device)#seqs with at least one positive label prediction
n = torch.tensor(0,dtype=int).to(device)
seqwise_precision = BinaryPrecision(threshold = DECISION_TH,
                                    multidim_average='samplewise').to(device)
seqwise_recall = BinaryRecall(threshold = DECISION_TH,
                            multidim_average='samplewise').to(device)
#For debugging
with torch.no_grad():
    for batch_idx, (sequences,sequence_lengths,labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
        sequences,sequence_lengths,labels = (sequences.to(device),
                                        sequence_lengths.to(device),
                                        labels.to(device))
        
        n+=len(labels)
        logits = model(sequences,sequence_lengths)
        probabilities = torch.sigmoid(logits)

        at_least_one_positive_pred+=(probabilities>DECISION_TH).any(axis=1).sum()
        seqwise_precision(probabilities,labels)
        seqwise_recall(probabilities,labels)
        
        if DEBUG:
            if batch_idx>=3:
                break
            print("Batch index:",batch_idx,end="\t")
            print("| Batch size:",labels.shape[0],end="\t")
            print("| Sequences shape:",sequences.shape,end="\t")
            print("| Sequences mask shape:",sequence_lengths.shape,end="\t")
            print("| Labels shape:",labels.shape,end="\t")
    
    average_precision = seqwise_precision.compute().sum()/at_least_one_positive_pred
    average_recall = seqwise_recall.compute().mean()
    average_f1_score = 2*average_precision*average_recall/(average_precision+average_recall)
    print(average_f1_score,average_precision,average_recall)
