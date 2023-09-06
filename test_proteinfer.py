
from src.data.datasets import ProteinDataset
from torch.utils.data import DataLoader
from src.data.collators import collate_variable_sequence_length
from src.models.protein_encoders import ProteInfer
from src.utils.proteinfer import transfer_tf_weights_to_torch
from torchmetrics.classification import BinaryPrecision,BinaryRecall
from src.utils.data import read_json,load_gz_json
from src.utils.proteinfer import normalize_confidences
import torch
import numpy as np
from tqdm import tqdm
import logging

#TODO: all of these paths shouldnt be here. Could use config, hydra.
TRAIN_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/train_GO.fasta'
VAL_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/dev_GO.fasta'
TEST_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO.fasta'
AMINO_ACID_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json'
GO_LABEL_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/GO_label_vocab.json'
MODEL_WIEGHTS_PATH = '/home/samirchar/ProteinFunctions/models/proteinfer/GO_model_weights.pkl'
PARENTHOOD_LIB_PATH = '/home/samirchar/ProteinFunctions/parenthood.json.gz'
NUM_LABELS = 32102
TEST_BATCH_SIZE = 2**7
DEBUG = False
DECISION_TH = 0.710102#0.88

logging.basicConfig( level=logging.INFO)

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


if DEBUG:
    test_dataset = ProteinDataset(data_path=TRAIN_DATA_PATH,
                                sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                                label_vocabulary_path=GO_LABEL_VOCAB_PATH)
    
    test_dataset.data = [i for i in test_dataset.data if i[1][0] in ['P69891','Q7AP54']]
else:
    val_dataset,test_dataset = ProteinDataset\
        .create_multiple_datasets(data_paths=[VAL_DATA_PATH,TEST_DATA_PATH],
                                sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                                label_vocabulary_path=GO_LABEL_VOCAB_PATH)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=TEST_BATCH_SIZE,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=collate_variable_sequence_length)

val_loader = DataLoader(dataset=val_dataset,
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

transfer_tf_weights_to_torch(model,MODEL_WIEGHTS_PATH)
model.to(device)
model = model.eval()

vocab = read_json(GO_LABEL_VOCAB_PATH)
label_normalizer = load_gz_json(PARENTHOOD_LIB_PATH)

if DECISION_TH is None:
    val_probas =[]
    val_labels = []
    with torch.no_grad():
        for batch_idx, (sequences,sequence_lengths,labels) in tqdm(enumerate(val_loader),total=len(val_loader)):
            sequences,sequence_lengths,labels = (sequences.to(device),
                                            sequence_lengths.to(device),
                                            labels.to(device))
            
            logits = model(sequences,sequence_lengths)
            probabilities = torch.sigmoid(logits)
            probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                label_vocab=vocab,
                                applicable_label_dict=label_normalizer),device=probabilities.device)

            
            val_probas.append(probabilities)
            val_labels.append(labels)

    val_probas = torch.cat(val_probas)
    val_labels = torch.cat(val_labels)
    best_th = 0
    best_f1 = 0

    for th in np.arange(0.1,1,0.01):
        seqwise_precision = BinaryPrecision(threshold = th,
                                            multidim_average='samplewise').to(device)
        seqwise_recall = BinaryRecall(threshold = th,
                                    multidim_average='samplewise').to(device)
        at_least_one_positive_pred=(val_probas>th).any(axis=1).sum()

        seqwise_precision(val_probas,val_labels)
        seqwise_recall(val_probas,val_labels)
        average_precision = seqwise_precision.compute().sum()/at_least_one_positive_pred
        average_recall = seqwise_recall.compute().mean()
        average_f1_score = 2*average_precision*average_recall/(average_precision+average_recall)
        if average_f1_score>best_f1:
            best_f1 = average_f1_score
            best_th = th
        print('TH:',th,'F1:',average_f1_score)
    print('Best Val F1:',best_f1,'Best Val TH:',best_th)
    DECISION_TH =   best_th

at_least_one_positive_pred = torch.tensor(0,dtype=int).to(device)#seqs with at least one positive label prediction
n = torch.tensor(0,dtype=int).to(device)
seqwise_precision = BinaryPrecision(threshold = DECISION_TH,
                                    multidim_average='samplewise').to(device)
seqwise_recall = BinaryRecall(threshold = DECISION_TH,
                            multidim_average='samplewise').to(device)


all_probas = []
all_labels = []
with torch.no_grad():
    for batch_idx, (sequences,sequence_lengths,labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
        sequences,sequence_lengths,labels = (sequences.to(device),
                                        sequence_lengths.to(device),
                                        labels.to(device))
        
        n+=len(labels)
        logits = model(sequences,sequence_lengths)
        probabilities = torch.sigmoid(logits)
        probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                            label_vocab=vocab,
                            applicable_label_dict=label_normalizer),device=probabilities.device)
        all_probas.append(probabilities)
        all_labels.append(labels)
        at_least_one_positive_pred+=(probabilities>DECISION_TH).any(axis=1).sum()
        seqwise_precision(probabilities,labels)
        seqwise_recall(probabilities,labels)
        
        if DEBUG:
            print("Batch index:",batch_idx,end="\t")
            print("| Batch size:",labels.shape[0],end="\t")
            print("| Sequences shape:",sequences.shape,end="\t")
            print("| Sequences mask shape:",sequence_lengths.shape,end="\t")
            print("| Labels shape:",labels.shape,end="\t")
    
    average_precision = seqwise_precision.compute().sum()/at_least_one_positive_pred
    print('raw average precision:',seqwise_precision.compute().mean())
    average_recall = seqwise_recall.compute().mean()
    average_f1_score = 2*average_precision*average_recall/(average_precision+average_recall)
    coverage = at_least_one_positive_pred/n

    print(average_f1_score,average_precision,average_recall,coverage)

    all_probas = torch.cat(all_probas)
    all_labels = torch.cat(all_labels)

torch.cuda.empty_cache()
