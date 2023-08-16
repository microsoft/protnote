import torch
from torch.utils.data import Dataset
from src.utils.data import read_fasta, read_json, get_vocab_mappings
from typing import Optional,List,Text



class ProteinDataset(Dataset):
    def __init__(self, data_path:str, sequence_vocabulary_path: Optional[Text] = None, label_vocabulary_path: Optional[Text] = None):
        self.data_path = data_path
        self.data = read_fasta(data_path)
        self.sequence_vocabulary_path = sequence_vocabulary_path
        self.label_vocabulary_path = label_vocabulary_path
        self.max_seq_len = None
        self._process_vocab()
    
    def _process_vocab(self):
        self._process_sequence_vocab()
        self._process_label_vocab()

    def _process_sequence_vocab(self):

        if self.sequence_vocabulary_path is not None:
            self.sequence_vocabulary = read_json(self.sequence_vocabulary_path)
        else:
            self.sequence_vocabulary = set()
            for obs in self.data:
                self.sequence_vocabulary.update(list(obs[0]))
            self.sequence_vocabulary = sorted(list(self.sequence_vocabulary))
        self.sequence_vocabulary_size = len(self.sequence_vocabulary)
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(self.sequence_vocabulary)
        
    def _process_label_vocab(self):

        if self.label_vocabulary_path is not None:
            self.label_vocabulary = read_json(self.label_vocabulary_path)
        else:
            self.label_vocabulary = set()
            for obs in self.data:
                self.label_vocabulary.update(obs[1][1:])
            self.label_vocabulary = sorted(list(self.label_vocabulary))
        self.label_vocabulary_size = len(self.label_vocabulary)
        self.label2int, self.int2label = get_vocab_mappings(self.label_vocabulary)

    def get_max_seq_len(self):
        if self.max_seq_len is None:
            self.max_seq_len = max(len(i[0]) for i in self.data)
        return self.max_seq_len

    def __len__(self):
        return len(self.data)

    def process_example(self,sequence,labels):
        sequence_id, labels = labels[0], labels[1:]

        sequence_ints = torch.tensor([self.aminoacid2int[aa] for aa in sequence],dtype=torch.long)
        sequence_length = len(sequence_ints)
        labels_ints =torch.tensor([self.label2int[l] for l in labels],dtype=torch.long)

        sequence_onehots = torch.nn.functional.one_hot(sequence_ints,num_classes =self.sequence_vocabulary_size ).permute(1,0)        
        labels_multihot =  torch.nn.functional.one_hot(labels_ints,num_classes = self.label_vocabulary_size ).sum(dim=0)

        return sequence_onehots, labels_multihot,sequence_length

    def __getitem__(self, idx):
        sequence, labels = self.data[idx]
        return self.process_example(sequence,labels)







