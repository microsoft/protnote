import torch
from torch.utils.data import Dataset
from src.utils.data import read_fasta, read_json, get_vocab_mappings, ints_to_multihot
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

    def __getitem__(self, idx):
        sequence, labels = self.data[idx]

        sequence_id, labels = labels[0], labels[1:]
        sequence_ints = [self.aminoacid2int[aa] for aa in sequence]
        
        labels_ints = [self.label2int[l] for l in labels]
        labels_multihot = ints_to_multihot(int_list=labels_ints,
                                           num_labels=self.label_vocabulary_size)

        return torch.tensor(sequence_ints,dtype=torch.int64), torch.tensor(labels_multihot,dtype=torch.int64)







