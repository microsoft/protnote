from torch.utils.data import Dataset
from src.utils.data import read_fasta
from src.utils.proteins import RESIDUE_TO_INT, INT_TO_RESIDUE
from typing import Optional,List

class ProteinDataset(Dataset):
    def __init__(self, data_path:str, vocabulary: Optional[List] = None):
        self.data_path = data_path
        self.data = read_fasta(data_path)
        self.vocabulary = vocabulary
        self.max_seq_len = None
        self._process_vocab()

    def _process_vocab(self):
        if self.vocabulary is None:
            self.aa2int = RESIDUE_TO_INT
            self.int2aa = INT_TO_RESIDUE
        else:
            self.aa2int = {aa: idx for idx, aa in enumerate(self.vocabulary)}
            self.int2aa = {idx:aa for aa,idx in self.aa2int.items()}

    def get_max_seq_len(self):
        if self.max_seq_len is None:
            self.max_seq_len = max(len(i[0]) for i in self.data)
        return self.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, labels = self.data[idx]
        sequence_id, labels = labels[0], labels[1:]
        sequence = [self.aa2int[aa] for aa in sequence]

        return sequence, labels






