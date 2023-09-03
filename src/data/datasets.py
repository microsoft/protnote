import torch
from torch.utils.data import Dataset
from src.utils.data import read_fasta, read_json, get_vocab_mappings, read_pickle, filter_annotations 
from typing import Optional,List,Text
import os
import numpy as np
import pandas as pd
import logging

# Initialize logging
logger = logging.getLogger(__name__)

class ProteinDataset(Dataset):
    def __init__(self, data_path:str, sequence_vocabulary_path: Optional[Text] = None, label_vocabulary_path: Optional[Text] = None, sequence_embedding_path: Optional[Text] = None, label_embedding_path: Optional[Text] = None):
        # Initialize class variables
        self.data_path = data_path
        self.sequence_vocabulary_path = sequence_vocabulary_path
        self.label_vocabulary_path = label_vocabulary_path
        self.max_seq_len = None
        self.sequence_embedding_path = sequence_embedding_path or os.path.join(os.path.dirname(data_path), 'sequence_embeddings.pk1')  
        self.label_embedding_path = label_embedding_path or os.path.join(os.path.dirname(data_path), 'label_embeddings.pk1')  

        # Load raw data
        logging.info(f"Loading data from {data_path}...")
        raw_data = read_fasta(data_path)

        # Remove all labels that are not present in the source-of-truth
        allowed_annotations =  set(pd.read_csv('/home/ncorley/protein/ProteinFunctions/data/go_annotations.csv')['go_id'])
        logging.info("Filtering annotations...")
        self.data = filter_annotations(raw_data, allowed_annotations)
        logging.info(f"Loaded {len(self.data)} sequences.")

        # Generate vocabularies
        # TODO: Explore whether we can add back or remap some of the GO annotations that were changed
        logging.info("Processing vocabularies...")
        self._process_vocab()

        # Load embeddings into class variables for future reference
        logging.info("Loading embeddings...")
        self._load_embeddings()
        
        logging.info("Finished initializing ProteinDataset.")
    
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
        

    def _load_embeddings(self):
        # Process sequence embeddings
        if os.path.exists(self.sequence_embedding_path):
            self.sequence_embeddings = read_pickle(self.sequence_embedding_path)[['sequence', 'embedding']].set_index('sequence')['embedding'].to_dict()
        else:
            # Alert the user that the sequence embeddings need to be pre-processed
            logging.error("Sequence embeddings need to be pre-processed to increase efficiency. Please run the script embed_sequences.py.")
            self.sequence_embeddings = None

        # Process label embeddings
        if os.path.exists(self.label_embedding_path):
            self.label_embeddings = read_pickle(self.label_embedding_path)[['go_id', 'embedding']].set_index('go_id')['embedding'].to_dict()
        else:
            # Alert the user that the label embeddings need to be pre-processed
            logging.error("Label embeddings need to be pre-processed to increase efficiency. Please run the script embed_go_annotations.py.")
            self.label_embeddings = None
    
    def _lookup_label_embeddings(self, labels):
        """
        Method to get label embeddings from list of labels
        """
        embeddings_list = []
        
        for label in labels:
            embedding = self.label_embeddings.get(label)
            if embedding is not None:
                embeddings_list.append(embedding)
            else:
                logging.error(f"No embedding found for label: {label}")
                return None
                 
        embeddings_array = np.array(embeddings_list)
        label_embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        
        return label_embeddings

    def _lookup_sequence_embeddings(self, sequence):
        """
        Method to get sequence embeddings for a single sequence.
        """
        embedding = self.sequence_embeddings.get(sequence)
        
        if embedding is not None:
            return torch.tensor(embedding)
        else:
            logging.error(f"No embedding found for sequence: {sequence}")
            return None

    def get_max_seq_len(self):
        if self.max_seq_len is None:
            self.max_seq_len = max(len(i[0]) for i in self.data)
        return self.max_seq_len

    def __len__(self):
        return len(self.data)

    def process_example(self, sequence, labels):
        sequence_id, labels = labels[0], labels[1:]

        # Print out all the labels that are not in the label2int dictionary
        print(f"Labels not in label2int dictionary BEFORE: {set(labels) - set(self.label2int.keys())}")

        # Remove the labels that are not in the label2int dictionary
        # TODO: It would be more efficient to remove these upfront, either on __init__ or in the raw data file
        label_set = set(self.label2int.keys())
        labels = [l for l in labels if l in label_set]

        # Print out all the labels that are not in the label2int dictionary
        print(f"Labels not in label2int dictionary AFTER: {set(labels) - set(self.label2int.keys())}")
    
        # Get one-hot encoding of sequence and multi-hot encoding of labels labels
        sequence_ints = torch.tensor([self.aminoacid2int[aa] for aa in sequence],dtype=torch.long)
        sequence_length = torch.tensor(len(sequence_ints))
        labels_ints = torch.tensor([self.label2int[l] for l in labels],dtype=torch.long)  
        sequence_onehots = torch.nn.functional.one_hot(sequence_ints,num_classes = self.sequence_vocabulary_size ).permute(1,0)        
        labels_multihot =  torch.nn.functional.one_hot(labels_ints,num_classes = self.label_vocabulary_size ).sum(dim=0)

        # TODO: Lookup sequence embedding
        sequence_embedding = self._lookup_sequence_embeddings(sequence)

        # Lookup the label embeddings
        label_embeddings = self._lookup_label_embeddings(labels)

        return sequence_onehots, labels_multihot, sequence_length, sequence_embedding, label_embeddings

    def __getitem__(self, idx):
        print("Getting item...")
        sequence, labels = self.data[idx]
        print(f"Sequence: {sequence}")
        print(f"Labels: {labels}")
        return self.process_example(sequence,labels)
    
    @classmethod
    def create_multiple_datasets(cls,
                                 data_paths:list,
                                 sequence_vocabulary_path: str,
                                 label_vocabulary_path: Optional[Text] = None):
        return [cls(data_path,sequence_vocabulary_path,label_vocabulary_path) for data_path in data_paths]


def set_padding_to_sentinel(padded_representations, sequence_lengths, sentinel):
    # Create a sequence mask
    seq_mask = torch.arange(padded_representations.size(1)).to(sequence_lengths.device) < sequence_lengths[:, None]

    #Sentinel and padded_representations should be same dtype
    sentinel = torch.tensor(sentinel, dtype=padded_representations.dtype, device=padded_representations.device)
                            
    # Use broadcasting to expand the mask to match the shape of padded_representations
    return torch.where(seq_mask.unsqueeze(-1), padded_representations, sentinel)