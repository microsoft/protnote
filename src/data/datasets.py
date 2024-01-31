import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.data import read_fasta, read_json, get_vocab_mappings, read_pickle
from src.utils.models import tokenize_labels, get_label_embeddings
from typing import Dict
import pandas as pd
import numpy as np
import logging
from typing import List
from src.data.collators import collate_variable_sequence_length
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
from functools import partial
from collections import Counter
from src.data.samplers import GridBatchSampler,observation_sampler_factory
import random
import math
import blosum as bl

class ProteinDataset(Dataset):
    """
    Dataset class for protein sequences with GO annotations.
    """
    def __init__(
        self,
        data_paths: dict,
        config: dict,
        vocabularies: dict,
        logger=None,
        require_label_idxs=False,
        subset_fraction: float = 1.0,
        deduplicate: bool = False,
        label_tokenizer=None,
    ):
        """
        data_paths (dict): Dictionary containing paths to the data and vocabularies.
            data_path (str): Path to the FASTA file containing the protein sequences and corresponding GO annotations
            dataset_type (str): One of 'train', 'validation', or 'test'
            go_descriptions_path (str): Path to the pickled file containing the GO term descriptions mapped to GO term IDs
        deduplicate (bool): Whether to remove duplicate sequences (default: False)
        """
        self.logger = logger
        # Error handling: check for missing keys and invalid dataset types
        required_keys = ["data_path", "dataset_type"]
        for key in required_keys:
            if key not in data_paths:
                raise ValueError(
                    f"Missing required key in paths dictionary: {key}")

        assert data_paths["dataset_type"] in [
            "train",
            "validation",
            "test",
        ], "dataset_type must be one of 'train', 'val', or 'test'"
        
        # Tokenizer
        self.label_tokenizer = label_tokenizer

        # Set the dataset type and data path
        self.dataset_type = data_paths["dataset_type"]
        self.data_path = data_paths["data_path"]

        # Set and process the vocabularies
        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["GO_label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]
        self._process_vocab()
        
        # Initialize data augmentation parameters
        self.masked_msa_token = "<MASK>"
        self.augment_sequence_probability = config["params"]["AUGMENT_SEQUENCE_PROBABILITY"]
        self.use_residue_masking = config["params"]["USE_RESIDUE_MASKING"]
        self.augment_labels = config["params"]["AUGMENT_LABELS_WITH"] is not None
        self.augment_labels_with = config["params"]["AUGMENT_LABELS_WITH"].split('+') if self.augment_labels else []
        
        # Load the BLOSUM62 matrix and convert to defaultdict using dictionary comprehension
        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(dict, {aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()} for aa1 in blosum62.keys()})
        
        # Initialize class variables and pre-computed embedding matrices
        self.data = read_fasta(data_paths["data_path"])
        self.label_embedding_matrix = self.sequence_embedding_df = None
        
        # Flag to know how Dataset indexing will be handled
        self.require_label_idxs = require_label_idxs

        # Subset the data if subset_fraction is provided
        if subset_fraction < 1.0:
            logging.info(
                f"Subsetting {subset_fraction*100}% of the {self.dataset_type} set..."
            )
            self.data = self.data[:int(subset_fraction * len(self.data))]

        # Deduplicate the data if deduplicate is True
        self._clean_data(deduplicate=deduplicate,
            max_sequence_length=config["params"]["MAX_SEQUENCE_LENGTH"])

        '''
        # Load the map from alphanumeric label id to text label
        self.temp = {key: value[config["params"]['GO_DESCRIPTION_TYPE']] for key, value in read_pickle(
            data_paths["go_annotations_path"]).to_dict(orient='index').items()}

        self.label_annotation_map={}
        for key, value in read_pickle(data_paths["go_annotations_path"]).to_dict(orient='index').items():
            self.label_annotation_map[key] = [value[config["params"]['GO_DESCRIPTION_TYPE']]] + value['synonym_exact'] 
        '''

        self.label_annotation_map = self.create_label_annotation_map(go_annotations_path = data_paths["go_annotations_path"],
                                                                     go_description_type = config["params"]['GO_DESCRIPTION_TYPE'],
                                                                     augment_with = self.augment_labels_with
                                                                     )
    @staticmethod
    def create_label_annotation_map(go_annotations_path: str,
                                    go_description_type:str,
                                    augment_with: list):
        def ensure_list(value):
            # Case 1: If the value is already a list
            if isinstance(value, list):
                return value
            # Case 2: If the value is NaN
            elif value is math.nan or (isinstance(value, float) and math.isnan(value)):
                return []
            # Case 3: For all other cases (including strings)
            else:
                return [value]
        
        assert go_description_type not in augment_with, f'''Can't include {go_description_type} in AUGMENT_LABELS_WITH 
                                                            because this is already the base description set by GO_DESCRIPTION_TYPE.
                                                            Doing this would yield to "double counting" of descriptions of type {go_description_type}
                                                        '''
        label_annotation_map={}
        c=1
        for key, value in read_pickle(go_annotations_path).to_dict(orient='index').items():
            label_annotation_map[key] = []            
            
            #Add descriptionts from main column and augmentation in predictable order.
            for column in sorted(augment_with+[go_description_type]):
                label_annotation_map[key]+=ensure_list(value[column] )

        return label_annotation_map

        
    # Helper functions for setting embedding dictionaries
    def set_sequence_embedding_df(self, embedding_df: pd.DataFrame):
        self.sequence_embedding_df = embedding_df

    def set_label_embedding_specs(self,
                                  label_embedding_matrix: torch.Tensor,
                                  label_annotation_map_idxs : list,
                                  base_label_idxs: list):
        """Sets the embedding matrix (2D Tensor) that store embeddings of all GO descriptions with 
        their possible variants. If a label has multiple possible descriptions their embeddings
        are flattened to fit the 2D embedding matrix.

        Separately receives a mapping that stores the range of indexes where the possible descriptions
        of a go label lie in the flattened array. This is used to recover the dataset from its flattened version


        :param label_embedding_matrix: Matrix that stores the embeddings of all possible label descriptions (with synonyms/augmentations)
        :type label_embedding_matrix: torch.Tensor
        :param label_annotation_map_idxs:  mapping that stores the range of indexes where the possible descriptions
        of a go label lie in the flattened array. This is used to recover the dataset from its flattened version
        :type label_annotation_map_idxs: list
        :param base_label_idxs: list that stores the indexes from embedding matrix corresponding to the original base label descriptions 
        :type base_label_idxs: list
        """        
        
        self.label_embedding_matrix = label_embedding_matrix
        self.label_annotation_map_idxs = label_annotation_map_idxs
        self.base_label_idxs = base_label_idxs
        
    def _clean_data(self,deduplicate,max_sequence_length):
        """
        Remove duplicate sequences from self.data, keeping only the first instance of each sequence
        Use pandas to improve performance
        """
        
        # Convert self.data to a DataFrame
        df = pd.DataFrame(self.data, columns=["sequence", "labels"])

        if deduplicate:
            # Drop duplicate rows based on the 'sequence' column, keeping the first instance
            df = df.drop_duplicates(subset="sequence", keep="first")

            # Log the number of duplicate sequences removed
            num_duplicates = len(self.data) - len(df)
            logging.info(
                f"Removing {num_duplicates} duplicate sequences from {self.data_path}...")
        
        if (max_sequence_length is not None) & (self.dataset_type == "train"):
            
            seq_length_mask = df["sequence"].apply(len)<=max_sequence_length
            num_long_sequences = (~seq_length_mask).sum()
            df = df[seq_length_mask]
            logging.info(
                f"Removing {num_long_sequences} sequences longer than {max_sequence_length} from {self.data_path}...")

        # Convert the DataFrame back to the list of tuples format
        self.data = list(df.itertuples(index=False, name=None))

    # Helper functions for processing and loading vocabularies
    def _process_vocab(self):
        self._process_amino_acid_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_amino_acid_vocab(self):
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(
            self.amino_acid_vocabulary
        )

    def _process_label_vocab(self):
        self.label2int, self.int2label = get_vocab_mappings(
            self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_id2int, self.int2sequence_id = get_vocab_mappings(
            self.sequence_id_vocabulary
        )

    def __len__(self) -> int:
        return len(self.data)
    
    def _sample_based_on_blosum62(self, amino_acid: str) -> str:
        """
        Sample an amino acid based on the BLOSUM62 substitution matrix, favoring likely silent mutations. 
        In most cases, the amino acid will be unchanged.
        Args:
            amino_acid (str): The amino acid to find a substitution for.
        Returns:
            str: The substituted amino acid.
        """
        # Get the substitutions for the amino acid, ensuring only amino acids within the vocabulary are considered
        substitutions = self.blosum62[amino_acid]
        substitutions = {aa: score for aa, score in substitutions.items() if aa in self.amino_acid_vocabulary}
        amino_acids, scores = zip(*substitutions.items())
        
        # Use only non-negative scodes
        probabilities = [max(0, score) for score in scores]
        total = sum(probabilities)
        
        # If all scores are negative, do not change the amino acid
        if total == 0:
            return amino_acid
        else:
            # Normalize the scores to sum to 1 and sample from the distribution
            probabilities = [p / total for p in probabilities]
            return random.choices(amino_acids, weights=probabilities, k=1)[0]
    
    def _augment_sequence(self, sequence: str) -> str:
        """
        Augment the sequence by randomly masking amino acids.
        Each position has a probability of being masked based on self.augment_sequence_probability.
        If self.use_residue_masking is True, the masked positions are replaced with a special token. Otherwise, the masked positions are replaced with an amino acid sampled from BLOSUM62.
        Args:
            sequence (str): The amino acid sequence to augment.

        Returns:
            str: The augmented amino acid sequence.
        """
        augmented_sequence = ""
        for amino_acid in sequence:
            if random.random() < self.augment_sequence_probability:
                rand_choice = random.random()
                if rand_choice < 0.10:
                    # Replace with a uniformly sampled random amino acid
                    augmented_sequence += random.choice(self.amino_acid_vocabulary)
                elif rand_choice < 0.20:
                    # Replace with a residue sampled from BLOSUM62
                    augmented_sequence += self._sample_based_on_blosum62(amino_acid)
                elif rand_choice < 0.30:
                    # Leave the amino acid as it is
                    augmented_sequence += amino_acid
                else:
                    # Replace with a special token, if using residue masking
                    if self.use_residue_masking:
                        augmented_sequence += self.masked_msa_token
                    # Otherwise, replace with an amino acid sampled from BLOSUM62
                    else:
                        augmented_sequence += self._sample_based_on_blosum62(amino_acid)
            else:
                # No replacement, keep the original amino acid
                augmented_sequence += amino_acid
        
        return augmented_sequence
    
    def _get_go_descriptions(self) -> list[str]:
        """
        Augment label text by randomly selecting additional GPT summarizations for some labels.
        Each label has a probability of being augmented based on self.augment_label_probability.
        Args:
            labels (list[str]): The labels to augment.

        Returns:
            list[str]: The augmented labels.
        """
        
        label_descriptions = []
        for label in self.label_vocabulary:
            label_descriptions.append(self.label_annotation_map[label][0]) # index 0 is the default description
        
        return label_descriptions

    def process_example(self, sequence: str, labels: list[str], label_idxs:list[int] = None) -> dict:
        sequence_id_alphanumeric, labels = labels[0], labels[1:]
        
        # One-hot encode the labels for use in the loss function (not a model input, so should not be impacted by augmentation)
        labels_ints = torch.tensor(
            [self.label2int[label] for label in labels], dtype=torch.long
        )
        
        # If training, augment the sequence with probability defined in the config
        if self.dataset_type == "train":
            # If self.augment_sequence_probability > 0, augment the sequence
            if self.augment_sequence_probability > 0:
                sequence = self._augment_sequence(sequence)
            
        # Convert the sequence and labels to integers for one-hot encoding (impacted by augmentation)
        amino_acid_ints = torch.tensor(
            [self.aminoacid2int[aa] for aa in sequence], dtype=torch.long
        )

        # Get the length of the sequence
        sequence_length = torch.tensor(len(amino_acid_ints))

        # Get multi-hot encoding of sequence and labels
        sequence_onehots = torch.nn.functional.one_hot(
            amino_acid_ints, num_classes=len(self.amino_acid_vocabulary)
        ).permute(1, 0)
        label_multihots = torch.nn.functional.one_hot(
            labels_ints, num_classes=len(self.label_vocabulary)
        ).sum(dim=0)

        if label_idxs is not None:
            label_idxs = torch.tensor(label_idxs)

        tokenized_labels = None
        label_embeddings = None
        # Set the label embeddings, if provided
        if self.label_embedding_matrix is not None:
            # Augment label descriptions by sampling from synonyms (if available)
            #NOTE: WE AUGMENT LABELS PER SEQUENCE, BUT INSIDE COLLATOR ONLY USE THE LABELS FROM FIRST SEQUENCE IN BATCh
            if self.augment_labels & (self.dataset_type == "train"):
                sampled_description_idxs = []
                for l,description_index_range in self.label_annotation_map_idxs.items():
                    sampled_idx = np.random.randint(description_index_range[0],description_index_range[1]+1) #Add +1 to high b/c randing high range is exclusive
                    sampled_description_idxs.append(sampled_idx)
                
                label_embeddings = self.label_embedding_matrix[sampled_description_idxs]

            else:
                label_embeddings = self.label_embedding_matrix[self.base_label_idxs]
        else:
            # Tokenize the labels if we don't have label embedding
            # extract label descriptions and optionally augment
            label_descriptions = self._get_go_descriptions()

            tokenized_labels = tokenize_labels(label_descriptions, self.label_tokenizer)

        # Get the sequence embedding, if provided
        sequence_embedding = None
        if self.sequence_embedding_df is not None:
            sequence_embedding = torch.tensor(
                self.sequence_embedding_df.loc[sequence_id_alphanumeric].values)

        # Return a dict containing the processed example
        return {
            "sequence_onehots": sequence_onehots,
            "sequence_id": sequence_id_alphanumeric,
            "sequence_embedding": sequence_embedding,
            "sequence_length": sequence_length,
            "label_multihots": label_multihots,
            "tokenized_labels": tokenized_labels,
            "label_embeddings": label_embeddings,
            "label_idxs": label_idxs
        }

    def __getitem__(self, idx) -> tuple:
        
        if self.require_label_idxs:
            sequence_idx,label_idxs = idx[0],idx[1]
            sequence = self.data[sequence_idx][0]
            labels = self.data[sequence_idx][1]
        else:
            label_idxs=None
            sequence, labels = self.data[idx]
        

        return self.process_example(sequence, labels,label_idxs)
    
    def calculate_pos_weight(self):
        self.logger.info("Calculating bce_pos_weight...")
        def count_labels(chunk):
            num_positive_labels_chunk = 0
            num_negative_labels_chunk = 0
            for _, labels in chunk:
                labels = labels[1:]
                num_positive = len(labels)
                num_positive_labels_chunk += num_positive
                num_negative_labels_chunk += len(self.label_vocabulary) - num_positive
            return num_positive_labels_chunk, num_negative_labels_chunk

        chunk_size = len(self.data) // cpu_count()  # Adjust chunk size if necessary.

        results = Parallel(n_jobs=-1)(
            delayed(count_labels)(self.data[i:i+chunk_size]) for i in range(0, len(self.data), chunk_size)
        )

        num_positive_labels = sum(res[0] for res in results)
        num_negative_labels = sum(res[1] for res in results)
        pos_weight = torch.tensor((num_negative_labels / num_positive_labels))
        self.logger.info(f"Calculated bce_pos_weight= {pos_weight.item()}")
        return pos_weight


    def calculate_label_weights(self, inv_freq= True, power=0.3, normalize = True,return_list=False):
        self.logger.info("Calculating label weights...")
        def count_labels(chunk):
            label_freq = Counter()
            for _, labels in chunk:
                labels = labels[1:]
                label_freq.update(labels)
            return label_freq

        # Adjust chunk size if necessary.
        chunk_size = max(len(self.data) // cpu_count(), 1)

        results = Parallel(n_jobs=-1)(
            delayed(count_labels)(self.data[i:i+chunk_size]) for i in range(0, len(self.data), chunk_size)
        )

        #Label frequency
        label_weights = Counter()
        for result in results:
            label_weights.update(result)
        
        if inv_freq:
            # Inverse frequency
            total = sum(label_weights.values())
            num_labels = len(label_weights.keys())
            label_weights = {k: (total/v)**power for k, v in label_weights.items()}
            
        if normalize:
            sum_raw_weights = sum(label_weights.values())
            label_weights = {k:v*num_labels/sum_raw_weights for k,v in label_weights.items()}
        
        #Complete weights with labels not seen in training set but in vocab
        label_weights = {self.label2int[k]:v for k,v in label_weights.items()}
        missing_label_weights = {v:0 for v in self.label2int.values() if v not in label_weights}
        label_weights.update(missing_label_weights)

        self.logger.info(f"# always negative labels: {len(missing_label_weights)}")

        if return_list:
            #Sort weights by vocabulary order
            label_weights = torch.tensor([value for _, value in sorted(label_weights.items())]).float()
        else:
            label_weights = {self.int2label[k]:v for k,v in label_weights.items()}

        return label_weights




def calculate_sequence_weights(data: list, label_inv_freq: dict):
    """
    Calculate the sequence weights for weighted sampling. 
    The sequence weights are the sum of the inverse frequencies of the labels in the sequence.
    """
    # TODO: Convert inverse frequencies into a more normal distribution
    
    def sum_label_inverse_freqs(chunk):
        sequence_weights = []
        for _, labels in chunk:
            labels = labels[1:]
            sequence_weight = sum([label_inv_freq[label] for label in labels])
            sequence_weights.append(sequence_weight)
        return sequence_weights

    # Adjust chunk size if necessary.
    chunk_size = max(len(data) // cpu_count(), 1)

    results = Parallel(n_jobs=-1)(
        delayed(sum_label_inverse_freqs)(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)
    )

    sequence_weights = []
    for result in results:
        sequence_weights.extend(result)

    return sequence_weights


def set_padding_to_sentinel(
    padded_representations: torch.Tensor,
    sequence_lengths: torch.Tensor,
    sentinel: float,
) -> torch.Tensor:
    """
    Set the padding values in the input tensor to the sentinel value.

    Parameters:
        padded_representations (torch.Tensor): The input tensor of shape (batch_size, dim, max_sequence_length)
        sequence_lengths (torch.Tensor): 1D tensor containing original sequence lengths for each sequence in the batch
        sentinel (float): The value to set the padding to

    Returns:
        torch.Tensor: Tensor with padding values set to sentinel
    """

    # Get the shape of the input tensor
    batch_size, dim, max_sequence_length = padded_representations.shape

    # Get the device of the input tensor
    device = padded_representations.device

    # Create a mask that identifies padding, ensuring it's on the same device
    mask = torch.arange(max_sequence_length, device=device).expand(
        batch_size, max_sequence_length
    ) >= sequence_lengths.unsqueeze(1).to(device)

    # Expand the mask to cover the 'dim' dimension
    mask = mask.unsqueeze(1).expand(-1, dim, -1)

    # Use the mask to set the padding values to sentinel
    padded_representations = torch.where(
        mask, sentinel, padded_representations)

    return padded_representations





def create_multiple_loaders(
    datasets: dict,
    params: dict,
    label_sample_sizes: dict = None,
    grid_sampler: bool = False,
    shuffle_labels: bool = False,
    in_batch_sampling: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0,
    sequence_weights: torch.Tensor = None
) -> List[DataLoader]:
    loaders = defaultdict(list)
    for dataset_type, dataset_list in datasets.items():
        batch_size_for_type = params[f"{dataset_type.upper()}_BATCH_SIZE"]

        # Get the number of labels to sample for the current dataset type, if provided
        label_sample_size = None
        if label_sample_sizes:
            label_sample_size = label_sample_sizes.get(dataset_type)

        for dataset in dataset_list:
                            
            batch_sampler = None
            drop_last = True

            if dataset_type == "train":
                sequence_sampler = observation_sampler_factory(
                    distribute_labels = params["DISTRIBUTE_LABELS"],
                    weighted_sampling = params["WEIGHTED_SAMPLING"],
                    dataset = dataset,
                    world_size = world_size,
                    rank = rank,
                    sequence_weights=sequence_weights)
                
                if grid_sampler:
                    assert label_sample_size is not None,"Provide label_sample_size when using grid sampler"
                    batch_sampler=GridBatchSampler(
                        observation_sampler=sequence_sampler,
                        observations_batch_size=batch_size_for_type,
                        drop_last_observation_batch=True,
                        num_labels=len(dataset.label_vocabulary),
                        labels_batch_size=label_sample_size,
                        shuffle_grid=True
                    )
                    
                    # When defining a BatchSampler, these paramters are ignored in the Dataloader. Must be set 
                    # To these values to avoid pytorch error.
                    batch_size_for_type = 1
                    sequence_sampler = None
                    drop_last = False
            else:
                # No sampling in validation and test sets
                sequence_sampler = None

            loader = DataLoader(
                dataset,
                batch_size=batch_size_for_type,
                shuffle=False,
                collate_fn=partial(
                    collate_variable_sequence_length,
                    label_sample_size=label_sample_size,
                    grid_sampler = grid_sampler & (dataset_type == "train"),
                    shuffle_labels=shuffle_labels,
                    in_batch_sampling = in_batch_sampling & (dataset_type == "train"),
                    distribute_labels=params["DISTRIBUTE_LABELS"],
                    world_size=world_size,
                    rank=rank),

                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                sampler=sequence_sampler,
                batch_sampler=batch_sampler
            )
            loaders[dataset_type].append(loader)

    return loaders