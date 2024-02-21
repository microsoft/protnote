import torch
import logging
import random
import blosum as bl
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
from functools import partial
from collections import Counter
from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.data.collators import collate_variable_sequence_length
from src.utils.data import read_fasta, get_vocab_mappings
from src.data.samplers import GridBatchSampler,observation_sampler_factory
from src.utils.data import generate_vocabularies

class ProteinDataset(Dataset):
    """
    Dataset class for protein sequences with GO annotations.
    """
    def __init__(
        self,
        data_paths: dict,
        config: dict,
        logger=None,
        require_label_idxs=False,
        label_tokenizer=None
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
            "test"
        ], "dataset_type must be one of 'train', 'val', or 'test'"
        
        
        # Tokenizer
        self.label_tokenizer = label_tokenizer

        # Set the dataset type and data path
        self.dataset_type = data_paths["dataset_type"]
        self.data_path = data_paths["data_path"]
        
        # Initialize data augmentation parameters
        self.augment_residue_probability = config["params"]["AUGMENT_RESIDUE_PROBABILITY"]
        self.label_augmentation_descriptions = config["params"]["LABEL_AUGMENTATION_DESCRIPTIONS"].split('+')
        
        # Load the BLOSUM62 matrix and convert to defaultdict using dictionary comprehension
        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(dict, {aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()} for aa1 in blosum62.keys()})
        
        # Initialize class variables and pre-computed embedding matrices
        self.data = read_fasta(data_paths["data_path"])
        
        # Parameter for noising the label embeddings
        self.label_embedding_noising_alpha = config['params']['LABEL_EMBEDDING_NOISING_ALPHA']
        
        # Flag to know how Dataset indexing will be handled
        self.require_label_idxs = require_label_idxs

        # Subset the data if subset_fraction is provided
        subset_fraction=config['params'][f"{self.dataset_type.upper()}_SUBSET_FRACTION"]
        if subset_fraction < 1.0:
            logging.info(
                f"Subsetting {subset_fraction*100}% of the {self.dataset_type} set..."
            )
            self.data = self.data[:int(subset_fraction * len(self.data))]
        
        # Define description types used for inference. Can be 1 or more. If more than 1, then predictions 
        # will be ensembled per go term
        self.inference_go_descriptions = config['params']['INFERENCE_GO_DESCRIPTIONS'].split('+') 

        # Set the vocabularies.
        # If extract_vocabularies is null, generate vocab from self.data
        self.extract_vocabularies_from = config["params"]["EXTRACT_VOCABULARIES_FROM"]
        vocabulary_path = config['paths'][self.extract_vocabularies_from] if self.extract_vocabularies_from is not None else self.data_path
        logging.info(f"Extracting vocabularies for {self.dataset_type} from {vocabulary_path}")
        vocabularies = generate_vocabularies(file_path = vocabulary_path)
        
        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["GO_label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]

        # Preprocess dataset
        self.label_frequency = None
        self._preprocess_data(
            deduplicate=config['params']["DEDUPLICATE"],
            max_sequence_length=config["params"]["MAX_SEQUENCE_LENGTH"],
            remove_unrepresented_labels=config["params"]["REMOVE_UNREPRESENTED_LABELS"]
            )

        # TODO: This path could be constructed in get_setup
        INDEX_OUTPUT_PATH = config['LABEL_EMBEDDING_PATH'].split('.')
        INDEX_OUTPUT_PATH = '_'.join([INDEX_OUTPUT_PATH[0] ,'index']) + '.'+ INDEX_OUTPUT_PATH[1]
        index_mapping = torch.load(INDEX_OUTPUT_PATH)
        self.label_embeddings_index, self.label_embeddings, self.label_token_counts, self.label_descriptions = self._process_label_embedding_mapping(mapping = index_mapping,
                                                                                                                                                     embeddings = torch.load(config['LABEL_EMBEDDING_PATH']))
        logging.info('Number of unique labels in the label embeddings index: %s', len(self.label_embeddings_index))
        logging.info('Total number of label embeddings: %s', len(self.label_embeddings))
        logging.info('Total number of label token counts: %s', len(self.label_token_counts))
        
    def _preprocess_data(self,deduplicate,max_sequence_length,remove_unrepresented_labels):
        """
        Remove duplicate sequences from self.data, keeping only the first instance of each sequence
        Use pandas to improve performance
        """
        self.logger.info("Cleaning data...")
        # Convert self.data to a DataFrame
        df = pd.DataFrame(self.data, columns=["sequence", "sequence_id", "labels"])

        if deduplicate:
            # Drop duplicate rows based on the 'sequence' column, keeping the first instance
            df = df.drop_duplicates(subset="sequence", keep="first")

            # Log the number of duplicate sequences removed
            num_duplicates = len(self.data) - len(df)
            logging.info(
                f"Removing {num_duplicates} duplicate sequences from {self.data_path}...")
        
        # In train, remove sequences longer than max_sequence_length
        if (max_sequence_length is not None) & (self.dataset_type == "train"):
            seq_length_mask = df["sequence"].apply(len)<=max_sequence_length
            num_long_sequences = (~seq_length_mask).sum()
            df = df[seq_length_mask]
            logging.info(
                f"Removing {num_long_sequences} sequences longer than {max_sequence_length} from {self.data_path}...")

        # Convert the DataFrame back to the list of tuples format
        self.data = list(df.itertuples(index=False, name=None))

        # Calculate label frequency
        self.calculate_label_frequency()

        # Process vocabulary
        # TODO: remove unrepresented labels is depracted in favor of using extract_vocabularies_from = null
        if (remove_unrepresented_labels) and (self.dataset_type == "train"):
            self.logger.info("Removing unrepresented labels from the training set vocabulary")

            self.label_vocabulary = [label for label in self.label_vocabulary if label in self.label_frequency]
        
        self._process_vocab()

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
        Each position has a probability of being augmented based on self.augment_residue_probability.
        Args:
            sequence (str): The amino acid sequence to augment.

        Returns:
            str: The augmented amino acid sequence.
        """
        augmented_sequence = ""
        for amino_acid in sequence:
            if random.random() < self.augment_residue_probability:
                # Replace with a residue sampled from BLOSUM62
                # Most likely, this will be the same residue, so the actual probability of change is lower than augment_residue_probability)
                augmented_sequence += self._sample_based_on_blosum62(amino_acid)
            else:
                # No replacement, keep the original amino acid
                augmented_sequence += amino_acid
        
        return augmented_sequence

    def _process_label_embedding_mapping(self,
                                         mapping:pd.DataFrame,
                                         embeddings:torch.Tensor):

        assert set(self.inference_go_descriptions).issubset(set(['name','label'])), """only supporting name, label or name+label"""
        
        self.logger.info("Processing label embeddings...")
        descriptions_considered = self.label_augmentation_descriptions if self.dataset_type=='train' else self.inference_go_descriptions

        # Select only desired description types and ids from label vocabulary
        mask = (mapping['description_type'].isin(descriptions_considered))\
               &(mapping['id'].isin(self.label_vocabulary))\
                .values
        mapping = mapping[mask]
        embeddings = embeddings[mask]

        mapping = mapping\
            .reset_index(drop=True)\
            .reset_index()

        #For safety 
        assert len(embeddings) == len(mapping)
        
        # Extract number of tokens for each label
        token_counts = mapping['token_count'].values
        
        # Descriptions
        descriptions = mapping['description'].values
        
        # And filtering the tensor as well.
        mapping = mapping.groupby('id').agg(min_idx=('index','min'),max_idx=('index','max')).to_dict(orient='index')

        self.logger.info("Done")
        return mapping, embeddings, token_counts, descriptions
        
    def _sample_label_embeddings(self):
        label_embedding_idxs_list = []
        
        for go_term in self.label_vocabulary:    

            idx = np.random.randint(low=self.label_embeddings_index[go_term]['min_idx'],
                                    high=self.label_embeddings_index[go_term]['max_idx']+1
                                    )
            label_embedding_idxs_list.append(idx)

        return self.label_embeddings[label_embedding_idxs_list], self.label_token_counts[label_embedding_idxs_list]

    def process_example(self, sequence: str, sequence_id_alphanumeric: str, labels: list[str], label_idxs:list[int] = None) -> dict:

        # One-hot encode the labels for use in the loss function (not a model input, so should not be impacted by augmentation)
        labels_ints = torch.tensor(
            [self.label2int[label] for label in labels], dtype=torch.long
        )
        
        # If training, augment the sequence with probability defined in the config
        if self.dataset_type == "train":
            # If self.augment_residue_probability > 0, augment the sequence
            if self.augment_residue_probability > 0:
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

        # Augment label descriptions by sampling from synonyms (if available)
        # NOTE: WE AUGMENT LABELS PER SEQUENCE, BUT INSIDE COLLATOR ONLY USE THE LABELS FROM FIRST SEQUENCE IN BATCh
        if self.dataset_type == "train" and len(self.label_augmentation_descriptions) > 1:
            # Use the augmentation pipeline, which is O(n) where n is the number of labels
            label_embeddings, label_token_counts = self._sample_label_embeddings()
        else:
            # Use the original label embeddings and token counts, which is O(1)
            label_embeddings = self.label_embeddings
            label_token_counts = self.label_token_counts
            
        # Noise the label embedding during training
        if self.dataset_type == "train" and self.label_embedding_noising_alpha > 0:
            # scaling the entire noise vector by a factor of α/√(Ld)
            # L is the sequence length, d is the embedding dimension, and α is a tunable parameter
            denominator = torch.sqrt(torch.tensor(label_token_counts, dtype=torch.float32) * label_embeddings.shape[1])
            scalars = self.label_embedding_noising_alpha / denominator

            # Generate random noise of the same shape as label_embeddings
            # Adjust values to be in the range [-1, 1)
            noise = 2 * torch.rand_like(label_embeddings) - 1

            # Reshape for broadcasting and scale the noise
            scaled_noise = noise * scalars.view(-1, 1)

            # Add the scaled noise to the original label embeddings
            label_embeddings = label_embeddings + scaled_noise
            
        # Return a dict containing the processed example
        return {
            "sequence_onehots": sequence_onehots,
            "sequence_id": sequence_id_alphanumeric,
            "sequence_length": sequence_length,
            "label_multihots": label_multihots,
            "label_embeddings": label_embeddings,
            "label_idxs": label_idxs
        }

    def __getitem__(self, idx) -> tuple:
        if self.require_label_idxs:
            # For Grid sampler, idx is a tuple of (sequence_idx, label_idxs)
            sequence_idx, label_idxs = idx[0], idx[1]
            sequence, sequence_id, labels = self.data[sequence_idx] # We throw away sequence_id
        else:
            # Otherwise, idx is just the sequence index
            label_idxs = None
            sequence, sequence_id, labels = self.data[idx] # We throw away sequence_id
        
        return self.process_example(sequence, sequence_id, labels, label_idxs)
    
    def calculate_pos_weight(self):
        #TODO: UPDATE THIS CODE TO LEVERAGE LABEL FREQUENCY ATTRIBUTE INSTEAD
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

    # def calculate_label_frequency(self):
    #     if self.label_frequency is None:
    #         self.logger.info("Calculating label frequency...")
    #         def count_labels(chunk):
    #             label_freq = Counter()
    #             for _, _, labels in chunk:
    #                 label_freq.update(labels)
    #             return label_freq

    #         # Adjust chunk size if necessary.
    #         chunk_size = max(len(self.data) // cpu_count(), 1)

    #         # Count labels in parallel
    #         results = Parallel(n_jobs=-1)(
    #             delayed(count_labels)(self.data[i:i+chunk_size]) for i in range(0, len(self.data), chunk_size)
    #         )

    #         #Label frequency
    #         self.label_frequency = Counter()
    #         for result in results:
    #             self.label_frequency.update(result)
    def calculate_label_frequency(self):
        if self.label_frequency is None:
            self.logger.info("Calculating label frequency...")

            # Initialize a Counter object for counting label frequencies
            label_freq = Counter()

            # Directly iterate over the data to count labels
            for _, _, labels in self.data:
                label_freq.update(labels)

            # Update the instance's label frequency with the calculated frequencies
            self.label_frequency = label_freq
        

    def calculate_label_weights(self, inv_freq= True, power=0.3, normalize = True,return_list=False):
        self.logger.info("Calculating label weights...")
        
        assert self.label_frequency is not None, "Must call calculate_label_frequency first"

        label_weights = self.label_frequency.copy()
        
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




# def calculate_sequence_weights(data: list, label_inv_freq: dict):
#     """
#     Calculate the sequence weights for weighted sampling. 
#     The sequence weights are the sum of the inverse frequencies of the labels in the sequence.
#     """
#     def sum_label_inverse_freqs(chunk):
#         sequence_weights = []
#         for _, _, labels in chunk:
#             labels = labels[1:]
#             sequence_weight = sum([label_inv_freq[label] for label in labels])
#             sequence_weights.append(sequence_weight)
#         return sequence_weights

#     # Adjust chunk size if necessary.
#     chunk_size = max(len(data) // cpu_count(), 1)

#     results = Parallel(n_jobs=-1)(
#         delayed(sum_label_inverse_freqs)(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)
#     )

#     sequence_weights = []
#     for result in results:
#         sequence_weights.extend(result)

#     return sequence_weights

def calculate_sequence_weights(data: list, label_inv_freq: dict):
    """
    Calculate the sequence weights for weighted sampling.
    The sequence weights are the sum of the inverse frequencies of the labels in the sequence.
    """
    sequence_weights = []
    for _, _, labels in data:
        labels = labels[1:]  # Assuming the first element is not a label
        sequence_weight = sum([label_inv_freq.get(label, 0) for label in labels])
        sequence_weights.append(sequence_weight)
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