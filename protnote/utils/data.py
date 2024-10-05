from Bio import SeqIO
import json
import pickle
from collections import defaultdict
import gzip
import os
import torch
import yaml
import random
import numpy as np
import wget
import hashlib
import math
import re
from Bio.ExPASy import Enzyme
import blosum as bl
from typing import Union, List, Set, Literal
import transformers
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import contextlib
from joblib import parallel


COMMON_AMINOACIDS = [
                    "A",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "K",
                    "L",
                    "M",
                    "N",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "V",
                    "W",
                    "Y",
                ]


def log_gpu_memory_usage(logger, device_id):
    # Initialize NVML and get handle for the device
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)

    # Get memory information using NVML
    info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total
    used_memory = info.used
    memory_percent = used_memory / total_memory * 100

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device_id)

    # Log memory usage information
    logger.info(
        f"GPU memory occupied: {used_memory // 1024 ** 2} MB ({memory_percent:.2f}% of total memory {total_memory // 1024 ** 2} MB). "
        f"Device {device_id} [Name: {torch.cuda.get_device_name(device_id)}]"
    )


def convert_float16_to_float32(df):
    float16_cols = df.select_dtypes(include="float16").columns
    df[float16_cols] = df[float16_cols].astype("float32")
    return df


def hash_alphanumeric_sequence_id(s: str):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def read_fasta(data_path: str, sep=" "):
    """
    Reads a FASTA file and returns a list of tuples containing sequences, ids, and labels.
    """
    sequences_with_ids_and_labels = []

    for record in SeqIO.parse(data_path, "fasta"):
        sequence = str(record.seq)
        components = record.description.split(sep)
        # labels[0] contains the sequence ID, and the rest of the labels are GO terms.
        sequence_id = components[0]
        labels = components[1:]

        # Return a tuple of sequence, sequence_id, and labels
        sequences_with_ids_and_labels.append((sequence, sequence_id, labels))
    return sequences_with_ids_and_labels


def read_yaml(data_path: str):
    with open(data_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def read_json(data_path: str):
    with open(data_path, "r") as file:
        data = json.load(file)
    return data


def write_json(data, data_path: str):
    with open(data_path, "w") as file:
        json.dump(data, file)


def get_vocab_mappings(vocabulary):
    assert len(vocabulary) == len(set(vocabulary)), "items in vocabulary must be unique"
    term2int = {term: idx for idx, term in enumerate(vocabulary)}
    int2term = {idx: term for term, idx in term2int.items()}
    return term2int, int2term


def generate_vocabularies(file_path: str = None, data: list = None) -> dict:
    """
    Generate vocabularies based on the provided data path.
    path must be .fasta file
    """
    if not ((file_path is None) ^ (data is None)):
        raise ValueError("Only one of file_path OR data must be passed, not both.")
    vocabs = {
        "amino_acid_vocab": set(),
        "label_vocab": set(),
        "sequence_id_vocab": set(),
    }
    if file_path is not None:
        if isinstance(file_path, str):
            data = read_fasta(file_path)
        else:
            raise TypeError(
                "File not supported, vocabularies can only be generated from .fasta files."
            )

    for sequence, sequence_id, labels in data:
        vocabs["sequence_id_vocab"].add(sequence_id)
        vocabs["label_vocab"].update(labels)
        vocabs["amino_acid_vocab"].update(list(sequence))

    for vocab_type in vocabs.keys():
        vocabs[vocab_type] = sorted(list(vocabs[vocab_type]))

    return vocabs


def save_to_pickle(item, file_path: str):
    with open(file_path, "wb") as p:
        pickle.dump(item, p)


def save_to_fasta(sequence_id_labels_tuples, output_file):
    """
    Save a list of tuples in the form (sequence, [labels]) to a FASTA file.

    :param sequence_label_tuples: List of tuples containing sequences and labels
    :param output_file: Path to the output FASTA file
    """
    records = []
    for _, (
        sequence,
        id,
        labels,
    ) in enumerate(sequence_id_labels_tuples):
        # Create a description from labels, joined by space
        description = " ".join(labels)

        record = SeqRecord(Seq(sequence), id=id, description=description)
        records.append(record)

    # Write the SeqRecord objects to a FASTA file
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
        print("Saved FASTA file to " + output_file)


def read_pickle(file_path: str):
    with open(file_path, "rb") as p:
        item = pickle.load(p)
    return item


def download_and_unzip(url, output_file):
    """
    Download a file from a given link and unzip it.

    Args:
        link (str): The URL to download the file from.
        filename (str): The absolute path to save the downloaded file.
    """

    # Download the file from the web
    zip_name = wget.download(url,out=os.path.dirname(output_file))

    # Unzip the downloaded file
    with gzip.open(zip_name, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            f_out.write(f_in.read())

    print(
        f"File {output_file + '.gz'} has been downloaded and unzipped to {output_file}."
    )


def seed_everything(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def load_gz_json(path):
    with open(path, "rb") as f:
        with gzip.GzipFile(fileobj=f, mode="rb") as gzip_file:
            return json.load(gzip_file)


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


def remove_obsolete_from_string(text):
    pattern = r"(?i)\bobsolete\.?\s*"
    return re.sub(pattern, "", text)


class Blossum62Mutations:
    def __init__(self, amino_acid_vocabulary: Union[Set, List] = None):
        if amino_acid_vocabulary is None:
            self.amino_acid_vocabulary = set(COMMON_AMINOACIDS)
        else:
            self.amino_acid_vocabulary = set(amino_acid_vocabulary)

        # Load the BLOSUM62 matrix and convert to defaultdict using dictionary comprehension
        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(
            dict,
            {
                aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()}
                for aa1 in blosum62.keys()
            },
        )

    def get_aa_scores(self, amino_acid: str):
        # Get the substitutions for the amino acid, ensuring only amino acids within the vocabulary are considered
        substitutions = self.blosum62[amino_acid]
        substitutions = {
            aa: score
            for aa, score in substitutions.items()
            if aa in self.amino_acid_vocabulary
        }
        amino_acids, scores = zip(*substitutions.items())
        return amino_acids, scores

    def get_most_extreme_mutation(
        self,
        amino_acid: str,
        mutation_type: Literal["conservative", "non-conservative"],
    ):
        amino_acids, scores = self.get_aa_scores(amino_acid=amino_acid)

        if mutation_type == "conservative":
            fun = max
            # TODO: fix to avoid always returning original aa
        else:
            fun = min

        return amino_acids[scores.index(fun(scores))]

    def corrupt_sequence(
        self,
        sequence: str,
        mutation_type: Literal["conservative", "non-conservative"],
        sample: bool,
    ):
        corrupted = ""
        for aa in sequence:
            corrupted += self.corrupt_amino_acid(
                amino_acid=aa, mutation_type=mutation_type, sample=sample
            )
        return corrupted

    def corrupt_amino_acid(
        self,
        amino_acid: str,
        mutation_type: Literal["conservative", "non-conservative"],
        sample: bool,
    ):
        if sample:
            return self.sample_aa(amino_acid=amino_acid, mutation_type=mutation_type)
        else:
            return self.get_most_extreme_mutation(
                amino_acid=amino_acid, mutation_type=mutation_type
            )

    def corrupt_sequence_at_locations(
        self,
        sequence: str,
        locations: set,
        mutation_type: Literal["conservative", "non-conservative"],
        sample: bool,
    ):
        corrupted = ""
        for loc, aa in enumerate(sequence):
            if loc in locations:
                corrupted += self.corrupt_amino_acid(
                    amino_acid=aa, mutation_type=mutation_type, sample=sample
                )
            else:
                corrupted += aa
        return corrupted

    def sample_aa(
        self,
        amino_acid: str,
        mutation_type: Literal["conservative", "non-conservative"],
    ) -> str:
        """
        Sample an amino acid based on the BLOSUM62 substitution matrix, favoring mutations based on mutation_type selected.
        Args:
            amino_acid (str): The amino acid to find a substitution for.
        Returns:
            str: The substituted amino acid.
        """

        amino_acids, scores = self.get_aa_scores(amino_acid=amino_acid)
        multiplier = -1 if mutation_type == "non-conservative" else 1
        # Use only non-negative scores
        probabilities = [max(0, score * multiplier) for score in scores]
        total = sum(probabilities)

        # If all scores are negative, do not change the amino acid
        if total == 0:
            return amino_acid
        else:
            # Normalize the scores to sum to 1 and sample from the distribution
            probabilities = [p / total for p in probabilities]
            return random.choices(amino_acids, weights=probabilities, k=1)[0]


def ec_number_to_code(ec_number: str, depth: int = 3) -> tuple:
    ec_code = [int(i) for i in re.findall("\d+", ec_number.strip())[:depth]]
    return tuple(ec_code + [0] * (depth - len(ec_code)))


def get_ec_class_descriptions(enzclass_path: str) -> dict:
    with open(enzclass_path) as handle:
        ec_classes = handle.readlines()[11:-5]

    # Dictionary to store the results
    ec_classes_dict = {}

    # Compile the regex pattern to identify the ID
    pattern = re.compile(r"^(\d+\.\s*(\d+|-)\.\s*(\d+|-)\.-)")

    # Contructs, description based on parents.. not the most efficient but doesn't matter for this case
    def get_deep_label(code):
        level_code = [0, 0, 0]
        label = ""
        for level in range(3):
            if code[level] > 0:
                level_code[level] = code[level]
                raw_label = ec_classes_dict[tuple(level_code)]["raw_label"].rstrip(".")
                if level > 0:
                    raw_label = raw_label[0].lower() + raw_label[1:]
                    prefix = ", "
                else:
                    prefix = ""
                label += prefix + raw_label
        return label

    # Process each line
    for line in ec_classes:
        # Find the ID using the regex
        match = pattern.search(line)
        if match:
            # Extract the ID
            ec_number = match.group(1).strip()
            # Everything after the ID is considered the description
            description = line[match.end() :].strip()
            code = ec_number_to_code(ec_number)
            # Add to the dictionary, removing excess spaces and newlines

            ec_classes_dict[code] = {
                "raw_label": description,
                "ec_number": ec_number.replace(" ", ""),
            }

    # Output the result
    for code in ec_classes_dict.keys():
        ec_classes_dict[code]["label"] = get_deep_label(code)

    return ec_classes_dict


def get_ec_number_description(enzyme_dat_path: str, ec_classes: dict) -> list:
    with open(enzyme_dat_path) as handle:
        ec_leaf_nodes = Enzyme.parse(handle)
        ec_leaf_nodes = [
            {
                "ec_number": record["ID"],
                "label": record["CA"],
                "parent_code": ec_number_to_code(record["ID"]),
            }
            for record in ec_leaf_nodes
        ]

    for leaf_node in ec_leaf_nodes:
        if leaf_node["label"] == "":
            leaf_node["label"] = ec_classes[leaf_node["parent_code"]]["label"]
    return ec_leaf_nodes


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


