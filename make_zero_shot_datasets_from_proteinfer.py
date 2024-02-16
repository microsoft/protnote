import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm
from src.utils.data import read_fasta, save_to_fasta
from datetime import datetime

def load_label_vocabulary(label_vocab_path):
    """
    Load the label vocabulary from a JSON file.

    Parameters:
    - label_vocab_path: str, path to the label vocabulary JSON file.

    Returns:
    - dict, label vocabulary loaded from the JSON file.
    """
    with open(label_vocab_path) as f:
        return json.load(f)

def split_labels(label_vocabulary):
    """
    Split the label vocabulary into training, validation, and test sets.

    Parameters:
    - label_vocabulary: list, a list of unique labels.

    Returns:
    - tuple of lists, containing train_labels, val_labels, and test_labels.
    """
    random.shuffle(label_vocabulary)
    train_size = len(label_vocabulary) * 80 // 100
    val_size = len(label_vocabulary) * 10 // 100
    train_labels = label_vocabulary[:train_size]
    val_labels = label_vocabulary[train_size:train_size + val_size]
    test_labels = label_vocabulary[train_size + val_size:]
    return train_labels, val_labels, test_labels


def filter_dataset(dataset, labels, desc="Filtering dataset"):
    """
    Filter the dataset to include only sequences with specified labels.

    Parameters:
    - dataset: list of tuples, where each tuple is (sequence, labels) to be filtered.
    - labels: list, labels to retain in the dataset.

    Returns:
    - list, filtered dataset with only specified labels.
    """
    labels_set = set(labels)  # Convert list to set for O(1) lookup
    filtered = [(sequence, [label for label in sequence_labels if label in labels_set]) for sequence, sequence_labels in tqdm(dataset, desc=desc)]
    return filtered


def main(train_path, val_path, test_path, label_vocab_path, output_path):
    """
    Main function to filter fasta files based on label vocabulary and save the filtered datasets.

    Parameters:
    - train_path: str, path to the training set fasta file.
    - val_path: str, path to the validation set fasta file.
    - test_path: str, path to the test set fasta file.
    - label_vocab_path: str, path to the label vocabulary JSON file.
    - output_path: str, directory where the filtered fasta files will be saved.
    """
    train = read_fasta(train_path)
    val = read_fasta(val_path)
    test = read_fasta(test_path)
    label_vocabulary = load_label_vocabulary(label_vocab_path)
    train_labels, val_labels, test_labels = split_labels(label_vocabulary)

    filtered_datasets = {
        "train_GO_zero_shot.fasta": filter_dataset(train, train_labels, desc="Filtering training set"),
        "dev_GO_zero_shot.fasta": filter_dataset(val, val_labels, desc="Filtering validation set"),
        "test_GO_zero_shot.fasta": filter_dataset(test, test_labels, desc="Filtering test set")
    }

    date_str = datetime.now().strftime("%Y-%m-%d_")
    for filename, dataset in filtered_datasets.items():
        save_to_fasta(dataset, os.path.join(output_path, date_str + filename))

if __name__ == "__main__":
    """
    Example usage:
    python make_zero_shot_datasets_from_proteinfer.py --train_path data/swissprot/proteinfer_splits/random/train_GO.fasta --val_path data/swissprot/proteinfer_splits/random/dev_GO.fasta --test_path data/swissprot/proteinfer_splits/random/test_GO.fasta --label_vocab_path data/vocabularies/proteinfer/GO_label_vocab.json --output_path data/swissprot/proteinfer_splits/random/ 
    """
    parser = ArgumentParser(description="Filter and save datasets for zero-shot learning.")
    parser.add_argument("--train_path", required=True, help="Path to the training set fasta file.")
    parser.add_argument("--val_path", required=True, help="Path to the validation set fasta file.")
    parser.add_argument("--test_path", required=True, help="Path to the test set fasta file.")
    parser.add_argument("--label_vocab_path", required=True, help="Path to the label vocabulary JSON file.")
    parser.add_argument("--output_path", required=True, help="Directory where the filtered fasta files will be saved.")
    
    args = parser.parse_args()
    
    main(args.train_path, args.val_path, args.test_path, args.label_vocab_path, args.output_path)
