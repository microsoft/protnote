import os
import random
from argparse import ArgumentParser
from tqdm import tqdm
from protnote.utils.data import read_fasta, save_to_fasta, generate_vocabularies
from datetime import datetime
from protnote.utils.configs import load_config


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
    val_labels = label_vocabulary[train_size : train_size + val_size]
    test_labels = label_vocabulary[train_size + val_size :]
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
    filtered = [
        (
            sequence,
            sequence_id,
            [label for label in sequence_labels if label in labels_set],
        )
        for sequence, sequence_id, sequence_labels in tqdm(dataset, desc=desc)
    ]
    return filtered


def main():
    """
    Main function to filter fasta files based on label vocabulary and save the filtered datasets.

    Parameters:
    - train_path: str, path to the training set fasta file.
    - val_path: str, path to the validation set fasta file.
    - test_path: str, path to the test set fasta file.
    - label_vocab_path: str, path to the label vocabulary JSON file.
    - output_path: str, directory where the filtered fasta files will be saved.
    """

    config, project_root = load_config()

    parser = ArgumentParser(
        description="Filter and save datasets for zero-shot learning."
    )
    parser.add_argument(
        "--train-path",
        required=False,
        help="Path to the training set fasta file.",
        default=config["paths"]['data_paths']['TRAIN_DATA_PATH'],
    )
    parser.add_argument(
        "--val-path",
        required=False,
        help="Path to the validation set fasta file.",
        default=config["paths"]['data_paths']['VAL_DATA_PATH'],
    )
    parser.add_argument(
        "--test-path",
        required=False,
        help="Path to the test set fasta file.",
        default=config["paths"]['data_paths']['TEST_DATA_PATH'],
    )
    args = parser.parse_args()

    output_path = project_root / "data" / "swissprot" / "proteinfer_splits" / "random"
    train = read_fasta(args.train_path)
    val = read_fasta(args.val_path)
    test = read_fasta(args.test_path)

    label_vocabulary = generate_vocabularies(file_path=config["paths"]['data_paths']['FULL_DATA_PATH'])["label_vocab"]

    train_labels, val_labels, test_labels = split_labels(label_vocabulary)

    filtered_datasets = {
        "train_GO_zero_shot.fasta": filter_dataset(
            train, train_labels, desc="Filtering training set"
        ),
        "dev_GO_zero_shot.fasta": filter_dataset(
            val, val_labels, desc="Filtering validation set"
        ),
        "test_GO_zero_shot.fasta": filter_dataset(
            test, test_labels, desc="Filtering test set"
        ),
    }

    for filename, dataset in filtered_datasets.items():
        save_to_fasta(dataset, os.path.join(output_path, 'fake_' + filename))


if __name__ == "__main__":
    """
    Example usage:
    python make_zero_shot_datasets_from_proteinfer.py --train_path data/swissprot/proteinfer_splits/random/train_GO.fasta --val_path data/swissprot/proteinfer_splits/random/dev_GO.fasta --test_path data/swissprot/proteinfer_splits/random/test_GO.fasta
    """
    main()
