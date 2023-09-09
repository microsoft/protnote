import argparse
import json
from src.utils.data import read_fasta
import os


def generate_vocabularies(data_path, output_type, output_dir):
    # Read the data from the FASTA file
    data = read_fasta(data_path)

    # Extract the required information
    go_labels = set()
    sequences = set()
    sequence_ids = set()

    for sequence, labels in data:
        sequence_ids.add(labels[0])
        go_labels.update(labels[1:])
        sequences.add(sequence)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the extracted information to JSON files based on the provided argument
    if output_type == "go_labels":
        with open(os.path.join(output_dir, "go_labels_vocab.json"), "w") as f:
            json.dump(sorted(list(go_labels)), f)
    elif output_type == "sequences":
        with open(os.path.join(output_dir, "sequences_vocab.json"), "w") as f:
            json.dump(sorted(list(sequences)), f)
    elif output_type == "sequence_ids":
        with open(os.path.join(output_dir, "sequence_ids_vocab.json"), "w") as f:
            json.dump(sorted(list(sequence_ids)), f)


if __name__ == "__main__":
    """
    Example usage: python generate_vocabularies.py --input_file data/swissprot/proteinfer_splits/random/full_GO.fasta --output_type go_labels --output_dir data/vocabularies
    """
    parser = argparse.ArgumentParser(
        description="Generate vocabularies from FASTA file.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input FASTA file.")
    parser.add_argument("--output_type", type=str, choices=["GO_labels", "sequences", "sequence_ids"],
                        help="Type of vocabulary to generate: GO_labels, sequences, or sequence_ids.")
    parser.add_argument("--output_dir", type=str, default="ProteinFunctions/data/vocabularies",
                        help="Directory to save the generated vocabularies.")
    args = parser.parse_args()

    generate_vocabularies(args.input_file, args.output_type, args.output_dir)
