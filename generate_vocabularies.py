import argparse
import json
from src.utils.data import read_fasta
import os
import logging

logging.basicConfig(level=logging.INFO)


def generate_vocabularies(data_path, output_types, output_dir, deduplicate):
    # Read the data from the FASTA file
    data = read_fasta(data_path)

    # Extract the required information
    go_labels = set()
    amino_acids = set()  # Renamed from sequences
    sequence_ids = set()

    for sequence, labels in data:
        # Ensure the sequence has not already been added
        if sequence not in amino_acids or not deduplicate:
            sequence_ids.add(labels[0])
            go_labels.update(labels[1:])
            amino_acids.update(list(sequence))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the extracted information to JSON files based on the provided arguments
    if "GO_labels" in output_types:
        with open(os.path.join(output_dir, "GO_label_vocab.json"), "w") as f:
            json.dump(sorted(list(go_labels)), f)
            logging.info(
                f"Saved {len(go_labels)} GO labels to {os.path.join(output_dir, 'GO_label_vocab.json')}")

    if "amino_acids" in output_types:
        with open(os.path.join(output_dir, "amino_acid_vocab.json"), "w") as f:
            json.dump(sorted(list(amino_acids)), f)
            logging.info(
                f"Saved {len(amino_acids)} amino acids to {os.path.join(output_dir, 'amino_acid_vocab.json')}")

    if "sequence_ids" in output_types:
        with open(os.path.join(output_dir, "sequence_id_vocab.json"), "w") as f:
            json.dump(sorted(list(sequence_ids)), f)
            logging.info(
                f"Saved {len(sequence_ids)} sequence IDs to {os.path.join(output_dir, 'sequence_id_vocab.json')}")


if __name__ == "__main__":
    """
    Example usage: python generate_vocabularies.py --input_file data/swissprot/proteinfer_splits/random/full_GO.fasta --output_types GO_labels amino_acids sequence_ids --output_dir data/vocabularies
    """
    parser = argparse.ArgumentParser(
        description="Generate vocabularies from FASTA file.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input FASTA file.")
    parser.add_argument("--output_types", type=str, nargs='+', choices=["GO_labels", "amino_acids", "sequence_ids"], required=True,
                        help="Types of vocabularies to generate: GO_labels, amino_acids, and/or sequence_ids.")
    parser.add_argument("--output_dir", type=str, default="ProteinFunctions/data/vocabularies",
                        help="Directory to save the generated vocabularies.")
    parser.add_argument("--deduplicate", action="store_true", default=False,
                        help="Whether to deduplicate sequences. Default: False")
    args = parser.parse_args()

    generate_vocabularies(args.input_file, args.output_types,
                          args.output_dir, args.deduplicate)
