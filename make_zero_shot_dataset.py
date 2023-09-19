import argparse
import pandas as pd
from Bio import SwissProt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pickle


def process_data(data_file_path, sequence_embedding_file_path, label_embedding_file_path, output_file_path):
    # Extract data from SwissProt records

    # See https://biopython.org/docs/1.75/api/Bio.SwissProt.html and https://web.expasy.org/docs/userman.html

    with open(data_file_path, 'r') as f:
        data = []
        records = SwissProt.parse(f)

        print("Extracting data from SwissProt records... This may take a while...")
        for record in records:
            # Extract sequence ID
            seq_id = record.accessions[0]

            # Extract sequence
            sequence = record.sequence

            # Extract GO ids
            go_ids = [ref[1] for ref in record.cross_references if ref[0]
                      == "GO" and len(ref) > 0]

            # Extract free-text description
            description = record.description

            # Extract organism and organism classification
            organism = record.organism
            organism_classification = record.organism_classification

            # Extract organelle
            organelle = record.organelle

            # Extract CC line as a dictionary
            cc = {}
            for comment in record.comments:
                key, value = comment.split(": ", 1)
                cc[key] = value

            data.append([seq_id, sequence, go_ids, description,
                        organism, organism_classification, organelle, cc])

    print("Finished extracting data from SwissProt records.")

    # Convert data into a pandas DataFrame and create a new column with the subcellular location
    df_2023 = pd.DataFrame(data, columns=["seq_id", "sequence", "go_ids",
                                          "description", "organism", "organism_classification", "organelle", "cc"])
    df_2023['subcellular_location'] = df_2023.cc.apply(
        lambda x: x.get('SUBCELLULAR LOCATION'))

    # Load the sequence embeddings from the file and make a set of the sequence strings
    # TODO: Pass something other than the sequence embedding file path. Maybe the sequence embedding vocabulary, or even the ProteInfer data file?
    with open(sequence_embedding_file_path, 'rb') as f:
        sequence_embeddings = pickle.load(f)
    sequence_strings_2019 = set(sequence_embeddings.keys())
    df_2023['in_proteinfer_dataset'] = df_2023.seq_id.apply(
        lambda x: x in sequence_strings_2019)

    # Load the label embeddings from the file
    # TODO: Same refactoring as above with sequences, but only need to load full ProteInfer dataset once
    with open(label_embedding_file_path, 'rb') as f:
        label_embeddings_2019 = pickle.load(f)

    # Make a set of the GO labels from the label embeddings
    label_ids_2019 = set(label_embeddings_2019.keys())

    # Make a set from all the GO labels that occur in the data
    label_ids_2023 = set(
        [item for sublist in df_2023.go_ids for item in sublist])

    # Find added labels
    new_go_labels = label_ids_2023 - label_ids_2019

    # Find protein sequences with added labels
    df_2023['new_labels'] = df_2023.go_ids.apply(
        lambda x: set(x) & new_go_labels)

    # Create a new dataframe of only the sequences that are new and have new labels
    filtered_df = df_2023[(df_2023.in_proteinfer_dataset == False) & (
        df_2023.new_labels != set())][['seq_id', 'sequence', 'new_labels']]

    # Set of 20 common amino acids
    common_amino_acids = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                             'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    # Create a column of the non-common amino acids
    filtered_df['non_common_amino_acids'] = filtered_df.sequence.apply(
        lambda x: set(x) - common_amino_acids)

    # Filter to only contain rows that contain common amino acids and rename
    # Filter to only contain rows that contain common amino acids
    SwissProt_2023_unseen_sequences_and_labels = filtered_df[filtered_df.non_common_amino_acids == set(
    )]

    # Rename columns
    SwissProt_2023_unseen_sequences_and_labels = SwissProt_2023_unseen_sequences_and_labels.rename(
        columns={'new_labels': 'go_ids'})

    print("Number of sequences in dataframe: " +
          str(len(SwissProt_2023_unseen_sequences_and_labels)))

    # Convert dataframe to FASTA format and save to a file
    records = [SeqRecord(Seq(row['sequence']), id=row['seq_id'], description=" ".join(
        row['go_ids'])) for _, row in SwissProt_2023_unseen_sequences_and_labels.iterrows()]
    SeqIO.write(records, output_file_path, "fasta")
    print("Saved FASTA file to " + output_file_path)


if __name__ == "__main__":
    """
    Example usage:
    python make_zero_shot_dataset.py data/swissprot/uniprot_sprot.dat data/embeddings/frozen_proteinfer_sequence_embeddings.pkl data/embeddings/frozen_PubMedBERT_label_embeddings.pkl data/zero_shot/SwissProt_2023_unseen_sequences_and_labels.fasta
    """
    parser = argparse.ArgumentParser(
        description="Process SwissProt data and generate a FASTA file.")
    parser.add_argument("data_file_path", type=str,
                        help="Path to the SwissProt data file.")
    parser.add_argument("sequence_embedding_file_path",
                        type=str, help="Path to the sequence embedding file.")
    parser.add_argument("label_embedding_file_path",
                        type=str, help="Path to the label embedding file.")
    parser.add_argument("output_file_path",
                        type=str, help="Path to the output file. Should be a FASTA file.")
    args = parser.parse_args()

    process_data(args.data_file_path, args.sequence_embedding_file_path,
                 args.label_embedding_file_path, args.output_file_path)
