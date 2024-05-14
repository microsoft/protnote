import argparse
import pandas as pd
from Bio import SwissProt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pickle
import os
from tqdm import tqdm
from src.utils.data import generate_vocabularies


def process_data(data_file_path, output_file_path,cache=True):
    # Extract data from SwissProt records

    # See https://biopython.org/docs/1.75/api/Bio.SwissProt.html and https://web.expasy.org/docs/userman.html

    if not (os.path.exists("/home/samirchar/ProteinFunctions/data/swissprot/swissprot_2023_full.pkl") & cache):
        with open(data_file_path, 'r') as f:
            data = []
            
            # records = SwissProt.parse(f)
            # num_records=0
            # for _ in records:
            #     num_records+=1
            #     print(num_records)

            records = SwissProt.parse(f)

            print("Extracting data from SwissProt records... This may take a while...")
            for record in tqdm(records,total=569793):
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

        # Save df_2023 to a file
        df_2023.to_pickle("/home/samirchar/ProteinFunctions/data/swissprot/swissprot_2023_full.pkl")
    else:
        df_2023 = pd.read_pickle("/home/samirchar/ProteinFunctions/data/swissprot/swissprot_2023_full.pkl")


    # Make a set of the GO labels from the label embeddings
    label_ids_2019  = set(pd.read_pickle('/home/samirchar/ProteinFunctions/data/annotations/go_annotations_2019_07_01.pkl').index)
    label_ids_2023 = set(pd.read_pickle('/home/samirchar/ProteinFunctions/data/annotations/go_annotations_2019_07_01_updated.pkl').index)
 
    # Find added labels
    new_go_labels = label_ids_2023 - label_ids_2019

    # Find protein sequences with added labels
    df_2023['new_labels'] = df_2023.go_ids.apply(
        lambda x: (set(x) & new_go_labels).union(set(['GO:0003674']))) #'GO:0003674' serves as a dummy GO. This is the most frquent term.

    filtered_df = df_2023[['seq_id', 'sequence', 'new_labels']]

    # [(df_2023.in_proteinfer_dataset == False) & (
    #         df_2023.new_labels != set())]
    
    # Set of 20 common amino acids
    common_amino_acids = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                             'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    # Create a column of the non-common amino acids
    filtered_df['non_common_amino_acids'] = filtered_df.sequence.apply(
        lambda x: set(x) - common_amino_acids)

    # Filter to only contain rows that contain common amino acids and rename
    # Filter to only contain rows that contain common amino acids
    SwissProt_2023 = filtered_df[filtered_df.non_common_amino_acids == set(
    )]

    # Rename columns
    SwissProt_2023 = SwissProt_2023.rename(
        columns={'new_labels': 'go_ids'})

    print("Number of sequences in dataframe: " +
          str(len(SwissProt_2023)))

    # Convert dataframe to FASTA format and save to a file
    records = [SeqRecord(Seq(row['sequence']), id=row['seq_id'], description=" ".join(
        row['go_ids'])) for _, row in SwissProt_2023.iterrows()]
    SeqIO.write(records, output_file_path, "fasta")
    print("Saved FASTA file to " + output_file_path)


if __name__ == "__main__":
    """
    Example usage:
    python make_zero_shot_dataset.py data/swissprot/uniprot_sprot.dat data/embeddings/proteinfer/frozen_proteinfer_sequence_embeddings.pkl data/embeddings/proteinfer/frozen_BioGPT_label_embeddings.pkl data/zero_shot/SwissProt_2023.fasta
    """
    # TODO: Refactor this into two scripts, and use vocabularies instead of embeddings
    parser = argparse.ArgumentParser(
        description="Process SwissProt data and generate a FASTA file.")
    parser.add_argument("--data-file-path", type=str,
                        help="Path to the SwissProt data file.")
    parser.add_argument("--output-file-path",
                        type=str, help="Path to the output file. Should be a FASTA file.")
    parser.add_argument("--cache",
                        type=bool, default=True, help="whether to download data from scratch or read file if exists")
    args = parser.parse_args()

    process_data(args.data_file_path, args.output_file_path,cache=args.cache)
