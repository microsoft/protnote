import argparse
import pandas as pd
from Bio import SwissProt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
from tqdm import tqdm
from typing import Literal
import collections

from src.utils.data import read_json, read_fasta,generate_vocabularies
def reverse_map(
    applicable_label_dict,
    label_vocab = None):
  """Flip parenthood dict to map parents to children.

  Args:
    applicable_label_dict: e.g. output of get_applicable_label_dict.
    label_vocab: e.g. output of inference_lib.vocab_from_model_base_path

  Returns:
    collections.defaultdict of k, v where:
    k: originally the values in applicable_label_dict
    v: originally the keys in applicable_label_dict.
    The defaultdict returns an empty frozenset for keys that are not found.
    This behavior is desirable for lifted clan label normalizers, where
    keys may not imply themselves.
  """
  # This is technically the entire transitive closure, so it is safe for DAGs
  # (e.g. GO labels).

  children = collections.defaultdict(set)
  for child, parents in applicable_label_dict.items():
    # Avoid adding children which don't appear in the vocab.
    if label_vocab is None or child in label_vocab:
      for parent in parents:
        children[parent].add(child)
  children = {k: frozenset(v) for k, v in children.items()}
  return collections.defaultdict(frozenset, children.items())

def process_data(data_file_path:str, 
                 output_file_path:str,
                 parenthood_file_path:str,
                 label_vocabulary:Literal['proteinfer','new','all'],
                 sequence_vocabulary:Literal['proteinfer_test','proteinfer_test','new','all'],
                 only_leaf_nodes:bool=False,
                 cache=True):
    # Extract data from SwissProt records

    # See https://biopython.org/docs/1.75/api/Bio.SwissProt.html and https://web.expasy.org/docs/userman.html

    if not (os.path.exists("data/swissprot/swissprot_2024_full.pkl") & cache):
        with open(data_file_path, 'r') as f:
            data = []

            records = SwissProt.parse(f)

            print("Extracting data from SwissProt records... This may take a while...")
            for record in tqdm(records,total=571609):
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
        df_2024 = pd.DataFrame(data, columns=["seq_id", "sequence", "go_ids",
                                            "description", "organism", "organism_classification", "organelle", "cc"])
        df_2024['subcellular_location'] = df_2024.cc.apply(
            lambda x: x.get('SUBCELLULAR LOCATION'))

        # Save df_2024 to a file
        df_2024.to_pickle("data/swissprot/swissprot_2024_full.pkl")
    else:
        df_2024 = pd.read_pickle("data/swissprot/swissprot_2024_full.pkl")


    # Make a set of the GO labels from the label embeddings
    label_ids_2019  = set(pd.read_pickle('data/annotations/go_annotations_2019_07_01.pkl').index)
    annotations_2024 = pd.read_pickle('data/annotations/go_annotations_jul_2024.pkl')
    pinf_train = read_fasta('data/swissprot/proteinfer_splits/random/train_GO.fasta')
    pinf_val = read_fasta('data/swissprot/proteinfer_splits/random/dev_GO.fasta')
    pinf_test = read_fasta('data/swissprot/proteinfer_splits/random/test_GO.fasta')

    label_ids_2024 = set(annotations_2024.index)
    
    # Find added labels
    new_go_labels = label_ids_2024 - label_ids_2019

    parenthood = read_json(parenthood_file_path)

    reverse_parenthood = reverse_map(parenthood)
    leaf_nodes=[]         
    for parent,children in reverse_parenthood.items():
        leaf_node=list(children)[0]
        if 'GO' in parent  and len(children)==1 and leaf_node in annotations_2024.index:
            if 'obsolete' not in annotations_2024.loc[leaf_node,'name']:
                leaf_nodes.append(leaf_node)
    leaf_nodes = set(leaf_nodes)

    def add_go_parents(go_terms:list):
        all_terms = set()
        for term in go_terms:
            all_terms.update(parenthood[term]) #Note that parents of term contain term itself
        return list(all_terms)
    
    #Update go terms to include all parents
    df_2024['go_ids'] = df_2024['go_ids'].apply(add_go_parents)

    if sequence_vocabulary == 'new':
        sequence_ids_2019 = set([id for _,id,_ in pinf_train+pinf_val])
        in_proteinfer_train_val= df_2024.seq_id.apply(
            lambda x: x in sequence_ids_2019)
        df_2024 = df_2024[(in_proteinfer_train_val == False)]
    elif sequence_vocabulary == 'proteinfer_test':
        proteinfer_test_set_seqs = set([id for _,id,_ in pinf_test])
        in_proteinfer_test = df_2024.seq_id.apply(
            lambda x: x in proteinfer_test_set_seqs)
        df_2024 = df_2024[(in_proteinfer_test == True)]
    elif sequence_vocabulary == 'proteinfer_train':
        proteinfer_train_set_seqs = set([id for _,id,_ in pinf_train])
        in_proteinfer_train = df_2024.seq_id.apply(
            lambda x: x in proteinfer_train_set_seqs)
        df_2024 = df_2024[(in_proteinfer_train == True)]
    elif sequence_vocabulary=='all':
        pass
    else:
        raise ValueError(f'{sequence_vocabulary} not recognized')
    
    if label_vocabulary =='proteinfer':
        vocab = set(generate_vocabularies('data/swissprot/proteinfer_splits/random/full_GO.fasta')['label_vocab'])
    elif label_vocabulary == 'new':
        vocab = new_go_labels
    elif label_vocabulary == 'all':
        vocab = set([j for i in df_2024.go_ids for j in i])

    if only_leaf_nodes:
        vocab &= leaf_nodes

    print('filtering labels')
    # Find protein sequences with added labels
    df_2024['go_ids'] = df_2024.go_ids.apply(
        lambda x: (set(x) & vocab))
    
    #Remove sequences with no applicable labels
    df_2024 = df_2024[(df_2024.go_ids != set())]

    filtered_df = df_2024[['seq_id', 'sequence', 'go_ids']]

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
    final_labels = set([j for i in SwissProt_2023['go_ids'] for j in i])
    print("Number of sequences in dataframe: " +
          str(len(SwissProt_2023)) +
          f" Number of labels in dataframe: {str(len(final_labels))}"
          )

    print('Writting to FASTA...')
    # Convert dataframe to FASTA format and save to a file
    records = [SeqRecord(Seq(row['sequence']), id=row['seq_id'], description=" ".join(
        row['go_ids'])) for _, row in SwissProt_2023.iterrows()]
    SeqIO.write(records, output_file_path, "fasta")
    print("Saved FASTA file to " + output_file_path)


if __name__ == "__main__":
    """
    Examples usage:
    # python make_dataset_from_swissprot.py --data-file-path data/swissprot/uniprot_sprot.dat --output-file-path data/swissprot/unseen_swissprot_jul_2024.fasta --only-unseen-seqs --label-vocabulary=new --parenthood-file-path data/vocabularies/parenthood_jul_2024.json
    # update test set python make_dataset_from_swissprot.py --data-file-path data/swissprot/uniprot_sprot.dat --output-file-path data/swissprot/test_jul_2024.fasta --only-unseen-seqs --label-vocabulary=new --parenthood-file-path data/vocabularies/parenthood_jul_2024.json


    
        
    """
    parser = argparse.ArgumentParser(
        description="Process SwissProt data and generate a FASTA file.")
    parser.add_argument("--data-file-path", type=str,
                        help="Path to the SwissProt data file.")
    parser.add_argument("--output-file-path",
                        type=str, help="Path to the output file. Should be a FASTA file.")
    parser.add_argument("--parenthood-file-path",type=str,
                        help="path to the parenthood json containing go term children to parent mapping")
    parser.add_argument("--sequence-vocabulary",
                         help="The sequences to use. Can be proteinfer_test, all, or new.",type=str)
    parser.add_argument("--only-leaf-nodes",
                         help="wether to only consider leaf nodes of the hierarchy",action="store_true", default=False)
    parser.add_argument("--label-vocabulary",
                         help="The label vocabulary to use: proteinfer, all, new. all = all observed terms in dataset. New = all observed and new since 2019",required=True)
    parser.add_argument("--no-cache", action="store_true", default=False, help="whether to download data from scratch or read file if exists")
    args = parser.parse_args()

    process_data(data_file_path=args.data_file_path,
                 output_file_path=args.output_file_path,
                 parenthood_file_path=args.parenthood_file_path,
                 label_vocabulary=args.label_vocabulary,
                 sequence_vocabulary=args.sequence_vocabulary,
                 only_leaf_nodes = args.only_leaf_nodes,
                 cache=not args.no_cache)
