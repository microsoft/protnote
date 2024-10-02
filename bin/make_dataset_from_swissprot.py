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
from protnote.utils.configs import get_project_root,construct_absolute_paths, load_config, get_logger
from protnote.utils.data import read_json, read_fasta, generate_vocabularies, COMMON_AMINOACIDS

logger = get_logger()

def reverse_map(applicable_label_dict, label_vocab=None):
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

      DIRECTLY TAKEN FROM PROTEINFER SORUCE CODE
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


def main(
    latest_swissprot_file: str,
    output_file_path: str,
    parenthood_file: str,
    label_vocabulary: Literal["proteinfer", "new", "all"],
    sequence_vocabulary: Literal["proteinfer_test", "proteinfer_test", "new", "all"],
    only_leaf_nodes: bool = False,
    cache=True,
):
    # Load the configuration and project root
    
    # Load the configuration and project root
    config, project_root = load_config()
    results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]
    swissprot_dir = project_root / 'data' / 'swissprot'
    annotations_dir = project_root / 'data' / 'annotations'
    latest_swissprot_file = swissprot_dir / latest_swissprot_file
    
    #Create the output directory from output_file_path if it does not exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Extract data from SwissProt records

    # See https://biopython.org/docs/1.75/api/Bio.SwissProt.html and https://web.expasy.org/docs/userman.html

    if not (os.path.exists(swissprot_dir / args.parsed_latest_swissprot_file) & cache):
        with open(latest_swissprot_file, "r") as f:
            data = []

            records = SwissProt.parse(f)

            logger.info("Extracting data from SwissProt records... This may take a while...")
            for record in tqdm(records, total=571609):
                # Extract sequence ID
                seq_id = record.accessions[0]

                # Extract sequence
                sequence = record.sequence

                # Extract GO ids
                go_ids = [
                    ref[1]
                    for ref in record.cross_references
                    if ref[0] == "GO" and len(ref) > 0
                ]

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

                data.append(
                    [
                        seq_id,
                        sequence,
                        go_ids,
                        description,
                        organism,
                        organism_classification,
                        organelle,
                        cc,
                    ]
                )

        logger.info("Finished extracting data from SwissProt records.")

        # Convert data into a pandas DataFrame and create a new column with the subcellular location
        df_latest = pd.DataFrame(
            data,
            columns=[
                "seq_id",
                "sequence",
                "go_ids",
                "description",
                "organism",
                "organism_classification",
                "organelle",
                "cc",
            ],
        )
        df_latest["subcellular_location"] = df_latest.cc.apply(
            lambda x: x.get("SUBCELLULAR LOCATION")
        )

        # Save df_latest to a file
        df_latest.to_pickle(swissprot_dir / args.parsed_latest_swissprot_file)
    else:
        df_latest = pd.read_pickle(swissprot_dir / args.parsed_latest_swissprot_file)

    # Make a set of the GO labels from the label embeddings
    label_ids_2019 = set(pd.read_pickle(annotations_dir / "go_annotations_2019_07_01.pkl").index)
    annotations_latest = pd.read_pickle(annotations_dir / args.latest_go_annotations_file)
    pinf_train = read_fasta(config['paths']['data_paths']['TRAIN_DATA_PATH'])
    pinf_val = read_fasta(config['paths']['data_paths']['VAL_DATA_PATH'])
    pinf_test = read_fasta(config['paths']['data_paths']['TEST_DATA_PATH'])

    label_ids_2024 = set(annotations_latest.index)

    # Find added labels
    new_go_labels = label_ids_2024 - label_ids_2019

    parenthood = read_json(project_root / "data" / "vocabularies" / parenthood_file)

    reverse_parenthood = reverse_map(parenthood)
    leaf_nodes = []
    for parent, children in reverse_parenthood.items():
        leaf_node = list(children)[0]
        if (
            "GO" in parent
            and len(children) == 1
            and leaf_node in annotations_latest.index
        ):
            if "obsolete" not in annotations_latest.loc[leaf_node, "name"]:
                leaf_nodes.append(leaf_node)
    leaf_nodes = set(leaf_nodes)

    def add_go_parents(go_terms: list):
        all_terms = set()
        for term in go_terms:
            all_terms.update(
                parenthood[term]
            )  # Note that parents of term contain term itself
        return list(all_terms)

    # Update go terms to include all parents
    df_latest["go_ids"] = df_latest["go_ids"].apply(add_go_parents)

    if sequence_vocabulary == "new":
        sequence_ids_2019 = set([id for _, id, _ in pinf_train + pinf_val])
        in_proteinfer_train_val = df_latest.seq_id.apply(lambda x: x in sequence_ids_2019)
        df_latest = df_latest[(in_proteinfer_train_val == False)]
    elif sequence_vocabulary == "proteinfer_test":
        proteinfer_test_set_seqs = set([id for _, id, _ in pinf_test])
        in_proteinfer_test = df_latest.seq_id.apply(
            lambda x: x in proteinfer_test_set_seqs
        )
        df_latest = df_latest[(in_proteinfer_test == True)]
    elif sequence_vocabulary == "proteinfer_train":
        proteinfer_train_set_seqs = set([id for _, id, _ in pinf_train])
        in_proteinfer_train = df_latest.seq_id.apply(
            lambda x: x in proteinfer_train_set_seqs
        )
        df_latest = df_latest[(in_proteinfer_train == True)]
    elif sequence_vocabulary == "all":
        pass
    else:
        raise ValueError(f"{sequence_vocabulary} not recognized")

    if label_vocabulary == "proteinfer":
        vocab = set(
            generate_vocabularies(
                str(config['paths']['data_paths']['FULL_DATA_PATH'])
            )["label_vocab"]
        )
    elif label_vocabulary == "new":
        vocab = new_go_labels
    elif label_vocabulary == "all":
        vocab = set([j for i in df_latest.go_ids for j in i])

    if only_leaf_nodes:
        vocab &= leaf_nodes

    logger.info("filtering labels")
    # Find protein sequences with added labels
    df_latest["go_ids"] = df_latest.go_ids.apply(lambda x: (set(x) & vocab))

    # Remove sequences with no applicable labels
    df_latest = df_latest[(df_latest.go_ids != set())]

    filtered_df = df_latest[["seq_id", "sequence", "go_ids"]]

    # Set of 20 common amino acids
    common_amino_acids = set(COMMON_AMINOACIDS)

    # Create a column of the non-common amino acids
    filtered_df["non_common_amino_acids"] = filtered_df.sequence.apply(
        lambda x: set(x) - common_amino_acids
    )

    # Filter to only contain rows that contain common amino acids and rename
    SwissProt_latest = filtered_df[filtered_df.non_common_amino_acids == set()]

    # Rename columns
    final_labels = set([j for i in SwissProt_latest["go_ids"] for j in i])
    logger.info(
        "Number of sequences in dataframe: "
        + str(len(SwissProt_latest))
        + f" Number of labels in dataframe: {str(len(final_labels))}"
    )

    logger.info("Writting to FASTA...")
    # Convert dataframe to FASTA format and save to a file
    records = [
        SeqRecord(
            Seq(row["sequence"]), id=row["seq_id"], description=" ".join(row["go_ids"])
        )
        for _, row in SwissProt_latest.iterrows()
    ]
    SeqIO.write(records, output_file_path, "fasta")
    logger.info("Saved FASTA file to " + output_file_path)


if __name__ == "__main__":
    """
    Examples usage:
    # python make_dataset_from_swissprot.py --latest-swissprot-file uniprot_sprot.dat --output-file-path unseen_swissprot_jul_2024.fasta --only-unseen-seqs --label-vocabulary=new --parenthood-file parenthood_jul_2024.json
    # updated test set: python make_dataset_from_swissprot.py --latest-swissprot-file uniprot_sprot.dat --output-file-path test_jul_2024.fasta --only-unseen-seqs --label-vocabulary=new --parenthood-file parenthood_jul_2024.json
    """
    parser = argparse.ArgumentParser(
        description="Process SwissProt data and generate a FASTA file."
    )
    parser.add_argument(
        "--latest-swissprot-file",
        type=str,
        help="Path to the SwissProt data file."
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        help="Path to the output file. Should be a FASTA file. ",
    )

    parser.add_argument(
        "--parsed-latest-swissprot-file",
        type=str,
        default='swissprot_2024_full.pkl',
        help="Path to the latest parsed SwissProt file if exists. Otherwise will be created for caching purposes. Date should match latest-go-annotations-file",
    )

    parser.add_argument(
        "--latest-go-annotations-file",
        type=str,
        default="go_annotations_jul_2024.pkl",
        help="Path to the latest go annotations file available. Date should match parsed-latest-swissprot-file"
    )

    parser.add_argument(
        "--parenthood-file",
        type=str,
        help="path to the parenthood json containing go term children to parent mapping",
    )
    parser.add_argument(
        "--sequence-vocabulary",
        help="The sequences to use. Can be proteinfer_test, all, or new.",
        type=str,
    )
    parser.add_argument(
        "--only-leaf-nodes",
        help="wether to only consider leaf nodes of the hierarchy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--label-vocabulary",
        help="The label vocabulary to use: proteinfer, all, new. all = all observed terms in dataset. New = all observed and new since 2019",
        required=True,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="whether to download data from scratch or read file if exists",
    )
    args = parser.parse_args()

    main(
        latest_swissprot_file=args.latest_swissprot_file,
        output_file_path=args.output_file_path,
        parenthood_file=args.parenthood_file,
        label_vocabulary=args.label_vocabulary,
        sequence_vocabulary=args.sequence_vocabulary,
        only_leaf_nodes=args.only_leaf_nodes,
        cache=not args.no_cache,
    )
