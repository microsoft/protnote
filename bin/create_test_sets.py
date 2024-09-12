import subprocess
from protnote.utils.notebooks import get_data_distributions
from protnote.utils.data import read_fasta, save_to_fasta
import pandas as pd
from protnote.utils.configs import load_config


def create_smaller_test_sets(df,output_dir):
    # Create smaller datasets to measure BLAST runtime
    for n in [1, 10, 100, 1000, 5000, 10_000, 20_000]:
        temp = df.query('split=="test"').sample(n=n, random_state=42)
        temp = [
            (row["sequence"], row["id"], row["labels"].split())
            for _, row in temp.iterrows()
        ]
        save_to_fasta(
            temp,
            output_file=output_dir / f"test_{n}_GO.fasta",
        )
        


def create_top_labels_test_set(df, output_path, go_term_distribution, k=3000):
    # Create top labels df for embeddings analysis
    go_term_distribution_p = go_term_distribution / len(df[df["split"] == "train"])
    top_k_labels = set(go_term_distribution_p.iloc[:k].index)
    test_df = df[df["split"] == "test"]
    test_df_top_k = test_df[
        test_df["labels"].str.split().apply(lambda x: set(x).issubset(top_k_labels))
    ]
    print(
        len(test_df_top_k) / len(test_df),
        go_term_distribution_p.iloc[:k].max(),
        go_term_distribution_p.iloc[:k].min(),
    )
    test_df_top_k = [
        (row["sequence"], row["id"], row["labels"].split())
        for _, row in test_df_top_k.drop_duplicates(subset=["sequence"])
        .sample(frac=0.1, random_state=42)
        .iterrows()
    ]
    save_to_fasta(test_df_top_k,output_path)


if __name__ == "__main__":


    # Load the configuration and project root
    config, project_root = load_config()
    results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]
    
    # Updated Supervised Test Set. pinf test seqs + new labels + new vocab
    subprocess(
        [
            "python",
            "make_dataset_from_swissprot.py",
            "--data-file-path",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DAT_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_2024_DATA_PATH'],
            "--sequence-vocabulary=proteinfer_test",
            "--label-vocabulary=all",
            "--parenthood-file-path",
            config["paths"]['data_paths']['PARENTHOOD_LIB_PATH']
        ],
        check=True,
        shell=True,
    )

    # Updated Supervised Test Set. pinf test seqs + new labels + pinf/old vocab
    subprocess(
        [
            "python",
            "make_dataset_from_swissprot.py",
            "--data-file-path",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DAT_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_2024_PINF_VOCAB_DATA_PATH'],
            "--sequence-vocabulary=proteinfer_test",
            "--label-vocabulary=proteinfer",
            "--parenthood-file-path",
            config["paths"]['data_paths']['PARENTHOOD_LIB_PATH']
        ],
        check=True,
        shell=True,
    )

    # GO Zero Shot 2024 Leaf Nodes. new seqs + new labels only leaf nodes + only added vocab terms
    subprocess(
        [
            "python",
            "make_dataset_from_swissprot.py",
            "--data-file-path",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DAT_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES'],
            "--sequence-vocabulary=new",
            "--only-leaf-nodes",
            "--label-vocabulary=new",
            "--parenthood-file-path",
            config["paths"]['data_paths']['PARENTHOOD_LIB_PATH']
        ],
        check=True,
        shell=True,
    )

    # GO Zero Shot 2024 new seqs + new labels + only added vocab terms
    subprocess(
        [
            "python",
            "make_dataset_from_swissprot.py",
            "--data-file-path",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DAT_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_DATA_PATH_ZERO_SHOT'],
            "--sequence-vocabulary=new",
            "--label-vocabulary=new",
            "--parenthood-file-path",
            config["paths"]['data_paths']['PARENTHOOD_LIB_PATH']
        ],
        check=True,
        shell=True,
    )

    # Updated Supervised Train Set. pinf train seqs + all new & old labels.
    subprocess(
        [
            "python",
            "make_dataset_from_swissprot.py",
            "--data-file-path",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DAT_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TRAIN_2024_DATA_PATH'],
            "--sequence-vocabulary=proteinfer_train",
            "--label-vocabulary=all",
            "--parenthood-file-path",
            config["paths"]['data_paths']['PARENTHOOD_LIB_PATH']
        ],
        check=True,
        shell=True,
    )

    train = read_fasta(config["paths"]['data_paths']['TRAIN_DATA_PATH'])
    test = read_fasta(config["paths"]['data_paths']['TEST_DATA_PATH'])

    for dataset in ["train", "test"]:
        exec(
            f'{dataset} = [(seq,id," ".join(labels),"{dataset}") for seq,id,labels in {dataset}]'
        )

    df = train + test
    df = pd.DataFrame(df, columns=["sequence", "id", "labels", "split"])

    vocab, amino_freq, labels = get_data_distributions(df[df["split"] == "train"])
    go_term_distribution = pd.Series(labels.values(), index=labels.keys()).sort_values(
        ascending=False
    )

    create_smaller_test_sets(df = df,
                             output_dir = project_root / "data" / "swissprot" / "proteinfer_splits" / "random"
                             )
    
    create_top_labels_test_set(df = df,
                               output_path= config["paths"]['data_paths']['TEST_TOP_LABELS_DATA_PATH'],
                               go_term_distribution=go_term_distribution
                               )
