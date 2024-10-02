import subprocess
from protnote.utils.notebooks import get_data_distributions
from protnote.utils.data import read_fasta, save_to_fasta
import pandas as pd
from protnote.utils.configs import load_config, get_logger

logger = get_logger()



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
            output_file=str(output_dir / f"test_{n}_GO.fasta"),
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
    save_to_fasta(test_df_top_k,str(output_path))


if __name__ == "__main__":


    # Load the configuration and project root
    config, project_root = load_config()
    results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]
    
    #Create fake train,val,test sets for hparam tuning
    logger.info("Creating fake train, val, test sets for hyperparameter tuning...")
    subprocess.run(["python", "bin/make_zero_shot_datasets_from_proteinfer.py"],check=True,shell=False)
    logger.info("Done.")

    # Updated Supervised Test Set. pinf test seqs + new labels + new vocab
    logger.info("Updated Supervised Test Set. ProteInfer test seqs with new labels & new vocab...")
    subprocess.run(
        [
            "python",
            "bin/make_dataset_from_swissprot.py",
            "--latest-swissprot-file",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DATA_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_2024_DATA_PATH'],
            "--sequence-vocabulary",
            "proteinfer_test",
            "--label-vocabulary",
            "all",
            "--parenthood-file",
            "parenthood_jul_2024.json"
        ],
        check=True,
        shell=False,
    )
    logger.info("Done.")

    # Updated Supervised Test Set. pinf test seqs + new labels + pinf/old vocab
    logger.info("Updated Supervised Test Set. ProteInfer test seqs with new labels & ProteInfer/old vocab...")
    subprocess.run(
        [
            "python",
            "bin/make_dataset_from_swissprot.py",
            "--latest-swissprot-file",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DATA_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_2024_PINF_VOCAB_DATA_PATH'],
            "--sequence-vocabulary",
            "proteinfer_test",
            "--label-vocabulary",
            "proteinfer",
            "--parenthood-file",
            "parenthood_jul_2024.json"
        ],
        check=True,
        shell=False,
    )
    logger.info("Done.")

    # GO Zero Shot 2024 Leaf Nodes. new seqs + new labels only leaf nodes + only added vocab terms
    logger.info("GO Zero Shot 2024 Leaf Nodes. New seqs with new labels only leaf nodes & only added vocab terms")
    subprocess.run(
        [
            "python",
            "bin/make_dataset_from_swissprot.py",
            "--latest-swissprot-file",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DATA_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES'],
            "--sequence-vocabulary",
            "new",
            "--only-leaf-nodes",
            "--label-vocabulary",
            "new",
            "--parenthood-file",
            "parenthood_jul_2024.json"
        ],
        check=True,
        shell=False,
    )
    logger.info("Done.")

    # GO Zero Shot 2024 new seqs + new labels + only added vocab terms
    logger.info("GO Zero Shot 2024 new seqs with new labels & only added vocab terms")
    subprocess.run(
        [
            "python",
            "bin/make_dataset_from_swissprot.py",
            "--latest-swissprot-file",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DATA_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TEST_DATA_PATH_ZERO_SHOT'],
            "--sequence-vocabulary",
            "new",
            "--label-vocabulary",
            "new",
            "--parenthood-file",
            "parenthood_jul_2024.json"
        ],
        check=True,
        shell=False,
    )
    logger.info("Done.")

    # Updated Supervised Train Set. pinf train seqs + all new & old labels.
    logger.info("Updated Supervised Train Set. ProteInfer train seqs with all new & old labels.")
    subprocess.run(
        [
            "python",
            "bin/make_dataset_from_swissprot.py",
            "--latest-swissprot-file",
            config["paths"]['data_paths']['LATEST_SWISSPROT_DATA_PATH'],
            "--output-file-path",
            config["paths"]['data_paths']['TRAIN_2024_DATA_PATH'],
            "--sequence-vocabulary",
            "proteinfer_train",
            "--label-vocabulary",
            "all",
            "--parenthood-file",
            "parenthood_jul_2024.json"
        ],
        check=True,
        shell=False,
    )
    logger.info("Done.")

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

    logger.info("Create smaller test sets for BLAST runtime calculation...")
    create_smaller_test_sets(df = df,
                             output_dir = project_root / "data" / "swissprot" / "proteinfer_splits" / "random"
                             )
    logger.info("Done.")
    
    logger.info("Create top labels test set for embeddings analysis.")
    create_top_labels_test_set(df = df,
                               output_path= config["paths"]['data_paths']['TEST_TOP_LABELS_DATA_PATH'],
                               go_term_distribution=go_term_distribution
                               )
    logger.info("Done.")