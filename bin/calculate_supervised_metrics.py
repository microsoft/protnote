import pandas as pd
import obonet
import gc
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from protnote.utils.notebooks import *
from protnote.utils.configs import load_config, construct_absolute_paths



def main():
    seeds = [12, 22, 32, 42, 52]
    pinf_model_ids = [13703706, 13703742, 13703997, 13704131, 13705631]

    # Load the configuration and project root
    config, project_root = load_config()
    results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]

    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="""
            Calculate the metrics for protnote,proteinfer and blast on the same test set given predictions for each. 
            It's design for multiple predictions per model (except blast), one for each seed.
            """
        )
    parser.add_argument(
        "--proteinfer-prediction-files",
        nargs="+",
        default=['test_logits_GO_TEST_DATA_PATH_proteinfer{id}.h5' for id in pinf_model_ids],
        required=False,
        help="List of Proteinfer prediction files, potentially from different seeds",
    )
    parser.add_argument(
        "--protnote-prediction-files",
        nargs="+",
        default=[f'test_1_logits_TEST_DATA_PATH_seed_replicates_v9_{seed}_sum_last_epoch.parquet' for seed in seeds],
        required=False,
        help="List of ProtNote prediction files, potentially from different seeds",
    )

    parser.add_argument(
        "--blast-predictions-file",
        type=str,
        default="blast_pivot_parsed_test_GO_train_GO_results_fixed.parquet",
        required=False,
        help="file with blast predictions",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default='supervised_metrics_df.parquet',
        required=False,
        help="the file that will store all the metrics",
    )

    parser.add_argument(
        "--go-graph-file",
        type=str,
        default="go_jul_2019.obo",
        required=False,
        help="the file of the appropriate go graph. The version/date of the go graph should match the date of the test set",
    )

    parser.add_argument(
        "--test-set-name",
        type=str,
        default="GO 2019 Supervised",
        required=False,
        help="The name of the test set used to evaluate the predictions",
    )


    args = parser.parse_args()

    args.proteinfer_prediction_files = construct_absolute_paths(results_dir,args.proteinfer_prediction_files)
    args.protnote_prediction_files = construct_absolute_paths(results_dir,args.protnote_prediction_files)
    args.blast_predictions_file = construct_absolute_paths(results_dir,[args.blast_predictions_file])
    args.output_file = results_dir / args.output_file
    graph_2019 = obonet.read_obo(project_root / "data" / "annotations" / args.go_graph_file)
    threshold = 0.5
    device = "cpu"

    ontology2alias = {
        "molecular_function": "MF",
        "biological_process": "BP",
        "cellular_component": "CC",
        "All": "All",
    }

    models_logits = {
        "Proteinfer": args.proteinfer_prediction_files,
        "ProtNote": args.protnote_prediction_files,
        "baseline_blast": args.blast_predictions_file
    }

    models_labels = pd.read_parquet(args.protnote_prediction_files[0].replace('logits','labels'))

    metrics_df = []
    models = list(models_logits.keys())
    for model in tqdm(models):
        for file in models_logits[model]:
            print(file)
            if file.endswith(".parquet"):
                logits_df = pd.read_parquet(file)
            elif file.endswith(".h5"):
                logits_df = pd.read_hdf(file, key="logits_df", mode="r").astype(
                    "float32"
                )
                print("read logits of shape: ", logits_df.shape)

            metrics = metrics_by_go_ontology(
                logits_df, models_labels, graph_2019, device, threshold
            )

            metrics = pd.DataFrame(metrics)
            metrics["model"] = model
            metrics["test_name"] = args.args.test_set_name
            metrics.index.name = "metric"
            metrics = metrics.set_index(["model", "test_name"], append=True)
            metrics_df.append(metrics)

            del logits_df
            gc.collect()

    metrics_df = pd.concat(metrics_df)
    metrics_df.columns = metrics_df.columns.map(ontology2alias)

    metrics_df.to_parquet(args.output_file)

if __name__ == "__main__":
    main()
