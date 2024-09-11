import pandas as pd
import obonet
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.notebooks import *


def get_supervised_metrics():
    test_name = "GO 2019 Supervised"
    models_logits = {
        "Proteinfer": [
            f"outputs/results/test_logits_GO_TEST_DATA_PATH_proteinfer{id}.h5"
            for id in pinf_model_ids
        ],
        "Ours": [
            f"outputs/results/test_1_logits_TEST_DATA_PATH_seed_replicates_v9_{seed}_sum_last_epoch.parquet"
            for seed in seeds
        ],
        "baseline_blast": [
            "outputs/results/blast_pivot_parsed_test_GO_train_GO_results_fixed.parquet"
        ],
    }

    models_labels = pd.read_parquet(
        f"outputs/results/test_1_labels_TEST_DATA_PATH_{model_checkpoint}.parquet"
    )

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
            metrics["test_name"] = test_name
            metrics.index.name = "metric"
            metrics = metrics.set_index(["model", "test_name"], append=True)
            metrics_df.append(metrics)

            del logits_df
            gc.collect()

    metrics_df = pd.concat(metrics_df)
    metrics_df.columns = metrics_df.columns.map(ontology2alias)

    metrics_df.to_parquet("outputs/results/supervised_metrics_df.parquet")


plt.rcParams["font.size"] = 14

seeds = [12, 22, 32, 42, 52]
pinf_model_ids = [13703706, 13703742, 13703997, 13704131, 13705631]
model_checkpoint = "seed_replicates_v9_12_sum_last_epoch"
graph_2019 = obonet.read_obo("data/annotations/go_jul_2019.obo")
threshold = 0.5
device = "cpu"
ontology2alias = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
    "All": "All",
}

if __name__ == "__main__":
    get_supervised_metrics()
