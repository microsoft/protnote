from itertools import product
import umap
from sklearn.preprocessing import StandardScaler
from protnote.utils.data import generate_vocabularies

import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import obonet
import argparse
from tqdm import tqdm
from protnote.utils.configs import load_config

# Load the configuration and project root
config, project_root = load_config()
results_dir = config["paths"]["output_paths"]["RESULTS_DIR"]

def save_fig(name):
    plt.savefig(f"{name}.pdf", format="pdf", dpi=1200, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UMAP plot hparam search")
    parser.add_argument(
        "--n-neighbors-vals",
        nargs="+",
        type=int,
        required=False,
        default=[50, 200],
        help="n neighbors values to try",
    )
    parser.add_argument(
        "--min-dist-vals",
        nargs="+",
        type=float,
        required=False,
        default=[0.5, 0.3],
        help="min dist values to try",
    )
    parser.add_argument(
        "--paired-hparams",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--num-seqs",
        type=int,
        required=False,
        default=100,
        help="Number of sequences to consider",
    )

    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=False,
        default= results_dir / "test_1_embeddings_TEST_TOP_LABELS_DATA_PATH_normal_test_label_aug_v4/batches_0_99.pt",
        help="The .pt path of the embeddings to perform umap on.",
    )

    parser.add_argument(
        "--go-graph-file",
        type=str,
        default="go_jul_2019.obo",
        required=False,
        help="the file of the appropriate go graph. The version/date of the go graph should match the date of the test set used to generate --embeddings-path",
    )

    parser.add_argument(
        "--test-data-path",
        type=str,
        default='TEST_TOP_LABELS_DATA_PATH',
        required=False,
        help="The path name used to generate embeddings-path",
    )

    args = parser.parse_args()
    figures_dir = results_dir / "figures"
    plt.rcParams["font.size"] = 14

    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    print("reading data...")
    embeddings = torch.load(args.embeddings_path,map_location="cpu")
    joint_embedding_dim = embeddings["joint_embeddings"].shape[-1]
    num_labels = embeddings["labels"].shape[-1]
    vocab = generate_vocabularies(config['paths']['data_paths'][args.test_data_path])["label_vocab"]
    graph = obonet.read_obo(project_root / 'data' / 'annotations' / args.go_graph_file)
    vocab_parents = [
        (graph.nodes[go_term]["namespace"] if go_term in graph.nodes else "missing")
        for go_term in vocab
    ]

    print("pre processing...")
    X = embeddings["output_layer_embeddings"][: num_labels * args.num_seqs, :]
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    hparams = [args.n_neighbors_vals, args.min_dist_vals]
    num_combinations = 1

    if args.paired_hparams:
        assert len(hparams[0]) == len(
            hparams[1]
        ), "hparams must be same lenght with paired_hparams = true"
        num_combinations = len(args.n_neighbors_vals)
        combos = list(zip(*hparams))
    else:
        for hparam in hparams:
            num_combinations *= len(hparam)
        combos = product(*hparams)
    print(f"Testing {num_combinations} hparam combinations")

    hue = vocab_parents * (args.num_seqs)
    mask = [i != "missing" for i in hue]
    hue_masked = [hue_val for hue_val, mask_val in zip(hue, mask) if mask_val]
    match_binary_mask = embeddings["labels"][: args.num_seqs, :].flatten()

    palette = sns.color_palette("tab10")

    print("running umap plots...")
    for n_neighbors, min_dist in tqdm(combos, total=num_combinations):
        
        X_r = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit(X_s).embedding_

        fig = plt.figure(figsize=(7, 7))
        title = f"match vs unmatch n_neighbors={n_neighbors}, min_dist={min_dist}, n = {len(X_r)}"
        # output layer showing separation between matching and un-matching protein-function pairs
        palette_ = palette[7:8] + palette[6:7]
        sns.scatterplot(
            x=X_r[:, 0],
            y=X_r[:, 1],
            marker=".",
            s=2,
            hue=match_binary_mask,
            edgecolor=None,
            palette=palette_,
        )
        plt.legend(
            markerscale=10,
            title="Protein-Function Label",
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center",
        )
        sns.despine()
        plt.title(title)
        save_fig(os.path.join(figures_dir, title))
        plt.show()

        # Output layer colored by GO Top hierarchy
        fig = plt.figure(figsize=(7, 7))
        
        palette_ = palette[4:5] + palette[8:10]
        match_binary_mask = match_binary_mask.astype(bool) & mask
        X_r = (
            umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            .fit(X_s[match_binary_mask])
            .embedding_
        )
        title = f"top hierarchy n_neighbors={n_neighbors}, min_dist={min_dist}, , n = {len(X_r)}"
        sns.scatterplot(
            x=X_r[:, 0],
            y=X_r[:, 1],
            marker=".",
            hue=[
                hue_val
                for hue_val, mask_val, binary_mask_val in zip(
                    hue, mask, match_binary_mask
                )
                if mask_val & binary_mask_val
            ],
            s=15,
            edgecolor=None,
            palette=palette_,
        )
        plt.legend(
            markerscale=1,
            title="Ontology",
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center",
        )
        sns.despine()
        plt.title(title)
        save_fig(os.path.join(figures_dir, title))
        plt.show()
