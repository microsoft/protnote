import pandas as pd
from protnote.utils.evaluation import metrics_per_label_df
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from protnote.utils.evaluation import EvalMetrics
from protnote.utils.data import ec_number_to_code
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
from collections import Counter


def complete_blast_preds(blast_df: pd.DataFrame, labels: list, seqs: list):
    blast_cols = set(blast_df.columns)

    # Add labels that blast missed
    blast_df[list(set(labels) - blast_cols)] = -15.0

    # Update blast columns
    blast_cols = set(blast_df.columns)
    blast_cols = [label for label in labels if label in blast_cols]

    # Consider only the sequences    in seqs, and add
    # sequences that blast missed
    blast_df = blast_df[blast_cols].reindex(seqs).fillna(-15.0)

    return blast_df


def get_metrics(logits_df, labels_df, device, threshold):
    logits = torch.tensor(logits_df.values, device=device)
    labels = torch.tensor(labels_df.values, device=device)
    probabilities = torch.sigmoid(logits)

    eval_metrics = EvalMetrics(device)
    selected_eval_metrics = eval_metrics.get_metric_collection_with_regex(
        pattern="f1_m.*", threshold=threshold, num_labels=labels.shape[-1]
    )
    mAP_micro = BinaryAUPRC(device=device)
    mAP_macro = MultilabelAUPRC(device=device, num_labels=labels.shape[-1])

    selected_eval_metrics(probabilities, labels)
    mAP_micro.update(probabilities.flatten(), labels.flatten())
    mAP_macro.update(probabilities, labels)

    metrics = {
        "mAP Macro": mAP_macro.compute(),
        "mAP Micro": mAP_micro.compute(),
        **selected_eval_metrics.compute(),
    }
    return {k: v.item() for k, v in metrics.items()}


def get_ontology_from_parenthood(go_term, parenthood):
    go_term_to_ontology = {
        "GO:0008150": "biological_process",
        "GO:0003674": "molecular_function",
        "GO:0005575": "celular_component",
    }
    for parent in parenthood[go_term]:
        if parent in go_term_to_ontology:
            ontology = go_term_to_ontology[parent]
            break
    return ontology


def filter_by_go_ontology(ontology: str, df: pd.DataFrame, graph=None, parenthood=None):
    assert (graph is not None) ^ (parenthood is not None)

    if graph is not None:
        col_mask = [
            (graph.nodes[go_term]["namespace"] if go_term in graph.nodes else "missing")
            for go_term in df.columns
        ]
    if parenthood is not None:
        col_mask = [
            get_ontology_from_parenthood(go_term, parenthood) for go_term in df.columns
        ]

    assert ontology in [
        "All",
        "biological_process",
        "cellular_component",
        "molecular_function",
    ]
    if ontology == "All":
        return df
    filtered_df = df.iloc[:, [ontology == i for i in col_mask]]
    return filtered_df


def filter_by_ec_level_1(ontology: str, df: pd.DataFrame, ec_class_descriptions: dict):
    ec_number_to_ec_level_1 = lambda ec_number: (ec_number_to_code(ec_number)[0], 0, 0)
    col_mask = [
        ec_class_descriptions[ec_number_to_ec_level_1(ec_number)]["label"]
        for ec_number in df.columns
    ]
    if ontology == "All":
        return df
    filtered_df = df.iloc[:, [i == ontology for i in col_mask]]
    return filtered_df


def metrics_by_go_ontology(df_logits, df_labels, graph, device, threshold):
    results = {}
    for ontology in [
        "All",
        "biological_process",
        "cellular_component",
        "molecular_function",
    ]:
        filtered_df_logits = filter_by_go_ontology(ontology, df_logits, graph)
        filtered_df_labels = filter_by_go_ontology(ontology, df_labels, graph)
        results[ontology] = get_metrics(
            filtered_df_logits, filtered_df_labels, device=device, threshold=threshold
        )
    return results


def metrics_by_ec_level_1(
    df_logits, df_labels, ec_class_descriptions, device, threshold
):
    results = {}
    ec_level_1s = [ec_class_descriptions[(i, 0, 0)]["label"] for i in range(1, 8)]
    for ec_level_1 in ["All"] + ec_level_1s:
        filtered_df_logits = filter_by_ec_level_1(
            ontology=ec_level_1,
            df=df_logits,
            ec_class_descriptions=ec_class_descriptions,
        )
        filtered_df_labels = filter_by_ec_level_1(
            ontology=ec_level_1,
            df=df_labels,
            ec_class_descriptions=ec_class_descriptions,
        )
        results[ec_level_1] = get_metrics(
            filtered_df_logits, filtered_df_labels, device=device, threshold=threshold
        )
    return results


def save_fig(name):
    plt.savefig(f"{name}.pdf", format="pdf", dpi=1200, bbox_inches="tight")


def plot_axes_add_letter_index(axes, loc=(0.1, 0.9)):
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(axes.flat):
        ax.text(
            *loc,
            f"{letters[i]})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )


def plot_category_performance(
    metrics_df: pd.DataFrame,
    test_name: str,
    metric: str,
    category_name: str,
    ylim: tuple = (0, 0.5),
    rotate_x_ticks: bool = False,
    pltshow=True,
    savefig=True,
    name: str = None,
    ax=None,
    figsize=None,
    palette=None,
):
    plot_df = (
        metrics_df.query("metric == @metric and test_name == @test_name")
        .melt(ignore_index=False, var_name=[category_name], value_name=metric)
        .reset_index()
    )
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=plot_df,
        x=category_name,
        y=metric,
        alpha=0.8,
        errorbar="sd",
        capsize=0.1,
        # err_kws={'linewidth':1.3},
        hue="model",
        ax=ax,
        palette=palette,
    )
    sns.stripplot(
        data=plot_df,
        x=category_name,
        y=metric,
        hue="model",
        edgecolor="black",
        # size=3,
        linewidth=0.5,
        alpha=0.8,
        palette=palette,
        dodge=True,
    )

    sns.despine()
    plt.ylim(*ylim)
    plt.title(f"{metric} - {test_name}")
    plt.legend(title="Model", bbox_to_anchor=(0.5, -0.2), loc="upper center")
    if rotate_x_ticks:
        plt.xticks(rotation=30)
    if savefig:
        assert name is not None, "Must add a name if saving the figure"
        save_fig(name)
    if pltshow:
        plt.show()


# Define threshold to calculate threshold-dependent metrics otherwise only threshold-agnostic metrics are calculated
def _get_metrics_by_label_and_freq(
    logits_df, labels_df, train_go_term_distribution, quantiles, threshold, device
):
    res = metrics_per_label_df(logits_df, labels_df, device=device, threshold=threshold)
    res["Frequency"] = res.index.map(train_go_term_distribution)

    # Bin frequencies
    freq_bins, freq_bin_edges = pd.qcut(
        train_go_term_distribution,
        q=quantiles,
        duplicates="drop",
        precision=0,
        retbins=True,
        labels=None,
    )

    res["Frequency Bin"] = res.index.map(freq_bins)
    # res['Frequency Edges']=res.index.map(freq_bin_edges)
    return res, freq_bins, freq_bin_edges


def get_metrics_by_label_and_freq(
    models, train_go_term_distribution, quantiles, threshold, device
):
    res_dfs = []
    for model, data in models.items():
        print(model)
        logits_df = data["logits_df"]
        labels_df = data["labels_df"]

        res, freq_bins, freq_bin_edges = _get_metrics_by_label_and_freq(
            logits_df=logits_df,
            labels_df=labels_df,
            train_go_term_distribution=train_go_term_distribution,
            quantiles=quantiles,
            threshold=threshold,
            device=device,
        )

        # Combine both dataframes
        res["model"] = model
        res_dfs.append(res)

    res_df = pd.concat(res_dfs, axis=0)

    # res_pivot = res_df.pivot(columns=['model'],values=[metric,'Frequency'])
    # res_pivot.columns = [i[0]+'_'+i[1] for i in res_pivot.columns]

    return res_df, freq_bins, freq_bin_edges


def plot_metric_by_label_freq(
    models, train_go_term_distribution, metric, quantiles, threshold, device
):
    res_df, freq_bins, freq_bin_edges = get_metrics_by_label_and_freq(
        models=models,
        train_go_term_distribution=train_go_term_distribution,
        quantiles=quantiles,
        threshold=threshold,
        device=device,
    )
    res_df.dropna(subset=[metric], inplace=True)
    freq_bins_pct = freq_bins.value_counts() * 100 / len(train_go_term_distribution)
    fig, ax = plt.subplots(figsize=(15, 6))

    # Annotate bars with the percentage of observations
    for index, value in enumerate(freq_bins_pct.sort_index().values):
        ax.text(
            index,
            ax.get_ylim()[1] * 0.01 + max(res_df[metric]) * 0.01,
            f"{value:.2f}%",
            ha="center",
        )

    sns.barplot(
        data=res_df.reset_index(drop=True),
        x="Frequency Bin",
        y=metric,
        alpha=0.8,
        errorbar=("ci", 95),
        hue="model",
    )
    ax.set(
        title=f"Individual label performance ({metric}) by label frequency quantiles",
        xlabel="Frequency of GO Function Annotation",
        ylabel=metric,
    )
    sns.despine()
    plt.ylim(0, 1)
    plt.show()


def get_data_distributions(data: pd.DataFrame):
    labels = Counter()
    vocab = set()
    amino_freq = Counter()
    for idx, row in data.iterrows():
        sequence = row["sequence"]
        row_labels = row["labels"]
        aa_list = list(sequence)
        if row_labels == "":
            print(row["id"], row["labels"])
        vocab.update(aa_list)
        amino_freq.update(aa_list)
        labels.update(row_labels.split(" "))
    return vocab, amino_freq, labels
