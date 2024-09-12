import obonet
import pandas as pd
import argparse
import logging
import re
import numpy as np
from protnote.utils.configs import get_project_root

logging.basicConfig(level=logging.INFO)


def calculate_label(row):
    """
    Helper function to calculate the label for a given row.
    Returns the definition of the row with any text between brackets removed.
    """
    definition = row.get("def", None)

    # Remove any text between brackets, e.g., PubMed citations
    # Remove leading and trailing quotation marks
    if definition is not None:
        definition = re.sub(r"\s*\[.*?\]\s*", "", definition)
        definition = definition.strip('"')

    return definition


def process_synonyms(row) -> dict:
    """extracts the synonyms of a GO Annotation

    :param row: Row of GO annotation dataset
    :type row: _type_
    :return: dict
    :rtype: lists of synonyms for relevant scopes
    """
    if row is np.nan or not row:
        return {
            "synonym_exact": [],
            "synonym_narrow": [],
            "synonym_related": [],
            "synonym_broad": [],
        }

    scopes = {"EXACT": [], "NARROW": [], "RELATED": [], "BROAD": []}
    for synonym in row:
        match = re.search(r"\"(.+?)\"\s+(EXACT|NARROW|RELATED|BROAD)\s+\[", synonym)
        if match:
            text, scope = match.groups()
            scopes[scope].append(text)

    return {
        "synonym_exact": scopes["EXACT"],
        "synonym_narrow": scopes["NARROW"],
        "synonym_related": scopes["RELATED"],
        "synonym_broad": scopes["BROAD"],
    }


def main(url: str, output_file: str):
    """
    Download the OBO file from the specified URL and save the GO ID and label to a pickle.
    """
    logging.info("Downloading and processing OBO file...")

    output_file = get_project_root() / 'data' / 'annotations' / output_file
    # Load the .obo file directly from the URL into a networkx graph using obonet
    graph = obonet.read_obo(url, ignore_obsolete=False)

    # Convert the graph nodes (terms) into a pandas dataframe
    df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")

    logging.info("Calculating labels...")
    # Create a new column called "label"
    df["label"] = df.apply(calculate_label, axis=1)

    # Extract synonyms to augment dataset
    df_synonyms = df["synonym"].apply(process_synonyms)
    df_synonyms = pd.DataFrame(df_synonyms.tolist(), index=df.index)

    # Merge the new columns back to the original DataFrame with the same index
    df = pd.concat([df, df_synonyms], axis=1)

    # Filter the dataframe to retain only 'label', 'name' and 'synonym' columns, with the 'id' column as the index
    df_filtered = df[["label", "name"] + list(df_synonyms.columns) + ["is_obsolete"]]

    # Save the filtered dataframe as a pickle
    df_filtered.to_pickle(output_file)

    logging.info(f"Saved filtered dataframe as a pickle to {output_file}")


if __name__ == "__main__":
    """
    2019 Option: python download_GO_annotations.py http://release.geneontology.org/2019-07-01/ontology/go.obo go_annotations_2019_07_01.pkl
    2023 Option: python download_GO_annotations.py http://release.geneontology.org/2023-07-27/ontology/go.obo go_annotations_2023_07_27.pkl
    """
    # TODO: Filename can be figured out from URL
    parser = argparse.ArgumentParser(
        description="Download OBO file and save GO ID and label to a pickle."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://release.geneontology.org/2023-07-27/ontology/go.obo",
        help="URL to the OBO file.",
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save the resulting pickle file."
    )
    args = parser.parse_args()

    main(args.url, args.output_file)
