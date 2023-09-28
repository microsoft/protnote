import obonet
import pandas as pd
import argparse
import logging
import re

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
        definition = re.sub(r'\s*\[.*?\]\s*', '', definition)
        definition = definition.strip('"')

    return definition


def download_and_process_obo(url: str, output_file: str):
    """
    Download the OBO file from the specified URL and save the GO ID and label to a pickle.
    """
    logging.info("Downloading and processing OBO file...")

    # Load the .obo file directly from the URL into a networkx graph using obonet
    graph = obonet.read_obo(url, ignore_obsolete=False)

    # Convert the graph nodes (terms) into a pandas dataframe
    df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')

    logging.info("Calculating labels...")
    # Create a new column called "label"
    df["label"] = df.apply(calculate_label, axis=1)

    # Filter the dataframe to retain only 'label' column, with the 'id' column as the index
    df_filtered = df[['label']]

    # Save the filtered dataframe as a pickle
    df_filtered.to_pickle(output_file)

    logging.info(f"Saved filtered dataframe as a pickle to {output_file}")


if __name__ == "__main__":
    """
    2019 Option: python download_GO_annotations.py http://release.geneontology.org/2019-07-01/ontology/go.obo data/annotations/go_annotations_2019_07_01.pkl
    2023 Option: python download_GO_annotations.py http://release.geneontology.org/2023-07-27/ontology/go.obo data/annotations/go_annotations_2023_07_27.pkl
    """
    # TODO: Make more general so accepts any obo URL
    # TODO: output path should be enforced for standardization. e.g. final pkl should always be in data/annotations/.
    # TODO: Filename can be figured out from URL
    parser = argparse.ArgumentParser(
        description="Download OBO file and save GO ID and label to a pickle.")
    parser.add_argument("url", type=str,
                        help="URL to the OBO file.")
    parser.add_argument("output_file", type=str,
                        help="Path to save the resulting pickle file.")
    args = parser.parse_args()

    # NOTE: The supplement here (https://google-research.github.io/proteinfer/latex/supplement.pdf) says that they used "2019_01", which is ambiguous. From trial-and-error, it looks like they used 2019-07-01
    download_and_process_obo(args.url, args.output_file)
