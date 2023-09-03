import obonet
import pandas as pd
import argparse
from src.utils.PubMedBERT import load_PubMedBERT, get_PubMedBERT_embedding
from tqdm import tqdm

def add_out_degree(df):
    """
    Add a column 'out_degree' to the GO annotations dataframe. Nodes with an `out_degree` of 0 are leaf nodes.
    """
    # Load the Gene Ontology graph
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    graph = obonet.read_obo(url)

    # Create a new column 'out_degree' and populate it with the out_degree of each GO term and convert to integer
    tqdm.pandas(desc="Adding out_degree")
    df['out_degree'] = df['go_id'].progress_apply(lambda go_id: graph.out_degree(go_id) if go_id in graph else None).astype('Int64')

    return df

def embed_go_annotations(tokenizer, model, df, batch_size=32):
    """
    Embed the GO terms using PubMedBERT.
    """
    embeddings = []

    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding GO terms"):
        batch_texts = df['annotation_class_label'].iloc[i:i+batch_size].tolist()
        batch_embeddings = get_PubMedBERT_embedding(tokenizer, model, batch_texts)
        embeddings.extend(batch_embeddings)

    # Assign the embeddings to the dataframe
    df['embedding'] = embeddings

    return df

if __name__ == "__main__":
    """
    Example usage: python embed_go_annotations.py data/go_annotations.csv data/label_embeddings.pk1
    Save the modified df as a pickle file
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add out_degree column to GO annotations CSV.")
    parser.add_argument("go_annotations_path", type=str, help="Path to the GO annotations CSV file.")
    parser.add_argument("output_path", type=str, help="Path to save the pickle.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding GO terms.")
    args = parser.parse_args()

    # Load PubMedBERT
    (tokenizer, model) = load_PubMedBERT()

    # Load the GO annotations CSV
    df = pd.read_csv(args.go_annotations_path)

    # Add the out degree for each observation
    df = add_out_degree(df)

    # Embed the GO term using PubMedBERT with the specified batch size
    df = embed_go_annotations(tokenizer, model, df, batch_size=args.batch_size)

    # Save the modified dataframe to the specified output path as a pickle file
    print(f"Saving embedded GO annotations to {args.output_path}...")
    df.to_pickle(args.output_path)

    # Print a message to indicate that the file has been saved
    print(f"Saved embedded GO annotations to {args.output_path}")