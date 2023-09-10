import pandas as pd
import argparse
from src.utils.PubMedBERT import load_PubMedBERT, get_PubMedBERT_embedding
from src.utils.data import read_pickle, save_to_pickle
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def embed_go_annotations(tokenizer, model, df, batch_size=32):
    """
    Embed the GO terms using PubMedBERT.
    """
    mapping = {}

    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding GO terms"):
        batch_texts = df['label'].iloc[i:i + batch_size].tolist()
        batch_go_ids = df.index[i:i + batch_size].tolist()
        batch_embeddings = get_PubMedBERT_embedding(
            tokenizer, model, batch_texts)

        for go_id, embedding in zip(batch_go_ids, batch_embeddings):
            mapping[go_id] = embedding

    return mapping


if __name__ == "__main__":
    """
    Example usage: python generate_go_embeddings.py data/annotations/go_annotations_2019_07_01.pkl data/embeddings/frozen_PubMedBERT_label_embeddings.pkl
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Embed GO annotations using PubMedBERT.")
    parser.add_argument("go_annotations_path", type=str,
                        help="Path to the GO annotations pkl file.")
    parser.add_argument("output_path", type=str,
                        help="Path to save the pickle file.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding GO terms.")
    args = parser.parse_args()

    # Load PubMedBERT
    (tokenizer, model) = load_PubMedBERT()

    # Load the GO annotations
    df = read_pickle(args.go_annotations_path)

    # Embed the GO term using PubMedBERT with the specified batch size
    mapping = embed_go_annotations(
        tokenizer, model, df, batch_size=args.batch_size)

    save_to_pickle(
        mapping, args.output_path)
    logging.info(
        f"Saved embeddings to {args.output_path}.")
