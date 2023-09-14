import argparse
import logging
import os
import torch
from tqdm import tqdm
from src.utils.data import read_pickle, save_to_pickle
from src.utils.models import load_model_and_tokenizer, tokenize_inputs, get_embeddings_from_tokens
import time

# Set the TOKENIZERS_PARALLELISM environment variable to False
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO)


def embed_go_annotations(tokenizer, model, df, batch_size=32):
    """
    Embed the GO terms using the provided model.
    """
    mapping = {}
    timer = time.time()
    for i in tqdm(range(0, len(df), batch_size), desc="Embedding GO terms"):
        batch_texts = df['label'].iloc[i:i + batch_size].tolist()
        batch_go_ids = df.index[i:i + batch_size].tolist()

        # Tokenize the batch_texts
        tokens = tokenize_inputs(tokenizer, batch_texts)

        # Get embeddings for the tokens
        embeddings = get_embeddings_from_tokens(model, tokens)

        # Convert the embeddings to numpy and map them to the GO ids
        batch_embeddings = embeddings.cpu().numpy()
        for go_id, embedding in zip(batch_go_ids, batch_embeddings):
            mapping[go_id] = embedding

    logging.info(
        f"Finished embedding GO terms in {time.time() - timer} seconds.")

    return mapping


if __name__ == "__main__":
    """
    Example usage: python generate_go_embeddings.py data/annotations/go_annotations_2019_07_01.pkl data/embeddings/frozen_PubMedBERT_label_embeddings.pkl
    """
    torch.cuda.empty_cache()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Embed GO annotations using a specified model.")
    parser.add_argument("go_annotations_path", type=str,
                        help="Path to the GO annotations pkl file.")
    parser.add_argument("output_path", type=str,
                        help="Path to save the pickle file.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding GO terms.")
    args = parser.parse_args()

    # TODO: Make this into an argument for the script, with the default to PubMedBERT
    checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    # Load the GO annotations
    df = read_pickle(args.go_annotations_path)

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(checkpoint)

    # Embed the GO term using the provided model with the specified batch size
    mapping = embed_go_annotations(
        tokenizer, model, df, batch_size=args.batch_size)

    save_to_pickle(mapping, args.output_path)
    logging.info(f"Saved embeddings to {args.output_path}.")
