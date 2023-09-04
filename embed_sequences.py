import requests
import argparse
import pandas as pd
from Bio import SeqIO
import logging

logging.basicConfig(filename='embeddings.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# counter
total_sequences = 0

def get_sequence_embeddings_from_api(fasta_file):
    """
    Extract sequences from the FASTA file and get their embeddings using the API.
    """
    global total_sequences

    # Extract sequences from the FASTA file, but only the first 3 sequences
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

    # Remove duplicates
    logging.info(f"Removing {len(sequences) - len(list(set(sequences)))} duplciates...")
    sequences = list(set(sequences))   

    # Call the API to get embeddings
    url = "http://127.0.0.1:5000/get_embeddings"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'protein_sequences': sequences
    }

    logging.info(f"Calling API for {fasta_file}...")
    print(f"Calling API for {fasta_file}... Look to the API console for progress updates.")

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        logging.info("Success! Returned with status code 200.")
        embeddings_response  = response.json()["embeddings"]
        embeddings = [(seq, embeddings_response[seq]['0']) for seq in sequences if seq in embeddings_response]
        total_sequences += len(embeddings)
        logging.info(f"Returned {len(embeddings)} embeddings for {fasta_file}. Total sequences processed so far: {total_sequences}")
        return embeddings
    else:
        logging.error(f"Error: {response.status_code}")
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    """
    Example usage: python embed_sequences.py --input_files data/swissprot/proteinfer_splits/random/dev_GO.fasta data/swissprot/proteinfer_splits/random/train_GO.fasta data/swissprot/proteinfer_splits/random/test_GO.fasta --output_file data/swissprot/proteinfer_splits/random/sequence_embeddings.pk1
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for protein sequences from multiple FASTA files.")
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="Paths to the FASTA files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the combined dataframe.")
    args = parser.parse_args()

    # Initialize an empty list to store sequence-embedding pairs
    all_data = []

    # Loop through the input files and get embeddings
    for input_file in args.input_files:
        embeddings = get_sequence_embeddings_from_api(input_file)
        if embeddings:
            all_data.extend(embeddings)

    # Convert the list of sequence-embedding pairs to a DataFrame
    combined_df = pd.DataFrame(all_data, columns=['sequence', 'embedding'])

    # Save the combined dataframe to the specified output file
    if not combined_df.empty:
        combined_df.to_pickle(args.output_file)
        logging.info(f"Saved {total_sequences} combined embeddings to {args.output_file}")
    else:
        print("Failed to get embeddings.")