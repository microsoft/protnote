import argparse
import gzip
import wget
from pathlib import Path
from src.utils.data import download_and_unzip

if __name__ == "__main__":
    """
    Example usage:
    python download_dataset.py "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz" "data/swissprot/uniprot_sprot.dat"
    """
    parser = argparse.ArgumentParser(
        description="Download and unzip a file from a given link."
    )
    parser.add_argument("url", type=str, help="The URL to download the file from.")
    parser.add_argument(
        "output_file",
        type=str,
        help="The relative or absolute path to save the downloaded, unzipped file (not .gz file).",
    )

    args = parser.parse_args()

    print("Downloading and unzipping file...")
    download_and_unzip(args.url, args.output_file)
