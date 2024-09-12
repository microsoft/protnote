import argparse
from protnote.utils.data import download_and_unzip
from protnote.utils.configs import get_project_root
if __name__ == "__main__":
    """
    Example usage:
    python download_dataset.py --url "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz" --output-file "uniprot_sprot_latest.dat"
    """
    parser = argparse.ArgumentParser(
        description="Download and unzip swissprot file from a given link."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="The URL to download the file from."
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="The relative or absolute path to save the downloaded, unzipped file (not .gz file).",
    )

    args = parser.parse_args()
    args.output_file = get_project_root() / 'data' / 'swissprot' / args.output_file

    print("Downloading and unzipping file...")
    download_and_unzip(args.url, args.output_file)
