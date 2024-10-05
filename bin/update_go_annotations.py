from protnote.utils.data import read_pickle, save_to_pickle
import argparse
import pandas as pd


def update_annotations(old_file_path, new_file_path, output_file_path):
    """Updates old annotations using the new annotations file and saves the updated file."""
    # Load new and old annotations
    new_annotations = read_pickle(new_file_path)
    old_annotations = read_pickle(old_file_path)

    # Find added terms (terms in new but not in old)
    added_terms = set(new_annotations.index) - set(old_annotations.index)

    # Update old annotations by adding new terms
    old_updated = pd.concat([new_annotations.loc[added_terms], old_annotations])

    #Sort old_updated by index
    old_updated = old_updated.sort_index()


    # Save the updated annotations to the output file path
    save_to_pickle(old_updated,output_file_path)

    print(f"Updated annotations saved to {output_file_path}")

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Update old GO annotations file using a new one.")
    
    # Add arguments for file paths
    parser.add_argument('--old-annotations-file-path', required=True, help='Path to the old annotations file')
    parser.add_argument('--new-annotations-file-path', required=True, help='Path to the new annotations file')
    parser.add_argument('--output-file-path', required=True, help='Path to save the updated annotations file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the update_annotations function with the provided arguments
    update_annotations(args.old_annotations_file_path, args.new_annotations_file_path, args.output_file_path)

if __name__ == "__main__":
    main()
