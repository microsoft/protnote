import pandas as pd
import re
import os
import wget
import argparse
from src.utils.data import (
    read_pickle,
    save_to_pickle,
    get_ec_class_descriptions,
    get_ec_number_description,
)


def main(output_dir: str):
    ec_classes_data = "https://ftp.expasy.org/databases/enzyme/enzclass.txt"
    ec_numbers_data = "https://ftp.expasy.org/databases/enzyme/enzyme.dat"
    enzclass_output_path = os.path.join(output_dir, "enzclass.txt")
    enzyme_dat_output_path = os.path.join(output_dir, "enzyme.dat")
    annotations_output_path = os.path.join(output_dir, "ec_annotations.pkl")

    wget.download(ec_classes_data, out=enzclass_output_path)
    wget.download(ec_numbers_data, out=enzyme_dat_output_path)

    ec_classes = get_ec_class_descriptions(enzclass_output_path)
    ec_numbers = get_ec_number_description(enzyme_dat_output_path, ec_classes)

    ec_annotations = pd.concat(
        [
            pd.DataFrame.from_records(list(ec_classes.values())),
            pd.DataFrame.from_records(ec_numbers),
        ],
        axis=0,
    )[["ec_number", "label"]]

    ec_annotations["ec_number"] = "EC:" + ec_annotations["ec_number"]
    ec_annotations.set_index("ec_number", inplace=True)

    ec_annotations.index.name = None
    ec_annotations["name"] = ec_annotations["synonym_exact"] = ec_annotations["label"]
    ec_annotations["synonym_exact"] = ec_annotations["synonym_exact"].apply(
        lambda x: [x]
    )

    save_to_pickle(ec_annotations, annotations_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download OBO file and save GO ID and label to a pickle."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save the resulting pickle file.",
    )
    args = parser.parse_args()

    main(args.output_dir)
