import pandas as pd
import wget
import os
from protnote.utils.data import (
    save_to_pickle,
    get_ec_class_descriptions,
    get_ec_number_description,
)
from protnote.utils.configs import get_project_root

def main():
    output_dir = get_project_root() / 'data' / 'annotations'
    ec_classes_data = "https://ftp.expasy.org/databases/enzyme/enzclass.txt"
    ec_numbers_data = "https://ftp.expasy.org/databases/enzyme/enzyme.dat"
    enzclass_output_path = output_dir / "enzclass.txt"
    enzyme_dat_output_path = output_dir / "enzyme.dat"
    annotations_output_path = output_dir / "ec_annotations.pkl"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wget.download(ec_classes_data, out=str(enzclass_output_path))
    wget.download(ec_numbers_data, out=str(enzyme_dat_output_path))

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
    main()
