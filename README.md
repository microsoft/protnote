# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Instalation
```
git clone https://github.com/microsoft/protnote.git
cd protnote
conda env create -f environment.yml
conda activate protnote
pip install -e ./  # make sure ./ is the dir including setup.py
```

## Config
Most hyperparameters and paths are managed through base_config.yaml. Whenever reasonable we enforce certain files to be in specific directories to increase consistency and reproducibility. In general, we adhere to the following data argument naming conventions in scripts: 
* Argument ending in "dir" corresponds to the full path of the folder where a file is located. E.g., data/swissprot/
* Argument ending in "path" corresponds to the full file path. E.g., data/swissprot/myfile.fasta
* Argument ending in "file" corresponds to the full file name alone (including the extension). E.g., myfile.fasta. This is used for files with enforced location within the data folder structure

## Get model predictions for all datasets
Get the predictions of the selected models for all the specified datasets. Useful to get the predictions of the same model with different seeds. Warning: the script is designed such that all models have the same hyperparameters but the only difference lies on the weights. 


```
python test_models.py --model-paths \
    models/ProtNote/seed_replicates_v9_12_sum_last_epoch.pt \
    models/ProtNote/seed_replicates_v9_22_sum_last_epoch.pt \
    models/ProtNote/seed_replicates_v9_32_sum_last_epoch.pt \
    models/ProtNote/seed_replicates_v9_42_sum_last_epoch.pt \
    models/ProtNote/seed_replicates_v9_52_sum_last_epoch.pt \
    --test-paths-names "TEST_DATA_PATH_ZERO_SHOT" "TEST_EC_DATA_PATH_ZERO_SHOT" \
    --save-prediction-results
```

## Data



## ProteInfer data
ProteInfer models and datasets


## Reproducing 
Run the following instructions in order to avoid any dependency issues.

### Data
We provide all the data required to run ProtNote and reproduce our results, but if you insist, this section explains how to download, process and create all the datasets. 

### ProteInfer Data
Perform the following stepts to download the original ProteInfer dataset TFRecords:

Install gcloud:

```
sudo snap install google-cloud-cli --classic
```

Then, login with a google account (e.g., gmail). The following command with promp a browser window for authentication:

```
gcloud init
```

Download the data:

```
gsutil -m cp -r gs://brain-genomics-public/research/proteins/proteinfer/datasets/swissprot .
```

Move the random and clustered folders to the directory data/swissprot/proteinfer_splits/

To create the fasta versions of these files run the following commands from root:

```
python bin/make_proteinfer_dataset.py --dataset-type random --annotation-types GO
python bin/make_proteinfer_dataset.py --dataset-type random --annotation-types EC
```

### Download annotations
Download GO and EC annotations.

TODO: 

Need to add command to download go annotations. Are the go annotations from Jul 2023 or Jul 2024?? need to clarify this.

I believe I downloaded not from the releases, but whatever was latest on Jul 2024. Currently as of Sept 9 2024, the latest version in releases is of 06/17/2024. Maybe, my latest download of Jul 2024 corresponds to this June release. Otherwise, I might need to just provide the file i used since it wouldn't be possible to reproduce if the same file can't be downloaded from go.

Need to fix problem of wget not having permission to retrieve data from the website.


```
python bin/download_EC_annotations.py
```


### Zero Shot datasets, and few more
Run ```python bin/create_test_sets.py``` to create all the remaining datasets used in the paper. The datasets are:

*
* 
*
*
*
*
*
* Fake zero shot datasets. Includes the train, val, and test sets with the prefix "fake_"


### Cached embeddings


### Models
Example of two ProteInfer models. One is gor GO annotation prediction, the other is for EC number predictions. There are multiple models with this format in their cloud storage with different id's corresponding to distinct seeds. The seed is the 8-digit number before the file extension:

* https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13703706.tar.gz
* https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/noxpd2_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13703966.tar.gz

To dowlnoad and get the predictions for five ProteInfer seeds, run:

```
python bin/download_and_test_proteinfer_seeds.py --get-predictions
```