from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()

setup(
    name="protnote",
    version="1.0.0",
    author="Samir Char & Nate Corley",
    packages=find_packages(),
    include_package_data=True,
    description="ProtNote: a multimodal method for protein-function annotation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/ProteinFunctions", #TODO: Change repo name
    install_requires=[
        "torch==2.2.0",  # PyTorch
        "torchvision==0.15.2",  # TorchVision
        # "pytorch-cuda==11.8", #PyTorch CUDA
        "pandas==1.5.2",  # Pandas
        "joblib==1.2.0",  # Joblib
        "transformers==4.38.0",  # Huggingface Transformers
        "torchmetrics==1.2.0",  # PyTorch metrics
        "torchdata==0.7.1",  # PyTorch Data
        "wandb==0.15.11",  # Weights and Biases
        "sacremoses==0.0.53",  # Tokenizer
        "pynvml==11.5.0",  # Nvidia management library
        "torcheval==0.0.7",  # PyTorch evaluation
        "wget==3.2",  # wget
        "azureml-mlflow==1.53.0",  # Azure MLflow
        "loralib==0.1.2",  # LoRA library
        "tensorboard==2.15.1",  # Tensorboard
        "obonet==1.0.0",  # OBO ontologies
        "blosum==2.0.2",  # BLOSUM scoring matrices
        "biopython==1.84",  # Biopython for bioinformatics
        "ipykernel==6.29.5", #Jupyter notebook
        "scipy==1.13.1", # Viz
        "seaborn==0.13.2", # Viz
        "scikit-learn==1.5.0",
        "matplotlib==3.9.2", # Viz
        "umap-learn==0.5.4" # Viz
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)