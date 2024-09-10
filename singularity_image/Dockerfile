# vim: ft=mako

FROM singularitybase.azurecr.io/base/job/pytorch/acpt-2.0.1-py3.10-cuda11.8:20240416T115414238 as base

FROM base

RUN which mamba || conda install -yq -c conda-forge mamba
RUN mamba install -yq -c biobuilds -c pytorch -c huggingface -c nvidia -c anaconda -c conda-forge -c defaults pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 pandas=1.5.2 joblib=1.1.1 transformers=4.32.1 torchmetrics=1.2.0 torchdata=0.7.1 wandb=0.15.11 sacremoses=0.0.53 pynvml=11.5.0 torchmetrics=1.2.0 && conda clean -yqa
RUN python -mpip install -q  torcheval==0.0.7 wget==3.2 azureml-mlflow==1.53.0 loralib==0.1.2 tensorboard==2.15.1 obonet==1.0.0 blosum==2.0.2 biopython==1.84

# set validation arguments for expected use of this image
ENV SINGULARITY_IMAGE_FRAMEWORK=pytorch
ENV SINGULARITY_IMAGE_ACCELERATOR=NVIDIA

