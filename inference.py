
import torch
import os
import argparse
import math
import re
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.utils.models import generate_label_embeddings_from_text
from src.utils.data import read_yaml, read_pickle,ensure_list,remove_obsolete_from_string
from src.utils.data import (
    seed_everything,
    log_gpu_memory_usage
)
from src.utils.data import Blossum62Mutations
from src.utils.main_utils import validate_arguments
from src.data.datasets import ProteinDataset, create_multiple_loaders, calculate_sequence_weights
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.losses import get_loss
from src.utils.evaluation import EvalMetrics
from src.utils.models import count_parameters_by_layer, sigmoid_bias_from_prob,load_model
from src.utils.configs import get_setup
from src.utils.data import read_json, write_json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
# ---------------------- HANDLE ARGUMENTS ----------------------#
parser = argparse.ArgumentParser(description="Inference with mutations")

parser.add_argument("--input-path", type=str, default='inference.json',
                    help="The path to input data json")

parser.add_argument("--window-size", type=int,
                    help="The size of moving window for mutations")

parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                    help="(Relative) path to the configuration file.")


parser.add_argument("--name", type=str, default="inference",
                    help="Name of the W&B run. If not provided, a name will be generated.")

parser.add_argument("--load-model", type=str, default=None,
                    help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

args = parser.parse_args()


ROOT_PATH = os.path.dirname(__file__)
TASK = "Identify the main categories, themes, or topics described in the following Gene Ontology (GO) term, which is used to detail a protein's function"

DATA_PATH = os.path.join(ROOT_PATH, "data")

OUTPUT_PATH = 'data/embeddings/inference_E5_multiling_inst_frozen_label_embeddings_mean.pt'
INDEX_OUTPUT_PATH = OUTPUT_PATH.split('.')
INDEX_OUTPUT_PATH = '_'.join([INDEX_OUTPUT_PATH[0] ,'index']) + '.'+ INDEX_OUTPUT_PATH[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = read_json(args.input_path)
names,labels_idxs = list(zip(*data['labels']))
sequence = data['sequence']

labels=synonym_exact=names

descriptions_file = pd.DataFrame({'name':names,'label':labels,'synonym_exact':synonym_exact},index=labels_idxs)

# Initialize label tokenizer and encoder
label_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
label_encoder = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').to(DEVICE)

embeddings_idx = {'id': [],'description_type': [],'description': [], 'token_count': []}
for go_term, desriptions in tqdm(
    descriptions_file[['name','label','synonym_exact']].iterrows(), total=len(descriptions_file)
):
    for desription_type, desription_set in desriptions.items():
        for description in ensure_list(desription_set):
            description = remove_obsolete_from_string(description)
            description = get_detailed_instruct(TASK,description)
            embeddings_idx['description'].append(description)
            embeddings_idx['id'].append(go_term)
            embeddings_idx['description_type'].append(desription_type)
            embeddings_idx['token_count'].append(len(label_tokenizer.tokenize(description))) # We need the token count for embedding normalization (longer descriptions will have more feature-rich embeddings)
    print(description)
# Remove Obsolete/Deprecated texts
logging.info("Extracting embeddings...")
embeddings = generate_label_embeddings_from_text(
    label_annotations=embeddings_idx['description'],
    label_tokenizer=label_tokenizer,
    label_encoder=label_encoder,
    pooling_method='mean',
    batch_size_limit=500,
    append_in_cpu=False,
    account_for_sos=True
).to('cpu')

#Convert to indexed pandas df
embeddings_idx = pd.DataFrame(embeddings_idx)

torch.save(embeddings, OUTPUT_PATH)
torch.save(embeddings_idx, INDEX_OUTPUT_PATH)


BM = Blossum62Mutations()
# Create DataSet
records=[]

#Add original sequence first
records.append(SeqRecord(Seq(sequence),id='none',description=" ".join(labels_idxs)))

for window_start in range(len(sequence)-args.window_size+1):
    window_end = window_start+args.window_size-1
    locations = set(range(window_start,window_end+1))
    sampled_sequence = BM.corrupt_sequence_at_locations(sequence,locations,'non-conservative',sample=False)
    id=f'{window_start}-{window_end}:{sequence[window_start:window_end+1]}->{sampled_sequence[window_start:window_end+1]}'
    records.append(SeqRecord(Seq(sampled_sequence),id=id,description=" ".join(labels_idxs)))

SeqIO.write(records, 'data/zero_shot/inference.fasta' , "fasta")

# Seed everything so we don't go crazy
seed_everything(42, DEVICE)

task = 'inference'
config = get_setup(
    config_path=args.config,
    run_name=args.name,
    overrides=["EXTRACT_VOCABULARIES_FROM",None],
    train_path_name=None,
    val_path_name=None,
    test_paths_names=['INFERENCE_DATA_PATH'],
    annotations_path_name = 'INFERENCE_ANNOTATIONS_PATH',
    base_label_embedding_name = 'INFERENCE_BASE_LABEL_EMBEDDING_PATH',
    amlt=False,
    is_master=False,
)
params, paths, timestamp, logger = config["params"], config[
    "paths"], config["timestamp"], config["logger"]


# Create individual datasets
test_dataset = ProteinDataset(
    data_paths=config['dataset_paths']['test'][0], 
    config=config,
    logger=logger,
    require_label_idxs=False,  # Label indices are not required for testing
    label_tokenizer=label_tokenizer,
) 

# Add datasets to a dictionary
datasets = {"test": [test_dataset]}
datasets = {k:v for k,v in datasets.items() if v[0] is not None}
label_sample_sizes = {"test": None}


logger.info("Initializing data loaders...")
# Define data loaders
loaders = create_multiple_loaders(
    datasets,
    params,
    label_sample_sizes=label_sample_sizes,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    num_workers=params["NUM_WORKERS"],
    sequence_weights=None
)

# Initialize ProteInfer
sequence_encoder = ProteInfer.from_pretrained(
    weights_path=paths[f"PROTEINFER_GO_WEIGHTS_PATH"],
    num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
    input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
    output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
    kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
    activation=torch.nn.ReLU,
    dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
    num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
    bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
)

model = ProTCL(
    # Parameters
    protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
    label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
    latent_dim=params["LATENT_EMBEDDING_DIM"],
    label_embedding_pooling_method=params["LABEL_EMBEDDING_POOLING_METHOD"],
    sequence_embedding_dropout=params["SEQUENCE_EMBEDDING_DROPOUT"],
    label_embedding_dropout=params["LABEL_EMBEDDING_DROPOUT"],
    label_embedding_noising_alpha=params["LABEL_EMBEDDING_NOISING_ALPHA"],

    # Encoders
    label_encoder=label_encoder,
    sequence_encoder=sequence_encoder,
    inference_descriptions_per_label=len(params["INFERENCE_GO_DESCRIPTIONS"].split('+')),

    # Output Layer
    output_mlp_hidden_dim_scale_factor=params["OUTPUT_MLP_HIDDEN_DIM_SCALE_FACTOR"],
    output_mlp_num_layers=params["OUTPUT_MLP_NUM_LAYERS"],
    output_neuron_bias=sigmoid_bias_from_prob(params["OUTPUT_NEURON_PROBABILITY_BIAS"]) if params["OUTPUT_NEURON_PROBABILITY_BIAS"] is not None else None,
    outout_mlp_add_batchnorm=params["OUTPUT_MLP_BATCHNORM"],
    projection_head_num_layers=params["PROJECTION_HEAD_NUM_LAYERS"],
    dropout=params["OUTPUT_MLP_DROPOUT"],
    projection_head_hidden_dim_scale_factor=params["PROJECTION_HEAD_HIDDEN_DIM_SCALE_FACTOR"],

    # Training options
    label_encoder_num_trainable_layers=0,

    # Batch size limits
    label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
    sequence_batch_size_limit=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
).to(DEVICE)



loss_fn = get_loss(config=config,
                    bce_pos_weight=1,
                    label_weights=None)

# Initialize trainer class to handle model training, validation, and testing
Trainer = ProTCLTrainer(
    model=model,
    device=DEVICE,
    config=config,
    logger=logger,
    timestamp=timestamp,
    run_name=args.name,
    use_wandb=False,
    use_amlt=False,
    loss_fn=loss_fn,
    is_master=True
)

# Log the number of parameters by layer
count_parameters_by_layer(Trainer._get_model())

# Load the model weights if --load-model argument is provided (using the DATA_PATH directory as the root)
# TODO: Process model loading in the get_setup function

load_model(
    trainer=Trainer,
    checkpoint_path=os.path.join(config["DATA_PATH"], args.load_model),
    from_checkpoint=False
)
logger.info(
    f"Loading model checkpoing from {os.path.join(config['DATA_PATH'], args.load_model)}. If training, will continue from epoch {Trainer.epoch+1}.\n")

# Initialize EvalMetrics
eval_metrics = EvalMetrics(device=DEVICE)

label_sample_sizes = {k:(v if v is not None else len(datasets[k][0].label_vocabulary)) 
                        for k,v in label_sample_sizes.items() if k in datasets.keys()}



# TODO: If best_val_th is not defined, alert an error to either provide a decision threshold or a validation datapath
test_metrics = Trainer.evaluate(
    data_loader=loaders["test"][0],
    eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*",
                                                                threshold=0.3,
                                                                num_labels=label_sample_sizes["test"]),
    save_results=True,
    metrics_prefix=f'window_{args.window_size}'
)

