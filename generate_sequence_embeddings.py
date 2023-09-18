
from src.data.datasets import ProteInferDataset
from torch.utils.data import DataLoader
from src.data.collators import proteinfer_collate_variable_sequence_length
from src.models.protein_encoders import ProteInfer
from src.utils.proteinfer import transfer_tf_weights_to_torch
import torch
import numpy as np
from tqdm import tqdm
import logging
from src.utils.data import save_to_pickle, read_yaml
import os

# Default to current directory if ROOT_PATH is not set
ROOT_PATH = os.environ.get('ROOT_PATH', '.')

# Load the configuration file
config = read_yaml(ROOT_PATH + '/config.yaml')
params = config['params']
paths = config['relative_paths']
embed_sequences_params = config['embed_sequences_params']

# TODO: Move relative paths to config file
FULL_DATA_PATH = os.path.join(
    ROOT_PATH, paths['FULL_DATA_PATH'])
AMINO_ACID_VOCAB_PATH = os.path.join(
    ROOT_PATH, paths['AMINO_ACID_VOCAB_PATH'])
GO_LABEL_VOCAB_PATH = os.path.join(
    ROOT_PATH, paths['GO_LABEL_VOCAB_PATH'])
MODEL_WIEGHTS_PATH = os.path.join(
    ROOT_PATH, paths['PROTEINFER_WEIGHTS_PATH'])
OUTPUT_DIR = os.path.join(ROOT_PATH, 'data/embeddings/')

# TODO: Move parameters to config file (as "embed_sequences_params")
BATCH_SIZE = params['TEST_BATCH_SIZE']

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device = {device}")

logging.info("Creating Dataset")
full_dataset = ProteInferDataset(data_path=FULL_DATA_PATH,
                                 sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                                 label_vocabulary_path=GO_LABEL_VOCAB_PATH)

logging.info("Creating DataLoader")
full_loader = DataLoader(dataset=full_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=2,
                         collate_fn=proteinfer_collate_variable_sequence_length)

model = ProteInfer(num_labels=embed_sequences_params['NUM_LABELS'],
                   input_channels=embed_sequences_params['INPUT_CHANNELS'],
                   output_channels=embed_sequences_params['OUTPUT_CHANNELS'],
                   kernel_size=embed_sequences_params['KERNEL_SIZE'],
                   activation=torch.nn.ReLU,
                   dilation_base=embed_sequences_params['DILATION_BASE'],
                   num_resnet_blocks=embed_sequences_params['NUM_RESNET_BLOCKS'],
                   bottleneck_factor=embed_sequences_params['BOTTLENECK_FACTOR'])

transfer_tf_weights_to_torch(model, MODEL_WIEGHTS_PATH)
model.to(device)
model = model.eval()

# Mapping from sequence ID to embedding vector
mapping = {}
# Mapping from sequence ID to unique integer
sequence_id_mapping = {}

logging.info("Generating embeddings...")
with torch.no_grad():
    for batch_idx, (sequences, sequence_lengths, labels, sequence_ids) in tqdm(enumerate(full_loader), total=len(full_loader)):
        sequences, sequence_lengths, labels, sequence_ids = (sequences.to(device),
                                                             sequence_lengths.to(
                                                                 device),
                                                             labels.to(device),
                                                             sequence_ids.to(device))
        embeddings = model.get_embeddings(sequences, sequence_lengths)

        # Loop through the batch
        for i in range(embeddings.size(0)):
            original_id = full_dataset.int2sequence_id[sequence_ids[i].item()]
            embedding_vector = embeddings[i].cpu().numpy()
            mapping[original_id] = embedding_vector
            sequence_id_mapping[original_id] = len(sequence_id_mapping)

save_to_pickle(
    mapping, f"{OUTPUT_DIR}frozen_proteinfer_sequence_embeddings.pkl")
logging.info(
    f"Saved embeddings to {OUTPUT_DIR}frozen_proteinfer_sequence_embeddings.pkl")

# Save mapping from ID to unique integer
save_to_pickle(sequence_id_mapping,
               f"{OUTPUT_DIR}sequence_id_map.pkl")
logging.info(
    f"Saved sequence ID map to to {OUTPUT_DIR}frozen_proteinfer_sequence_id_map.pkl")
