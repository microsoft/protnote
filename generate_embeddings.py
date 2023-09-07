
from src.data.datasets import ProteInferDataset
from torch.utils.data import DataLoader
from src.data.collators import proteinfer_collate_variable_sequence_length
from src.models.protein_encoders import ProteInfer
from src.utils.proteinfer import transfer_tf_weights_to_torch
import torch
import numpy as np
from tqdm import tqdm
import logging
from src.utils.data import save_to_pickle

#TODO: all of these paths shouldnt be here. Could use config, hydra.
FULL_DATA_PATH = '/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/full_GO.fasta'
AMINO_ACID_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json'
GO_LABEL_VOCAB_PATH = '/home/samirchar/ProteinFunctions/data/vocabularies/proteinfer_GO_label_vocab.json'
MODEL_WIEGHTS_PATH = '/home/samirchar/ProteinFunctions/models/proteinfer/GO_model_weights.pkl'
PROTEINFER_RESULTS_DIR = '/home/samirchar/ProteinFunctions/data/proteinfer_results/'
NUM_LABELS = 32102
BATCH_SIZE = 2**7

logging.basicConfig( level=logging.INFO)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device = {device}")

logging.info(f"Creating Dataset")
full_dataset = ProteInferDataset(data_path=FULL_DATA_PATH,
                            sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
                            label_vocabulary_path=GO_LABEL_VOCAB_PATH)

logging.info(f"Creating DataLoader")
full_loader = DataLoader(dataset=full_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=proteinfer_collate_variable_sequence_length)

model = ProteInfer(num_labels=NUM_LABELS,
                      input_channels=20,
                      output_channels=1100,
                      kernel_size=9,
                      activation=torch.nn.ReLU,
                      dilation_base=3,
                      num_resnet_blocks=5,
                      bottleneck_factor=0.5)

transfer_tf_weights_to_torch(model,MODEL_WIEGHTS_PATH)
model.to(device)
model = model.eval()
mapping = {}
logging.info(f"Generating embeddings")
with torch.no_grad():
    for batch_idx, (sequences,sequence_lengths,labels,sequence_ids) in tqdm(enumerate(full_loader),total=len(full_loader)):
        sequences,sequence_lengths,labels,sequence_ids = (sequences.to(device),
                                        sequence_lengths.to(device),
                                        labels.to(device),
                                        sequence_ids.to(device))
        embeddings = model.get_embeddings(sequences,sequence_lengths)

        # Loop through the batch
        for i in range(embeddings.size(0)):
            original_id = full_dataset.int2sequence_id[sequence_ids[i].item()]
            embedding_vector = embeddings[i].cpu().numpy()
            mapping[original_id] = embedding_vector

save_to_pickle(mapping,f"{PROTEINFER_RESULTS_DIR}proteinfer_embeddings.pkl")
