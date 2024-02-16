
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
from src.utils.configs import generate_label_embeddeing_path
from src.utils.data import read_yaml, read_pickle


logging.basicConfig(level=logging.INFO)

### SETUP ###
torch.cuda.empty_cache()


def ensure_list(value):
    # Case 1: If the value is already a list
    if isinstance(value, list):
        return value
    # Case 2: If the value is NaN
    elif value is math.nan or (isinstance(value, float) and math.isnan(value)):
        return []
    # Case 3: For all other cases (including strings)
    else:
        return [value]

def remove_obsolete_from_string(text):
    pattern= r'(?i)\bobsolete\.?\s*'
    return re.sub(pattern, '', text)

def main():
    # ---------------------- HANDLE ARGUMENTS ----------------------#
    parser = argparse.ArgumentParser(description="Train and/or Test the ProTCL model.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="(Relative) path to the configuration file.",
    )

    parser.add_argument(
        "--pooling-method",
        type=str,
        default="last_token",
        help="How to pool embeddings. mean, last_token, or all",
    )

    args = parser.parse_args()

    ROOT_PATH = os.path.dirname(__file__)
    CONFIG = read_yaml(os.path.join(ROOT_PATH, args.config))
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    OUTPUT_PATH = os.path.join(
        DATA_PATH,
        generate_label_embeddeing_path(
            params=CONFIG["params"],
            base_label_embedding_path=CONFIG["paths"]["data_paths"][
                "BASE_LABEL_EMBEDDING_PATH"
            ],
        ),
    )

    INDEX_OUTPUT_PATH = OUTPUT_PATH.split('.')
    INDEX_OUTPUT_PATH = '_'.join([INDEX_OUTPUT_PATH[0] ,'index']) + '.'+ INDEX_OUTPUT_PATH[1]

    GO_ANNOTATIONS_PATH = os.path.join(
        DATA_PATH, CONFIG["paths"]["data_paths"]["GO_ANNOTATIONS_PATH"]
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    go_descriptions = read_pickle(GO_ANNOTATIONS_PATH)

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["params"]["LABEL_ENCODER_CHECKPOINT"],
    )
    # Initialize label encoder
    label_encoder = AutoModel.from_pretrained(
        CONFIG["params"]["LABEL_ENCODER_CHECKPOINT"],
    ).to(DEVICE)

    logging.info("Flattening descriptions for batch processing...")

    embeddings_idx = {'id':[],'description_type':[],'description':[]}
    for go_term, desriptions in tqdm(
        go_descriptions[['name','label','synonym_exact']].iterrows(), total=len(go_descriptions)
    ):
        for desription_type, desription_set in desriptions.items():
            for description in ensure_list(desription_set):
                embeddings_idx['description'].append(remove_obsolete_from_string(description))
                embeddings_idx['id'].append(go_term)
                embeddings_idx['description_type'].append(desription_type)

    # Remove Obsolete/Deprecated texts
    logging.info("Extracting embeddings...")
    embeddings = generate_label_embeddings_from_text(
        label_annotations=embeddings_idx['description'],
        label_tokenizer=label_tokenizer,
        label_encoder=label_encoder,
        pooling_method=args.pooling_method,
        batch_size_limit=CONFIG["params"]["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
        append_in_cpu=False,
    ).to('cpu')
    
    #Removing text from embeddings file. Can include later.
    embeddings_idx.pop('description')

    #Convert to indexed pandas df
    embeddings_idx = pd.DataFrame(embeddings_idx)
    

    '''
    #Map embeddings to dict
    embeddings = {}
    for (go_term, desription_type), embedding in zip(
        flattend_go_terms_and_types, all_embeddings
    ):  
        if go_term not in embeddings:
            embeddings[go_term] = {}
        if desription_type not in embeddings[go_term]:
            embeddings[go_term][desription_type] = []

        embeddings[go_term][desription_type].append(embedding.cpu())
    
    
    logging.info("Wrapping up...")
    #Cast lists to np array and add description types
    for idx,(k,v) in enumerate(embeddings.items()):
        for k2,v2 in v.items():
            embeddings[k][k2] = np.array([i.tolist() for i in v2])
        embeddings[k]['description_types']=list(v.keys())
    '''
    logging.info("Saving to a torch .pt...")
    torch.save(embeddings, OUTPUT_PATH)
    torch.save(embeddings_idx,INDEX_OUTPUT_PATH)
    logging.info(f"Embeddings saved to {OUTPUT_PATH}")

if __name__=='__main__':
    main()
