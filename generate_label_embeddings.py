
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
from src.utils.configs import generate_label_embedding_path
from src.utils.data import read_yaml, read_pickle,remove_obsolete_from_string,ensure_list


logging.basicConfig(level=logging.INFO)

### SETUP ###
torch.cuda.empty_cache()


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
        default="mean",
        help="How to pool embeddings. mean, last_token, or all",
    )

    parser.add_argument(
        "--base-label-embedding-path",
        type=str,
        default="GO_BASE_LABEL_EMBEDDING_PATH",
        help="the base label embedding path name from config. value must exist in config.",
    )
    parser.add_argument("--annotations-path-name", type=str, default="GO_ANNOTATIONS_PATH",
                    help="Name of the annotation path. Defaults to GO.")
    
    parser.add_argument(
        "--account-for-sos",
        action='store_true',
        help="Whether to ignore the SOS token. Set to True for BioGPT and False for E5. Doesn't make any difference if pooling method = last_token"
    )

    parser.add_argument(
        "--add-instruction",
        action='store_true',
        help="Whether to format for instruction tuned model"
    )
    
    parser.add_argument(
        "--label-encoder-checkpoint",
        type=str,
        default="intfloat/multilingual-e5-large-instruct",
        help="The huggingface LLM to use. Others include microsoft/biogpt.",
    )
    

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    args = parser.parse_args()
    
    ROOT_PATH = os.path.dirname(__file__)
    CONFIG = read_yaml(os.path.join(ROOT_PATH, args.config))
    TASK = "Identify the main categories, themes, or topics described in the following Gene Ontology (GO) term, which is used to detail a protein's function"
    
    # Overwrite config pooling method
    CONFIG["params"]["LABEL_EMBEDDING_POOLING_METHOD"] = args.pooling_method
    CONFIG["params"]["LABEL_ENCODER_CHECKPOINT"] = args.label_encoder_checkpoint
    
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    OUTPUT_PATH = os.path.join(
        DATA_PATH,
        generate_label_embedding_path(
            params=CONFIG["params"],
            base_label_embedding_path=CONFIG["paths"]["data_paths"][
                args.base_label_embedding_path
            ],
        ),
    )

    print(OUTPUT_PATH)
    INDEX_OUTPUT_PATH = OUTPUT_PATH.split('.')
    INDEX_OUTPUT_PATH = '_'.join([INDEX_OUTPUT_PATH[0] ,'index']) + '.'+ INDEX_OUTPUT_PATH[1]

    ANNOTATIONS_PATH = os.path.join(
        DATA_PATH, CONFIG["paths"]["data_paths"][args.annotations_path_name]
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Pooled embeddings will be saved to {OUTPUT_PATH}\n Pooled embeddings index will be saved to {INDEX_OUTPUT_PATH} \n Using pooling method: {args.pooling_method}")

    descriptions_file = read_pickle(ANNOTATIONS_PATH)

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(
        args.label_encoder_checkpoint,
    )
    # Initialize label encoder
    label_encoder = AutoModel.from_pretrained(
        args.label_encoder_checkpoint,
    ).to(DEVICE)

    logging.info("Flattening descriptions for batch processing and calculating sequence token lengths...")
    print('SOS = ',args.account_for_sos,args.account_for_sos==False)
    embeddings_idx = {'id': [],'description_type': [],'description': [], 'token_count': []}
    for go_term, desriptions in tqdm(
        descriptions_file[['name','label','synonym_exact']].iterrows(), total=len(descriptions_file)
    ):
        for desription_type, desription_set in desriptions.items():
            for description in ensure_list(desription_set):
                description = remove_obsolete_from_string(description)

                if args.add_instruction:
                    description = get_detailed_instruct(TASK,description)

                embeddings_idx['description'].append(description)
                embeddings_idx['id'].append(go_term)
                embeddings_idx['description_type'].append(desription_type)
                embeddings_idx['token_count'].append(len(label_tokenizer.tokenize(description))) # We need the token count for embedding normalization (longer descriptions will have more feature-rich embeddings)
    # Remove Obsolete/Deprecated texts
    logging.info("Extracting embeddings...")

    embeddings = generate_label_embeddings_from_text(
        label_annotations=embeddings_idx['description'],
        label_tokenizer=label_tokenizer,
        label_encoder=label_encoder,
        pooling_method=args.pooling_method,
        batch_size_limit=CONFIG["params"]["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
        append_in_cpu=False,
        account_for_sos=args.account_for_sos
    ).to('cpu')

    #Convert to indexed pandas df
    embeddings_idx = pd.DataFrame(embeddings_idx)

    logging.info("Saving to a torch .pt...")
    torch.save(embeddings, OUTPUT_PATH)
    torch.save(embeddings_idx, INDEX_OUTPUT_PATH)
    logging.info(f"Embeddings saved to {OUTPUT_PATH}")
    logging.info(f"Index saved to {INDEX_OUTPUT_PATH}")

if __name__=='__main__':
    main()
