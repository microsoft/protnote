import os
import torch
import argparse
import os
import sys
import logging
from typing import Literal
from pprint import pprint
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
import pandas as pd
import subprocess
from src.utils.evaluation import EvalMetrics
from src.utils.data import generate_vocabularies, read_yaml
from test_model import TEST_COMMANDS,MODEL_PATH_TOKEN

def main(annotation_type:str, label_embedding_model:Literal['E5','BioGPT'], test_name:str, model_name:str, cache:bool):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-4s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    assert test_name in TEST_COMMANDS.keys(), f"{test_name} does not exist"

    if label_embedding_model=='E5':
        label_embeddings = '2024_E5_multiling_inst_frozen_label_embeddings_mean' 
    elif label_embedding_model=='BioGPT':
        label_embeddings = '2024_BioGPT_frozen_label_embeddings_mean' 


    proteinfer_predictions_path = f'outputs/results/test_logits_{annotation_type}_{test_name}_proteinfer.parquet'
    
    if (not os.path.exists(proteinfer_predictions_path)) | (not cache):
        logger.info('Running inference with Proteinfer...')
        subprocess.run(f"python test_proteinfer.py --test-paths-names {test_name} --only-inference --override EXTRACT_VOCABULARIES_FROM null --save-prediction-results --name {test_name}_proteinfer --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH",shell=True)
    else:
        logger.info('Found existing proteinfer predictions... caching')

    if (not os.path.exists(f'outputs/results/test_1_labels_{test_name}.parquet')) | (not cache):
        logger.info('Running zero_shot model...')
        subprocess.run(TEST_COMMANDS[test_name].replace(MODEL_PATH_TOKEN,f'models/ProTCL/{model_name}.pt'),shell=True)
    else:
        logger.info('Found existing zero shot model predictions... caching')


    if annotation_type=='EC':
        label_embeddings = 'ecv1_E5_multiling_inst_frozen_label_embeddings_mean'

    zero_shot_pinf_logits = pd.read_parquet(proteinfer_predictions_path)
    zero_shot_labels = pd.read_parquet(f'outputs/results/test_1_labels_{test_name}.parquet')
    embeddings = torch.load(f'data/embeddings/{label_embeddings}.pt')
    embeddings_idx = torch.load(f'data/embeddings/{label_embeddings}_index.pt')
    vocabularies = generate_vocabularies(file_path = f'data/swissprot/proteinfer_splits/random/full_GO.fasta')
    zero_shot_pinf_logits.columns = vocabularies['label_vocab']


    #Select embeddings based on name / short definition
    embedding_mask = embeddings_idx['description_type']=='name'
    embeddings_idx = embeddings_idx[embedding_mask].reset_index(drop=True)
    embeddings = embeddings[embedding_mask]

    #Create embeddings matrix of known proteinfer GO Term definitions
    train_embeddings_mask = embeddings_idx['id'].isin(vocabularies['label_vocab'])
    train_embeddings_idx = embeddings_idx[train_embeddings_mask].reset_index(drop=True)
    train_embeddings = embeddings[train_embeddings_mask]

    #Create embedding matrix of the new/unknown GO Term definitions
    zero_shot_embeddings_mask = embeddings_idx['id'].isin(zero_shot_labels.columns)
    zero_shot_embeddings_idx = embeddings_idx[zero_shot_embeddings_mask].reset_index(drop=True)
    zero_shot_embeddings = embeddings[zero_shot_embeddings_mask]


    #Create description mapping from seen to new
    label_train_2_zero_shot_similarities = (torch.nn.functional.normalize(zero_shot_embeddings)@torch.nn.functional.normalize(train_embeddings).T)
    zero_shot_label_mapping = {zero_shot_embeddings_idx['id'].iloc[zero_shot_label_idx]:train_embeddings_idx['id'].iloc[train_label_idx.item()] for zero_shot_label_idx,train_label_idx in enumerate(label_train_2_zero_shot_similarities.max(dim=-1).indices)}

    # Create baseline predictions by replicating logits of most similar labels
    zero_shot_pinf_baseline_logits = zero_shot_pinf_logits[[zero_shot_label_mapping[i] for i in zero_shot_labels.columns]]
    zero_shot_pinf_baseline_logits.columns = zero_shot_labels.columns

    zero_shot_pinf_baseline_logits.to_parquet(f'outputs/results/test_logits_{annotation_type}_{test_name}_{label_embedding_model}_baseline.parquet')

    #Evaluate performance of baseline
    eval_metrics = EvalMetrics(device='cuda')
    mAP_micro = BinaryAUPRC(device='cpu')
    mAP_macro = MultilabelAUPRC(device='cpu',num_labels=zero_shot_labels.shape[-1])
    metrics = eval_metrics\
            .get_metric_collection_with_regex(pattern='f1_m.*',
                                threshold=0.5,
                                num_labels=zero_shot_labels.shape[-1]
                                )

    metrics(torch.sigmoid(torch.tensor(zero_shot_pinf_baseline_logits.values,device='cuda')),
                torch.tensor(zero_shot_labels.values,device='cuda'))
    mAP_micro.update(torch.sigmoid(torch.tensor(zero_shot_pinf_baseline_logits.values)).flatten(),
                                torch.tensor(zero_shot_labels.values).flatten())
    mAP_macro.update(torch.sigmoid(torch.tensor(zero_shot_pinf_baseline_logits.values)),
                    torch.tensor(zero_shot_labels.values))


    metrics = metrics.compute()
    metrics.update({
                    "map_micro":mAP_micro.compute(),
                    "map_macro":mAP_macro.compute()
                    })
    metrics = {k:v.item() for k,v in metrics.items()}
    pprint(metrics)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
    description="Run baseline inference")
    parser.add_argument("--annotation-type", type=str, default='GO')
    parser.add_argument("--test-name", type=str, required=True, help='The name of the test set to run baseline on')
    parser.add_argument("--model-name", type=str, required=True,help='The name of the latest zero shot model without the .pt')
    parser.add_argument("--label-embedding-model", type=str, required=True,help='The name of LLM used for text embeddings: E5 or BioGPT')
    parser.add_argument("--cache", help='Whether to cache proteinfer predictions if available',action='store_true')
    
    args = parser.parse_args()

    main(annotation_type=args.annotation_type,
         test_name=args.test_name,
         model_name = args.model_name,
         label_embedding_model=args.label_embedding_model,
         cache = args.cache)
