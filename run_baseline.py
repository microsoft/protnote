import os
from pathlib import Path
import torch
import os
from pprint import pprint
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
import pandas as pd
from src.utils.evaluation import EvalMetrics
from src.utils.data import generate_vocabularies



annotation_type = 'GO'

if annotation_type =='GO':
    label_embeddings = '2024_E5_multiling_inst_frozen_label_embeddings_mean' 
elif annotation_type=='EC':
    label_embeddings = 'ecv1_E5_multiling_inst_frozen_label_embeddings_mean'

zero_shot_pinf_logits = pd.read_parquet(f'outputs/results/test_logits_{annotation_type}_unseen_proteinfer_ABCD.parquet')
zero_shot_labels = pd.read_parquet(f'outputs/results/test_1_labels_{annotation_type+"_" if annotation_type == "EC" else ""}normal_test_label_aug_v4.parquet')
embeddings = torch.load(f'data/embeddings/{label_embeddings}.pt')
embeddings_idx = torch.load(f'data/embeddings/{label_embeddings}_index.pt')
vocabularies = generate_vocabularies(file_path = f'data/swissprot/proteinfer_splits/random/full_GO.fasta')
zero_shot_pinf_logits.columns = vocabularies['label_vocab']


#Filter embeddings
embedding_mask = embeddings_idx['description_type']=='name'
embeddings_idx = embeddings_idx[embedding_mask].reset_index(drop=True)
embeddings = embeddings[embedding_mask]

train_embeddings_mask = embeddings_idx['id'].isin(vocabularies['label_vocab'])
train_embeddings_idx = embeddings_idx[train_embeddings_mask].reset_index(drop=True)
train_embeddings = embeddings[train_embeddings_mask]

zero_shot_embeddings_mask = embeddings_idx['id'].isin(zero_shot_labels.columns)
zero_shot_embeddings_idx = embeddings_idx[zero_shot_embeddings_mask].reset_index(drop=True)
zero_shot_embeddings = embeddings[zero_shot_embeddings_mask]


#Create description mapping from seen to new
label_train_2_zero_shot_similarities = (torch.nn.functional.normalize(zero_shot_embeddings)@torch.nn.functional.normalize(train_embeddings).T)
zero_shot_label_mapping = {zero_shot_embeddings_idx['id'].iloc[zero_shot_label_idx]:train_embeddings_idx['id'].iloc[train_label_idx.item()] for zero_shot_label_idx,train_label_idx in enumerate(label_train_2_zero_shot_similarities.max(dim=-1).indices)}

zero_shot_pinf_baseline_logits = zero_shot_pinf_logits[[zero_shot_label_mapping[i] for i in zero_shot_labels.columns]]
zero_shot_pinf_baseline_logits.columns = zero_shot_labels.columns


zero_shot_pinf_baseline_logits = zero_shot_pinf_baseline_logits.loc[zero_shot_pinf_baseline_logits.index.isin(mask),cols]
zero_shot_labels = zero_shot_labels.loc[zero_shot_labels.index.isin(zero_shot_pinf_baseline_logits.index),zero_shot_pinf_baseline_logits.columns]

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


