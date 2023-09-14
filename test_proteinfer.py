from src.data.datasets import ProteInferDataset
from torch.utils.data import DataLoader
from src.data.collators import proteinfer_collate_variable_sequence_length
from src.models.protein_encoders import ProteInfer
from src.utils.proteinfer import transfer_tf_weights_to_torch
from src.utils.evaluation import EvalMetrics, EvalMetricsOld
from torchmetrics.classification import F1Score
from src.utils.data import read_json, load_gz_json
from src.utils.proteinfer import normalize_confidences
import torch
import numpy as np
from tqdm import tqdm
import logging

# TODO: all of these paths shouldnt be here. Could use config, hydra.
TRAIN_DATA_PATH = "/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/train_GO.fasta"
VAL_DATA_PATH = "/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/dev_GO.fasta"
TEST_DATA_PATH = "/home/samirchar/ProteinFunctions/data/swissprot/proteinfer_splits/random/test_GO.fasta"
AMINO_ACID_VOCAB_PATH = (
    "/home/samirchar/ProteinFunctions/data/vocabularies/amino_acid_vocab.json"
)
GO_LABEL_VOCAB_PATH = (
    "/home/samirchar/ProteinFunctions/data/vocabularies/proteinfer_GO_label_vocab.json"
)
MODEL_WIEGHTS_PATH = (
    "/home/samirchar/ProteinFunctions/models/proteinfer/GO_model_weights.pkl"
)
PARENTHOOD_LIB_PATH = (
    "/home/samirchar/ProteinFunctions/data/vocabularies/parenthood.json.gz"
)
PROTEINFER_RESULTS_DIR = "/home/samirchar/ProteinFunctions/data/proteinfer_results/"
NUM_LABELS = 32102
TEST_BATCH_SIZE = 2**7
DEBUG = False
DECISION_TH = 0.1  # None#0.88
METRICS_AVERAGE = "macro"

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device = {device}")


if DEBUG:
    test_dataset = ProteInferDataset(
        data_path=TRAIN_DATA_PATH,
        sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
        label_vocabulary_path=GO_LABEL_VOCAB_PATH,
    )

    test_dataset.data = [
        i for i in test_dataset.data if i[1][0] in ["P69891", "Q7AP54"]
    ]
else:
    val_dataset, test_dataset = ProteInferDataset.create_multiple_datasets(
        data_paths=[VAL_DATA_PATH, TEST_DATA_PATH],
        sequence_vocabulary_path=AMINO_ACID_VOCAB_PATH,
        label_vocabulary_path=GO_LABEL_VOCAB_PATH,
    )

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=proteinfer_collate_variable_sequence_length,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=proteinfer_collate_variable_sequence_length,
)

model = ProteInfer(
    num_labels=NUM_LABELS,
    input_channels=20,
    output_channels=1100,
    kernel_size=9,
    activation=torch.nn.ReLU,
    dilation_base=3,
    num_resnet_blocks=5,
    bottleneck_factor=0.5,
)

transfer_tf_weights_to_torch(model, MODEL_WIEGHTS_PATH)
model.to(device)
model = model.eval()

vocab = read_json(GO_LABEL_VOCAB_PATH)
label_normalizer = load_gz_json(PARENTHOOD_LIB_PATH)

if DECISION_TH is None:
    val_probas = []
    val_labels = []
    with torch.no_grad():
        for batch_idx, (sequences, sequence_lengths, labels, sequence_ids) in tqdm(
            enumerate(val_loader), total=len(val_loader)
        ):
            sequences, sequence_lengths, labels, sequence_ids = (
                sequences.to(device),
                sequence_lengths.to(device),
                labels.to(device),
                sequence_ids.to(device),
            )

            logits = model(sequences, sequence_lengths)
            probabilities = torch.sigmoid(logits)
            probabilities = torch.tensor(
                normalize_confidences(
                    predictions=probabilities.detach().cpu().numpy(),
                    label_vocab=vocab,
                    applicable_label_dict=label_normalizer,
                ),
                device=probabilities.device,
            )

            val_probas.append(probabilities)
            val_labels.append(labels)

    val_probas = torch.cat(val_probas)
    val_labels = torch.cat(val_labels)
    best_th = 0.0
    best_f1 = 0.0

    for th in np.arange(0.01, 1, 0.01):
        f1 = F1Score(
            num_labels=NUM_LABELS,
            threshold=th,
            task="multilabel",
            average=METRICS_AVERAGE,
        ).to(device)

        f1(val_probas, val_labels)
        val_f1_score = f1.compute()
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
            best_th = th
        print("TH:", th, "F1:", val_f1_score)
    print("Best Val F1:", best_f1, "Best Val TH:", best_th)
    DECISION_TH = best_th


eval_metrics = EvalMetricsOld(
    num_labels=NUM_LABELS, threshold=DECISION_TH, average=METRICS_AVERAGE, device=device
)
eval_metrics_v2 = EvalMetrics(
    num_labels=NUM_LABELS, threshold=DECISION_TH, device=device
).get_metric_collection(type="all")

all_labels = []
all_probas = []
all_seqids = []
with torch.no_grad():
    for batch_idx, (sequences, sequence_lengths, labels, sequence_ids) in tqdm(
        enumerate(test_loader), total=len(test_loader)
    ):
        sequences, sequence_lengths, labels, sequence_ids = (
            sequences.to(device),
            sequence_lengths.to(device),
            labels.to(device),
            sequence_ids.to(device),
        )

        logits = model(sequences, sequence_lengths)
        probabilities = torch.sigmoid(logits)
        probabilities = torch.tensor(
            normalize_confidences(
                predictions=probabilities.detach().cpu().numpy(),
                label_vocab=vocab,
                applicable_label_dict=label_normalizer,
            ),
            device=probabilities.device,
        )

        eval_metrics(probabilities, labels)
        eval_metrics_v2(probabilities, labels)

        all_labels.append(labels)
        all_probas.append(probabilities)
        all_seqids.append(sequence_ids)

        if DEBUG:
            print("Batch index:", batch_idx, end="\t")
            print("| Batch size:", labels.shape[0], end="\t")
            print("| Sequences shape:", sequences.shape, end="\t")
            print("| Sequences mask shape:", sequence_lengths.shape, end="\t")
            print("| Labels shape:", labels.shape, end="\t")

    final_metrics = eval_metrics.compute()
    final_metrics_v2 = eval_metrics_v2.compute()
    print("Final Metrics:", final_metrics)
    print("Final Metrics V2:", final_metrics_v2)

    all_labels = torch.cat(all_labels)
    all_probas = torch.cat(all_probas)
    all_seqids = torch.cat(all_seqids)

    # np.save(PROTEINFER_RESULTS_DIR+'labels.npy',all_labels.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'probas.npy',all_probas.detach().cpu().numpy())
    # np.save(PROTEINFER_RESULTS_DIR+'seqids.npy',all_seqids.detach().cpu().numpy())

torch.cuda.empty_cache()
