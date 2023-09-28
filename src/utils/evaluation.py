import torch
import pandas as pd
import os
from torchmetrics.classification import (
    Precision,
    Recall,
    BinaryPrecision,
    BinaryRecall,
    F1Score,
    AveragePrecision,
)
from torchmetrics import MetricCollection, Metric
from typing import Literal, Tuple


class SamplewisePrecision(Metric):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.precision_samplewise = BinaryPrecision(
            threshold=threshold, multidim_average="samplewise"
        )

    def update(self, probas: torch.Tensor, labels: torch.Tensor):
        at_least_one_positive_pred = (probas > self.threshold).any(axis=1)
        # This method could be more
        if at_least_one_positive_pred.any().item():
            self.precision_samplewise(
                probas[at_least_one_positive_pred, :],
                labels[at_least_one_positive_pred, :],
            )

    def compute(self) -> torch.Tensor:
        # catch error if no objects to concatenate and return 0
        try:
            return self.precision_samplewise.compute().mean()
        except ValueError:
            print("No objects to concatenate. Returning 0.0")
            return torch.tensor(0.0)


class SamplewiseRecall(Metric):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.recall_samplewise = BinaryRecall(
            threshold=self.threshold, multidim_average="samplewise"
        )

    def update(self, probas: torch.Tensor, labels: torch.Tensor):
        self.recall_samplewise(probas, labels)

    def compute(self) -> torch.Tensor:
        return self.recall_samplewise.compute().mean()


class SamplewiseCoverage(Metric):
    def __init__(self, threshold: float, device: str):
        super(SamplewiseCoverage, self).__init__()
        self.threshold = threshold
        self.add_state(
            "at_least_one_positive_pred",
            default=torch.tensor(0, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.int),
            dist_reduce_fx="sum",
        )

    def update(self, probas: torch.Tensor, _labels: torch.Tensor):
        # Count samples that have at least one positive prediction
        self.at_least_one_positive_pred += (probas >
                                            self.threshold).any(axis=1).sum()
        # Count total samples
        self.total_samples += probas.size(0)

    def compute(self) -> torch.Tensor:
        return self.at_least_one_positive_pred.float() / self.total_samples


class SamplewiseF1Score(Metric):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.precision_samplewise = SamplewisePrecision(threshold)
        self.recall_samplewise = BinaryRecall(
            threshold=threshold, multidim_average="samplewise"
        )

    def update(self, probas: torch.Tensor, labels: torch.Tensor):
        self.precision_samplewise.update(probas, labels)
        self.recall_samplewise.update(probas, labels)

    def compute(self) -> torch.Tensor:
        precision = self.precision_samplewise.compute()
        recall = self.recall_samplewise.compute().mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return f1


class EvalMetrics:
    def __init__(self, num_labels: int, threshold: float, device: str):
        self.threshold = threshold
        self.device = device
        self.num_labels = num_labels

    def get_label_centered_metrics(self):
        label_centered_metrics = {}
        for average in ["micro", "macro", "weighted"]:
            label_centered_metrics[f"precision_{average}"] = Precision(
                num_labels=self.num_labels,
                threshold=self.threshold,
                task="multilabel",
                average=average,
            ).to(self.device)

            label_centered_metrics[f"recall_{average}"] = Recall(
                num_labels=self.num_labels,
                threshold=self.threshold,
                task="multilabel",
                average=average,
            ).to(self.device)

            label_centered_metrics[f"f1_{average}"] = F1Score(
                num_labels=self.num_labels,
                threshold=self.threshold,
                task="multilabel",
                average=average,
            ).to(self.device)

        label_centered_metrics["map_micro"] = AveragePrecision(
            num_labels=self.num_labels, task="multilabel", thresholds=100,average='micro'
        ).to(self.device)

        return label_centered_metrics

    def get_sample_centered_metrics(self):
        sample_centered_metrics = {}
        sample_centered_metrics["precision_samplewise"] = SamplewisePrecision(
            threshold=self.threshold
        ).to(self.device)
        sample_centered_metrics["f1_samplewise"] = SamplewiseF1Score(
            threshold=self.threshold
        ).to(self.device)
        sample_centered_metrics["recall_samplewise"] = SamplewiseRecall(
            threshold=self.threshold
        ).to(self.device)
        sample_centered_metrics["coverage_samplewise"] = SamplewiseCoverage(
            threshold=self.threshold, device=self.device
        ).to(self.device)
        return sample_centered_metrics

    def get_metric_collection(
        self, type: Literal["labeled_centered", "sample_centered", "all"]
    ):
        if type == "labeled_centered":
            metrics = self.get_label_centered_metrics()
        elif type == "sample_centered":
            metrics = self.get_sample_centered_metrics()
        elif type == "all":
            metrics = {
                **self.get_label_centered_metrics(),
                **self.get_sample_centered_metrics(),
            }
        else:
            raise ValueError(f"Unknown type {type}")
        return MetricCollection(metrics)
    
    def get_metric_collection_with_regex(
            self,pattern:str
    ):  
        import re
        metrics = self.get_metric_collection(type="all")
        metrics = {k: v for k, v in metrics.items() if re.match(pattern, k)}
        return MetricCollection(metrics)




def save_evaluation_results(results, label_vocabulary, run_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_df = pd.DataFrame(results['labels'],
                            columns=label_vocabulary,
                            index=results['sequence_ids'])

    probabilities_df = pd.DataFrame(results['probabilities'],
                                    columns=label_vocabulary,
                                    index=results['sequence_ids'])

    label_df.to_parquet(os.path.join(output_dir, f'labels_{run_name}.parquet'))
    probabilities_df.to_parquet(os.path.join(
        output_dir, f'probabilities_{run_name}.parquet'))
