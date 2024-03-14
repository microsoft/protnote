import torch
import pandas as pd
import os
from src.utils.data import convert_float16_to_float32
from torchmetrics.classification import (
    Precision,
    Recall,
    BinaryPrecision,
    BinaryRecall,
    F1Score,
    AveragePrecision,
)
from torchmetrics import MetricCollection, Metric
from typing import Literal
import re
import numpy as np
from tqdm import tqdm


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

def metric_collection_to_dict_float(metric_collection:MetricCollection,prefix=None):
    metric_collection = metric_collection.copy()
    
    for k, v in metric_collection.items():
        if (not isinstance(v,float))&(not isinstance(v,int)):
            if isinstance(v, torch.Tensor):
                metric_collection[k] = v.item()

            # Cast numpy floats to float32. Needed to store as parquet
            # because pyarrow doesn't support float16 from mixed precision
            if np.issubdtype(type(v), np.floating):
                metric_collection[k] = v.astype("float32")
        
    
    metric_collection = {(f'{prefix}_{k}' if prefix is not None else k):v for k,v in metric_collection.items()}
        
    return metric_collection

class EvalMetrics:
    def __init__(self, device: str):
        """_summary_

        :param device: _description_
        :type device: str
        """
        self.device = device

    def _get_label_centered_metrics(self, threshold: float, num_labels: int) -> dict:
        # TODO: Change micro metrics to task='binary' or use Binary version of the metrics
        # num labels is unnecessary for micro metrics
        """_summary_

        :param threshold: _description_
        :type threshold: float
        :return: _description_
        :rtype: dict
        """
        label_centered_metrics = {}
        for average in ["micro", "macro", "weighted"]:
            if (threshold is not None) & (num_labels is not None):
                label_centered_metrics[f"precision_{average}"] = Precision(
                    num_labels=num_labels,
                    threshold=threshold,
                    task="multilabel",
                    average=average,
                ).to(self.device)

                label_centered_metrics[f"recall_{average}"] = Recall(
                    num_labels=num_labels,
                    threshold=threshold,
                    task="multilabel",
                    average=average,
                ).to(self.device)

                label_centered_metrics[f"f1_{average}"] = F1Score(
                    num_labels=num_labels,
                    threshold=threshold,
                    task="multilabel",
                    average=average,
                ).to(self.device)

            if num_labels is not None:
                label_centered_metrics[f"map_{average}"] = AveragePrecision(
                    num_labels=num_labels, task="multilabel", average=average
                ).to(self.device)

        return label_centered_metrics

    def _get_sample_centered_metrics(self, threshold) -> dict:
        """_summary_

        :param threshold: _description_
        :type threshold: _type_
        :return: _description_
        :rtype: dict
        """
        sample_centered_metrics = {}
        if threshold is not None:
            sample_centered_metrics["precision_samplewise"] = SamplewisePrecision(
                threshold=threshold
            ).to(self.device)
            sample_centered_metrics["f1_samplewise"] = SamplewiseF1Score(
                threshold=threshold
            ).to(self.device)
            sample_centered_metrics["recall_samplewise"] = SamplewiseRecall(
                threshold=threshold
            ).to(self.device)
            sample_centered_metrics["coverage_samplewise"] = SamplewiseCoverage(
                threshold=threshold, device=self.device
            ).to(self.device)
        return sample_centered_metrics

    def get_metric_collection(
        self, type: Literal["labeled_centered", "sample_centered", "all"], threshold: float, num_labels: int
    ) -> MetricCollection:
        """_summary_

        :param type: _description_
        :type type: Literal[&quot;labeled_centered&quot;, &quot;sample_centered&quot;, &quot;all&quot;]
        :param threshold: _description_
        :type threshold: float
        :raises ValueError: _description_
        :return: _description_
        :rtype: MetricCollection
        """
        if type == "labeled_centered":
            metrics = self._get_label_centered_metrics(
                threshold=threshold, num_labels=num_labels)
        elif type == "sample_centered":
            metrics = self._get_sample_centered_metrics(threshold=threshold)
        elif type == "all":
            metrics = {
                **self._get_label_centered_metrics(threshold=threshold, num_labels=num_labels),
                **self._get_sample_centered_metrics(threshold=threshold),
            }
        else:
            raise ValueError(f"Unknown type {type}")
        return MetricCollection(metrics)

    def get_metric_collection_with_regex(
            self, pattern: str, num_labels: int, threshold: float = None
    ) -> MetricCollection:
        """_summary_

        :param pattern: _description_
        :type pattern: str
        :param threshold: _description_
        :type threshold: float
        :return: _description_
        :rtype: MetricCollection
        """
        metrics = self.get_metric_collection(
            type="all", threshold=threshold, num_labels=num_labels)
        metrics = {k: v for k, v in metrics.items() if re.match(pattern, k)}
        return MetricCollection(metrics)

    def get_metric_by_name(self, name: str, num_labels: int, threshold: float = None) -> Metric:
        """_summary_

        :param name: _description_
        :type name: str
        :param threshold: _description_, defaults to None
        :type threshold: float, optional
        :return: a single metric
        :rtype: Metric
        """
        metrics = self.get_metric_collection(
            type="all", num_labels=num_labels, threshold=threshold)
        assert name in metrics, f"Unknown metric {name}. Available metrics are {metrics.keys()}"
        return metrics[name]


def save_evaluation_results(results, label_vocabulary, run_name,output_dir,data_split_name):
    # Do not need to check if is_master, since this function is only called by the master node
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if len(label_vocabulary)!=results['logits'].shape[-1]:
        logits_df_cols = list(range(results['logits'].shape[-1]))
    else:
        logits_df_cols = label_vocabulary

    logits_df = pd.DataFrame(results['logits'],
                                    columns=logits_df_cols,
                                    index=results['sequence_ids']) 
    logits_df = convert_float16_to_float32(logits_df)
    logits_df_output_path = os.path.join(
        output_dir, f'{data_split_name}_logits_{run_name}.parquet')
    print(f"saving results to {logits_df_output_path}")
    logits_df.to_parquet(logits_df_output_path)

    
    label_df = pd.DataFrame(results['labels'],
                    columns=label_vocabulary,
                    index=results['sequence_ids'])
    label_df = convert_float16_to_float32(label_df)
    label_df_output_path = os.path.join(
        output_dir, f'{data_split_name}_labels_{run_name}.parquet')
    
    print(f"saving results to {label_df_output_path}")
    label_df.to_parquet(label_df_output_path)
    

        


def metrics_per_label_df(label_df:pd.DataFrame,pred_df:pd.DataFrame,device:str,threshold:None)->pd.DataFrame:
    """Calculate the following per label metrics: Average Precision,F1,Precision,Recall,label frequency

    :param label_df: dataframe where each column is a label and every row an observation. element row=i,col=j is 1
        if observation i has label j, 0 otherwise
    :type label_df: pd.DataFrame
    :param pred_df: dataframe where each column is a label and every row an observation. element row=i,col=j is the prediction probability
        that observation i has label j.
    :type pred_df: pd.DataFrame
    :param device: Device to make calculations (e.g., cuda)
    :type device: str
    :param threshold: Threshold to calculate threshold-dependent metrics like F1 Score
    :type threshold: None
    :return: The Average Preicion for each label
    :rtype: pd.DataFrame

    """    

    rows = []
    labels = label_df.columns
    mask_at_least_one_label = label_df.sum()>0
    labels_at_least_one_pos = label_df.columns[mask_at_least_one_label]
    labels_undefined = label_df.columns[~mask_at_least_one_label]
    threshold = threshold
    for col in tqdm(labels_at_least_one_pos):
        row = {}
        preds = torch.tensor(pred_df[col].values,device=device)
        ground_truth = torch.tensor(label_df[col].values,device=device)
        
        row['AUPRC'] = AveragePrecision(task="binary").to(device)(preds,ground_truth).item()
        row['Frequency'] = ground_truth.sum().item()
        row['Relative Frequency'] = row['Frequency']/len(ground_truth)

        if threshold is not None:
            row['Precision'] = Precision(task="binary",threshold=threshold).to(device)(preds,ground_truth).item()
            row['Recall'] = Precision(task="binary",threshold=threshold).to(device)(preds,ground_truth).item()
            row['F1'] = 2*row['Precision']*row['Recall']/(row['Precision']+row['Recall']) if (row['Precision']+row['Recall']) >0 else 0
        
        rows.append(row)

    metrics = pd.DataFrame(rows,index=labels_at_least_one_pos)
    metrics_undefined = pd.DataFrame(np.nan,columns=metrics.columns,index=labels_undefined)
    return pd.concat([metrics,metrics_undefined])
