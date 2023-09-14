
import torch
from torchmetrics.classification import Precision, Recall, BinaryPrecision, BinaryRecall, F1Score,AveragePrecision
from torchmetrics import MetricCollection, Metric
from typing import Literal, Tuple


class SamplewisePrecision(Metric):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.precision_samplewise = BinaryPrecision(
            threshold=threshold, multidim_average='samplewise')
        self.add_state("at_least_one_positive_pred", default=torch.tensor(0, dtype=torch.int))

    def update(self, probas: torch.Tensor, labels: torch.Tensor):
        self.precision_samplewise(probas, labels)
        self.at_least_one_positive_pred += (probas > self.threshold).any(axis=1).sum()

    def compute(self) -> torch.Tensor:
        return self.precision_samplewise.compute().sum() / self.at_least_one_positive_pred
    
class SamplewiseCoverage(Metric):
    def __init__(self, threshold: float, device: str):
        super(SamplewiseCoverage, self).__init__()
        self.threshold = threshold
        self.add_state("at_least_one_positive_pred", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, probas: torch.Tensor, _labels: torch.Tensor):
        # Count samples that have at least one positive prediction
        self.at_least_one_positive_pred += (probas > self.threshold).any(axis=1).sum()
        # Count total samples
        self.total_samples += probas.size(0)

    def compute(self) -> torch.Tensor:
        return self.at_least_one_positive_pred.float() / self.total_samples

class SamplewiseF1Score(Metric):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.precision_samplewise = SamplewisePrecision(threshold)
        self.recall_samplewise = BinaryRecall(threshold=threshold, multidim_average='samplewise')
        
    def update(self, probas: torch.Tensor, labels: torch.Tensor):
        self.precision_samplewise.update(probas, labels)
        self.recall_samplewise.update(probas, labels)

    def compute(self) -> torch.Tensor:
        precision = self.precision_samplewise.compute()
        recall = self.recall_samplewise.compute().mean()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  

        return f1
    
class EvalMetrics:
    def __init__(self, num_labels: int, threshold: float,  device: str):
        self.threshold = threshold
        self.device = device
        self.num_labels = num_labels
        
    def get_label_centered_metrics(self):
        label_centered_metrics = {}
        for average in ['micro', 'macro', 'weighted']:
            label_centered_metrics[f'precision_{average}'] = Precision(
                num_labels=self.num_labels, threshold=self.threshold, task='multilabel', average=average).to(self.device)
            
            label_centered_metrics[f'recall_{average}'] = Recall(
                num_labels=self.num_labels, threshold=self.threshold,task='multilabel', average=average).to(self.device)
            
            label_centered_metrics[f'f1_{average}'] = F1Score(
                num_labels=self.num_labels, threshold=self.threshold,task='multilabel', average=average).to(self.device)
        
        label_centered_metrics['map'] = AveragePrecision(num_labels=self.num_labels,task='multilabel',thresholds =100).to(self.device)

        return label_centered_metrics
    
    def get_sample_centered_metrics(self):
        sample_centered_metrics = {}
        sample_centered_metrics['precision_samplewise'] = SamplewisePrecision(threshold=self.threshold).to(self.device)
        sample_centered_metrics['f1_samplewise'] = SamplewiseF1Score(threshold=self.threshold).to(self.device)
        sample_centered_metrics['recall_samplewise'] = BinaryRecall(threshold=self.threshold, multidim_average='samplewise').to(self.device)
        sample_centered_metrics['coverage_samplewise'] = SamplewiseCoverage(threshold=self.threshold, device=self.device).to(self.device)
        return sample_centered_metrics
    
    def get_metric_collection(self,type:Literal['labeled_centered','sample_centered','all']):
        if type == 'labeled_centered':
            metrics = self.get_label_centered_metrics()
        elif type == 'sample_centered':
            metrics = self.get_sample_centered_metrics()
        elif type == 'all':
            metrics = {**self.get_label_centered_metrics(),**self.get_sample_centered_metrics()}
        else:
            raise ValueError(f'Unknown type {type}')
        return MetricCollection(metrics)
        
class EvalMetricsOld:
    # TODO: could be better implemented with a list of metrics and a for loop
    def __init__(self, num_labels: int, threshold: float, average: Literal['micro', 'macro', 'weighted'], device: str):
        self.precision = Precision(
            num_labels=num_labels, threshold=threshold, task='multilabel', average=average).to(device)
        self.recall = Recall(num_labels=num_labels, threshold=threshold,
                             task='multilabel', average=average).to(device)
        self.f1 = F1Score(num_labels=num_labels, threshold=threshold,
                          task='multilabel', average=average).to(device)
        
        self.map = AveragePrecision(num_labels=num_labels,task='multilabel',thresholds =100).to(device)

        self.precision_samplewise = BinaryPrecision(
            threshold=threshold, multidim_average='samplewise').to(device)
        self.recall_samplewise = BinaryRecall(
            threshold=threshold, multidim_average='samplewise').to(device)

        self.at_least_one_positive_pred = torch.tensor(0, dtype=int).to(device)
        self.n = torch.tensor(0, dtype=int).to(device)
        self.threshold = threshold
        self.device = device

    def __call__(self, probas: torch.Tensor, labels: torch.Tensor):
        self.recall(probas, labels)
        self.precision(probas, labels)
        self.f1(probas, labels)

        self.recall_samplewise(probas, labels)
        self.precision_samplewise(probas, labels)

        self.at_least_one_positive_pred += (probas >
                                            self.threshold).any(axis=1).sum()
        self.map(probas, labels)
        self.n += len(labels)

    def _compute_samplewise(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        precision_samplewise = self.precision_samplewise.compute().sum() / \
            self.at_least_one_positive_pred
        recall_samplewise = self.recall_samplewise.compute().mean()
        f1_score_samplewise = 2*precision_samplewise * \
            recall_samplewise/(precision_samplewise+recall_samplewise)
        coverage_samplewise = self.at_least_one_positive_pred/self.n
        return precision_samplewise, recall_samplewise, f1_score_samplewise, coverage_samplewise

    def compute(self) -> dict:
        metrics = {'precision': self.precision.compute(),
                   'recall': self.recall.compute(),
                   'f1': self.f1.compute(),
                   'Mean Average Precision': self.map.compute()
                   }

        (metrics['precision_samplewise'],
         metrics['recall_samplewise'],
         metrics['f1_samplewise'],
         metrics['coverage_samplewise']) = self._compute_samplewise()

        return {k: v.item() for k, v in metrics.items()}

    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.precision_samplewise.reset()
        self.recall_samplewise.reset()
        self.f1.reset()
        self.map.reset()
        self.at_least_one_positive_pred = torch.tensor(
            0, dtype=int).to(self.device)
        self.n = torch.tensor(0, dtype=int).to(self.device)