
import torch
from torchmetrics.classification import Precision,Recall,BinaryPrecision,BinaryRecall,F1Score
from typing import Literal,Tuple
class EvalMetrics:
    def __init__(self,num_labels:int,threshold:float,average:Literal['micro','macro','weighted'],device:str):
        self.precision = Precision(num_labels = num_labels, threshold = threshold,task = 'multilabel',average=average).to(device)
        self.recall = Recall(num_labels = num_labels, threshold = threshold,task = 'multilabel',average=average).to(device)
        self.f1 = F1Score(num_labels = num_labels, threshold = threshold,task = 'multilabel',average=average).to(device)

        self.precision_samplewise = BinaryPrecision(threshold = threshold,multidim_average='samplewise').to(device)
        self.recall_samplewise = BinaryRecall(threshold = threshold,multidim_average='samplewise').to(device)

        self.at_least_one_positive_pred = torch.tensor(0,dtype=int).to(device)
        self.n = torch.tensor(0,dtype=int).to(device)
        self.threshold = threshold
        self.device = device

    def __call__(self,probas:torch.Tensor,labels:torch.Tensor):
        self.recall(probas,labels)
        self.precision(probas,labels)
        self.f1(probas,labels)
        
        self.recall_samplewise(probas,labels)
        self.precision_samplewise(probas,labels)

        self.at_least_one_positive_pred+=(probas>self.threshold).any(axis=1).sum()
        self.n+=len(labels)
    
    def _compute_samplewise(self)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        precision_samplewise = self.precision_samplewise.compute().sum()/self.at_least_one_positive_pred
        recall_samplewise = self.recall_samplewise.compute().mean()
        f1_score_samplewise = 2*precision_samplewise*recall_samplewise/(precision_samplewise+recall_samplewise)
        coverage_samplewise = self.at_least_one_positive_pred/self.n
        return precision_samplewise,recall_samplewise,f1_score_samplewise,coverage_samplewise

    def compute(self)->dict:
        metrics = {'precision':self.precision.compute(),
                    'recall':self.recall.compute()}
        
        (metrics['precision_samplewise'],
         metrics['recall_samplewise'],
         metrics['f1_samplewise'],
         metrics['coverage_samplewise']) = self._compute_samplewise()

        return {k:v.item() for k,v in metrics.items()}
    
    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.precision_samplewise.reset()
        self.recall_samplewise.reset()
        self.at_least_one_positive_pred = torch.tensor(0,dtype=int).to(self.device)
        self.n = torch.tensor(0,dtype=int).to(self.device)