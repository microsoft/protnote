from src.utils.losses import FocalLoss, FocalLossUnstable
import torch

size = 10000
inputs = torch.rand((size,1))
targets = torch.randint(0,2,(size,1),dtype=float)
gamma = 10
alpha = 0.01
print(FocalLoss(gamma=gamma,alpha=alpha)(inputs,targets),FocalLossUnstable(gamma=gamma,alpha=alpha)(inputs,targets))