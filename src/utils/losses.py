import torch
import torch.nn.functional as F

# NOT CURRENTLY USING THIS
# def contrastive_loss(P_e, L_e, t, target):
#     """
#     Computes the symmetric contrastive loss between protein and label embeddings, given a temperature parameter and a ground truth annotation matrix.
#     P_e: Batch of projected protein embeddings [batch_size, joint_embedding_dim]
#     L_e: Batch of projected GO label embeddings [batch_size, joint_embedding_dim]
#     t: Temperature parameter
#     target: Batch of "target" GO annotations [batch_size, num_labels]
#     """
#     # Compute pairwise cosine similarities; vectors already normalized, so no need to divide by product of vector magnitudes
#     logits = torch.mm(P_e, L_e.t()) / t

#     # Compute loss for each direction (protein to label and label to protein)
#     overall_loss_p = compute_asymmetric_loss(logits, target, dim=1)
#     overall_loss_l = compute_asymmetric_loss(logits.t(), target.t(), dim=0)

#     # Return the average of the two losses
#     return (overall_loss_p + overall_loss_l) / 2


# def compute_asymmetric_loss(logits, target, dim):
#     """
#     Helper function to compute the asymmetric loss (either using the protein or label embeddings as the anchor)
#     """
#     epsilon = 1e-10  # Small value to avoid division by zero
#     loss = F.binary_cross_entropy_with_logits(
#         logits, target.float(), reduction='none')

#     # Create positive and negative label masks
#     target_pos = target.bool()
#     target_neg = ~target_pos

#     # Apply the positive and negative masks to the loss matrix
#     loss_pos = loss.clone()
#     loss_pos[target_neg] = 0

#     loss_neg = loss.clone()
#     loss_neg[target_pos] = 0

#     # Sum the loss for each element (either protein or label, depending on the dimension)
#     loss_pos = loss_pos.sum(dim=dim)
#     loss_neg = loss_neg.sum(dim=dim)

#     # Compute the number of positive and negative labels for each element, depending on the dimension
#     num_pos = target.sum(dim=dim)
#     num_neg = target.size(dim) - num_pos

#     # Return the weighted average of the positive and negative losses
#     return ((loss_pos / (num_pos + epsilon)) + (loss_neg / (num_neg + epsilon))).mean()

class RGDBCE(torch.nn.Module):
    def __init__(self,temp):
        super().__init__()
        self.temp = temp

    def forward(self,input,target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target,reduce='none')
        return (loss*torch.exp(torch.clamp(loss.detach(),max=self.temp)/(self.temp+1))).mean()

class BatchWeightedBCE(torch.nn.Module):
    def __init__(self, epsilon=1e-10):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, input, target):
        # Count the number of positives and negatives in the batch
        num_positives = target.sum() + self.epsilon  
        num_negatives = target.numel() - num_positives + self.epsilon  

        # Calculate the weights for positives and negatives
        total = num_positives + num_negatives
        weight_positives = (1.0 / num_positives) * (total / 2.0)
        weight_negatives = (1.0 / num_negatives) * (total / 2.0)
        
        # Create a weight tensor with the same shape as target
        weight_tensor = target * weight_positives + (1 - target) * weight_negatives
        
        # Compute weighted binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight_tensor)

class BatchLabelWeightedBCE(torch.nn.Module):
    def __init__(self, epsilon=1e-10):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, input, target):
        total_labels = target.sum() + self.epsilon #epsilon in case all labels are 0 due to high imbalance
        label_frequencies = target.sum(axis=0)/total_labels

        label_frequencies = torch.where(
            label_frequencies == 0, 
            torch.ones_like(label_frequencies), 
            1 / label_frequencies
        )

        weights = label_frequencies / label_frequencies.sum()

        # Compute weighted binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weights.unsqueeze(0))

class FocalLossV2(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(input, target)
        pt = torch.exp(-BCE_loss)
        loss = ((1-pt)**self.gamma) * BCE_loss

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean', epsilon=1e-10):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        # Compute the focal loss
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
    

    