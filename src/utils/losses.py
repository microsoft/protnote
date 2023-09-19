import torch
import torch.nn.functional as F


def contrastive_loss(P_e, L_e, t, target):
    """
    Computes the symmetric contrastive loss between protein and label embeddings, given a temperature parameter and a ground truth annotation matrix.
    P_e: Batch of projected protein embeddings [batch_size, joint_embedding_dim]
    L_e: Batch of projected GO label embeddings [batch_size, joint_embedding_dim]
    t: Temperature parameter
    target: Batch of "target" GO annotations [batch_size, num_labels]
    """
    # Compute pairwise cosine similarities; vectors already normalized, so no need to divide by product of vector magnitudes
    logits = torch.mm(P_e, L_e.t()) / t

    # Compute loss for each direction (protein to label and label to protein)
    overall_loss_p = compute_asymmetric_loss(logits, target, dim=1)
    overall_loss_l = compute_asymmetric_loss(logits.t(), target.t(), dim=0)

    # Return the average of the two losses
    return (overall_loss_p + overall_loss_l) / 2


def compute_asymmetric_loss(logits, target, dim):
    """
    Helper function to compute the asymmetric loss (either using the protein or label embeddings as the anchor)
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')

    # Create positive and negative label masks
    target_pos = target.bool()
    target_neg = ~target_pos

    # Apply the positive and negative masks to the loss matrix
    loss_pos = loss.clone()
    loss_pos[target_neg] = 0

    loss_neg = loss.clone()
    loss_neg[target_pos] = 0

    # Sum the loss for each element (either protein or label, depending on the dimension)
    loss_pos = loss_pos.sum(dim=dim)
    loss_neg = loss_neg.sum(dim=dim)

    # Compute the number of positive and negative labels for each element, depending on the dimension
    num_pos = target.sum(dim=dim)
    num_neg = target.size(dim) - num_pos

    # Return the weighted average of the positive and negative losses
    return ((loss_pos / (num_pos + epsilon)) + (loss_neg / (num_neg + epsilon))).mean()


class MultiLabelBCE(torch.nn.Module):
    """Binary Cross Entropy for Multi Label classification

    """    
    def __init__(self, pos_weight=None, symmetric=False):
        """
        :param pos_weight: weight for positive examples, defaults to None
        :type pos_weight: torch.Tensor, optional
        :param symmetric: whether to compute the symmetric loss, defaults to False. 
        :type symmetric: bool, optional
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.symmetric = symmetric
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def forward(self, logits, target):
        """        
        :param logits: Tensor of shape (batch_size, num_labels)
        :type logits: torch.Tensor
        :param target: Tensor of shape (batch_size, num_labels)
        :type target: torch.Tensor
        """
        loss = self.bce_with_logits(logits, target.float())

        #Compute standard multilabel BCE loss. Loss per example is summed over all labels
        standard_loss = loss.sum(dim=1).mean()

        if self.symmetric:
            #Compute loss as if the targets where the examples. Loss per label is summed over all examples.
            inverted_loss = loss.sum(dim=0).mean()
            
            return (standard_loss + inverted_loss)/2
        else:
            return standard_loss

