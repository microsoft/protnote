import torch
import numpy as np
import torch.nn.functional as F

# NOT CURRENTLY USING THIS
class SupCon(torch.nn.Module):
    def __init__(self,temperature:float,base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        assert temperature is not None,"temperature must be provided and not None"

    def forward(self,input,target):
        """
        Computes the symmetric contrastive loss between protein and label embeddings, given a temperature parameter and a ground truth annotation matrix.
        P_e: Batch of projected protein embeddings [batch_size, joint_embedding_dim]
        L_e: Batch of projected GO label embeddings [batch_size, joint_embedding_dim]
        t: Temperature parameter
        target: Batch of "target" GO annotations [batch_size, num_labels]
        """
        # Compute loss for each direction (protein to label and label to protein)
        overall_loss_p = one_way_supcon(logits=input,
                                        labels_multihot=target,
                                        dim=1)
        '''
        overall_loss_l = one_way_supcon(logits=input,
                                        labels_multihot=target,
                                        dim=0)'''

        # Return the average of the two losses
        #
        # return (overall_loss_p + overall_loss_l) / 2
        return overall_loss_p

def one_way_supcon(logits, labels_multihot, dim):
    '''
    dim=1 is traditional entropy of predicting labels
    '''
    # for numerical stability
    logits_max, _ = torch.max(logits, dim=dim, keepdim=True)
    logits = logits - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) 
    log_prob = logits - torch.log(exp_logits.sum(dim, keepdim=True))

    # compute mean of log-likelihood over positive
    norm = labels_multihot.sum(dim)
    mean_log_prob_pos = (labels_multihot * log_prob).sum(dim) / norm
    mean_log_prob_pos = torch.nan_to_num(mean_log_prob_pos,0) #In case dim = 0 and there are labels always negative
    # loss
    loss = - mean_log_prob_pos.mean()
    return loss


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
    def __init__(self,temperature:float):
        super().__init__()
        self.temperature = temperature
        assert temperature is not None,"temperature must be provided and not None"

    def forward(self,input,target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target,reduce='none')
        return (loss*torch.exp(torch.clamp(loss.detach(),max=self.temperature)/(self.temperature+1))).mean()

class CBLoss(torch.nn.Module):
    def __init__(self, label_weights: torch.Tensor, beta=0.9999):
        super().__init__()

        self.label_weights = label_weights
        self.beta=beta
        assert label_weights is not None,"label_weights must be provided and not None"

    def forward(self, input,target):
        no_of_classes = len(self.label_weights)
        effective_num = 1.0 - torch.pow(self.beta, self.label_weights)

        # Replace zeros in effective_num with 'inf' (infinity) to avoid division by zero
        effective_num = torch.where(effective_num == 0, torch.tensor(float('inf')), effective_num)

        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * no_of_classes

        weights = get_batch_weights_v2(weights,target)
        cb_loss = F.binary_cross_entropy_with_logits(input = input,target = target, weight = weights)

        return cb_loss


class WeightedBCE(torch.nn.Module):
    def __init__(self, label_weights: torch.Tensor):
        super().__init__()
        assert label_weights is not None,"label_weights must be provided and not None"
        self.label_weights = label_weights

    def forward(self, input, target):
        batch_weights = get_batch_weights_v2(label_weights=self.label_weights, target=target)
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=batch_weights)
    

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

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha:float, gamma:float, reduction='mean',label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        assert (alpha is not None)&(gamma is not None),"Both gamma and alpha must be provided and neither should be None"
        print("Focal Loss with alpha: {}, gamma: {}, reduction: {}, label_smoothing: {}".format(alpha, gamma, reduction, label_smoothing))

    def forward(self, input, target):
        # Apply label smoothing, if applicable
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

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
        

def get_batch_weights_v2(label_weights, target):
    """
    Computes the weights for each sample in the batch based on the target labels
    using broadcasting.
    
    Args:
        label_weights: torch.tensor of size [no_of_classes] with the weight of each label.
        target: torch.tensor of size [batch, no_of_classes].

    Returns:
        weights_for_samples: torch.tensor of size [batch, no_of_classes].
    """

    # Ensure label_weights is a float tensor for correct broadcasting and computation
    label_weights = label_weights.float()

    # Multiply weights with target labels using broadcasting
    # This step applies the specific class weights to the corresponding labels in the target.
    weighted_targets = label_weights * target

    # Sum the weighted targets along the class dimension to get a single weight per sample
    weights_for_samples = weighted_targets.sum(dim=1, keepdim=True)

    # Use broadcasting again for expanding weights across the class dimension
    # No need to repeat the tensor explicitly.
    weights_for_samples = weights_for_samples.expand_as(target)

    return weights_for_samples



def get_batch_weights_v1(label_weights, target=None):
    """
    This function applies the given Sample Weighting Scheme and returns the sample weights normalized over a batch
    Args:
        label_weights: A tensor of size [no_of_classes] with the weight of each label
        target: torch.tensor of size [batch, no_of_classes]
        beta: float,

    Returns:
        weights_for_samples: torch.tensor of size [batch, no_of_classes]
    """

    no_of_classes = len(label_weights)

    weights_for_samples = label_weights.unsqueeze(0)
    weights_for_samples = torch.tensor(np.array(weights_for_samples.repeat(target.shape[0], 1) * target))
    weights_for_samples = weights_for_samples.sum(1)
    weights_for_samples = weights_for_samples.unsqueeze(1)
    weights_for_samples = weights_for_samples.repeat(1, no_of_classes)
    return weights_for_samples


def get_loss(config:dict,
             label_weights: torch.Tensor = None,
             bce_pos_weight: torch.Tensor = None
             ):

    if config["params"]["LOSS_FN"] == "BCE":
        return torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_pos_weight)
    elif (config["params"]["LOSS_FN"] == "WeightedBCE"):
        return WeightedBCE(label_weights = label_weights)
    elif (config["params"]["LOSS_FN"] == "CBLoss"):
        return CBLoss(label_weights = label_weights)
    elif config["params"]["LOSS_FN"] == "BatchWeightedBCE":
        return BatchWeightedBCE()
    elif config["params"]["LOSS_FN"] == "FocalLoss":
        return FocalLoss(gamma=config["params"]["FOCAL_LOSS_GAMMA"], alpha=config["params"]["FOCAL_LOSS_ALPHA"], label_smoothing=config["params"]["LABEL_SMOOTHING"])
    elif config["params"]["LOSS_FN"] == "RGDBCE":
        return RGDBCE(temperature=config["params"]["RGDBCE_TEMP"])
    elif config["params"]["LOSS_FN"] == "SupCon":
        return SupCon(temperature=config["params"]["SUPCON_TEMP"])
    else:
        raise ValueError(
            f"Unknown loss function {config['params']['LOSS_FN']}")


  
    
