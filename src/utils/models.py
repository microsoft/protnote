import torch
import logging
import re
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
from collections import OrderedDict

def biogpt_train_last_n_layers(model,n):
    for param in model.parameters():
        param.requires_grad = False

    if n>0:
        max_layer_num = len(model.layers)-1
        for param_name,param in model.named_parameters():
            print(param_name)
            layer_num = re.search(r'layers\.(\d+)', param_name)

            if layer_num:
                number = int(layer_num.group(1))
                if number>max_layer_num-n:
                    param.requires_grad = True

        for param in model.layer_norm.parameters():
            param.requires_grad = True


def count_parameters_by_layer(model):
    """
    Logs the number of total and trainable parameters for each major category of a PyTorch model,
    and prints the names of the trainable layers.

    Args:
        model (torch.nn.Module): The PyTorch model for which parameters are to be counted.

    Outputs:
        Logs major categories along with their total and trainable parameters, and names of trainable layers.
    """
    total_params = 0
    trainable_params = 0
    category_params = {}
    trainable_layers = []

    for name, param in model.named_parameters():
        category = name.split('.')[0]
        num_params = param.numel()

        if category not in category_params:
            category_params[category] = {'total': 0, 'trainable': 0}

        category_params[category]['total'] += num_params
        if param.requires_grad:
            category_params[category]['trainable'] += num_params
            if name not in trainable_layers:
                trainable_layers.append(name)

        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    max_name_length = max([len(category) for category in category_params.keys()])

    # Formatting and logging
    line = "=" * 120
    logging.info(line)
    logging.info(f"{'Major Category':<{max_name_length}} {'Total Parameters':<20} {'Trainable Parameters'}")
    logging.info(line)
    for category, params in category_params.items():
        logging.info(f"{category:<{max_name_length}} {params['total']:<20} {params['trainable']}")

    assert trainable_params > 0, "No trainable parameters found. Check the config file to ensure that the model is not frozen."
    logging.info(line)
    logging.info(f"{'TOTAL':<{max_name_length}} {total_params:<20} {trainable_params}")
    logging.info(line)

    # Log names of trainable layers
    logging.info("Trainable Layers:")
    for layer in trainable_layers:
        logging.info(f"- {layer}")

    logging.info(line)




def tokenize_labels(text, tokenizer, max_length=1024):
    """
    Tokenize a list of text strings.

    Args:
        text (list): The list of text strings.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.

    Returns:
        dict: A dictionary containing tokenized labels as 'input_ids' and 'attention_mask'.
    """
    return tokenizer(
        text, padding='longest', truncation=True, max_length=max_length, return_tensors="pt"
    )


def compute_mean_hidden_states(last_hidden_states, attention_mask):
    """Compute the mean of the last hidden state for only the relevant tokens."""
    # Compute the number of relevant tokens for each sequence
    num_relevant_tokens = attention_mask.sum(dim=1, keepdim=True)
    # Mask the last_hidden_state tensor and compute the sum
    sum_hidden_states = (last_hidden_states *
                         attention_mask.unsqueeze(-1)).sum(dim=1)
    # Compute the mean of the last hidden state
    return sum_hidden_states / num_relevant_tokens


def get_label_embeddings(tokenized_labels, model, batch_size_limit=1000):
    """
    Get embeddings for a list of tokenized labels.
    Assumes that tokenized_labels and model are on the same device, ideally GPU.
    """
    total_labels = tokenized_labels["input_ids"].shape[0]

    if total_labels <= batch_size_limit:
        with autocast():
            last_hidden_states = model(
                input_ids=tokenized_labels["input_ids"],
                attention_mask=tokenized_labels["attention_mask"]
            ).last_hidden_state
        output = compute_mean_hidden_states(
            last_hidden_states, tokenized_labels["attention_mask"])
        del last_hidden_states
        return output

    else:
        # Convert dictionary values to tensors
        tensors = [tokenized_labels["input_ids"],
                   tokenized_labels["attention_mask"]]
        # Create TensorDataset and DataLoader
        dataset = TensorDataset(*tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size_limit,
                                shuffle=False, pin_memory=False, num_workers=0)

        all_label_embeddings = []
        for batch in dataloader:
            input_ids, attention_mask = batch
            with autocast():
                last_hidden_states = model(
                    input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mean_hidden_states = compute_mean_hidden_states(
                last_hidden_states, attention_mask)
            all_label_embeddings.append(mean_hidden_states)
            del last_hidden_states, mean_hidden_states
        # Concatenate all the label embeddings
        return torch.cat(all_label_embeddings, dim=0)


def generate_label_embeddings_from_text(label_annotations, label_tokenizer, label_encoder, batch_size_limit=1000):
    """Tokenize the labels and generate label embeddings."""
    tokenized_labels = tokenize_labels(label_annotations, label_tokenizer)

    # Move to GPU
    tokenized_labels["input_ids"] = tokenized_labels["input_ids"].to(
        label_encoder.device)
    tokenized_labels["attention_mask"] = tokenized_labels["attention_mask"].to(
        label_encoder.device)

    # Generate label embeddings
    return get_label_embeddings(tokenized_labels, label_encoder, batch_size_limit=batch_size_limit)


def sigmoid_bias_from_prob(prior_prob):
    return -np.log((1 - prior_prob) / prior_prob)


def save_checkpoint(model, optimizer, epoch, best_val_metric, model_path):
    """
    Save model and optimizer states as a checkpoint.

    Args:
    - model (torch.nn.Module): The model whose state we want to save.
    - optimizer (torch.optim.Optimizer): The optimizer whose state we want to save.
    - epoch (int): The current training epoch.
    - model_path (str): The path where the checkpoint will be saved.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_metric': best_val_metric,
    }

    torch.save(checkpoint, model_path)


def load_checkpoint(trainer, checkpoint_path):
    """
    Load the model's state dict, optimizer's state, and epoch number from the checkpoint.

    This function handles both DDP-wrapped and non-DDP checkpoints.

    :param model: The model into which the checkpoint's state dict should be loaded.
    :param trainer: The trainer instance containing the optimizer and epoch attributes.
    :param checkpoint_path: Path to the checkpoint file.
    """

    # Load the entire checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract the state_dict from the checkpoint
    model_state_dict = checkpoint['model_state_dict']

    # Check if the state_dict is from a DDP-wrapped model
    if list(model_state_dict.keys())[0].startswith('module.'):
        # Remove the "module." prefix
        new_model_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_model_state_dict[name] = v
        model_state_dict = new_model_state_dict

    # Load the state_dict into the model
    trainer.model.module.load_state_dict(model_state_dict)

    # Load the optimizer state and epoch number if they exist in the checkpoint
    if 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'epoch' in checkpoint:
        trainer.epoch = checkpoint['epoch']
    if 'best_val_metric' in checkpoint:
        trainer.best_val_metric = checkpoint['best_val_metric']