import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.cuda.amp import autocast


def count_parameters_by_layer(model):
    """
    Logs the number of total and trainable parameters for each major category of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which parameters are to be counted.

    Outputs:
        Logs major categories along with their total and trainable parameters.
    """
    total_params = 0
    trainable_params = 0
    category_params = {}

    for name, param in model.named_parameters():
        category = name.split('.')[0]
        num_params = param.numel()

        if category not in category_params:
            category_params[category] = {'total': 0, 'trainable': 0}

        category_params[category]['total'] += num_params
        if param.requires_grad:
            category_params[category]['trainable'] += num_params

        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    max_name_length = max([len(category)
                          for category in category_params.keys()])

    # Formatting and logging
    line = "=" * 120
    logging.info(line)
    logging.info(
        f"{'Major Category':<{max_name_length}} {'Total Parameters':<20} {'Trainable Parameters'}")
    logging.info(line)
    for category, params in category_params.items():
        logging.info(
            f"{category:<{max_name_length}} {params['total']:<20} {params['trainable']}")

    assert trainable_params > 0, "No trainable parameters found. Check the config file to ensure that the model is not frozen."
    logging.info(line)
    logging.info(
        f"{'TOTAL':<{max_name_length}} {total_params:<20} {trainable_params}")
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
    return get_label_embeddings(tokenized_labels, label_encoder, batch_size_limit=batch_size_limit).cpu()


def sigmoid_bias_from_prob(prior_prob):
    return -np.log((1 - prior_prob) / prior_prob)
