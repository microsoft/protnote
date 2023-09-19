from transformers import AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast
import logging
from src.utils.data import chunks
from contextlib import nullcontext


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


def get_embeddings(text, tokenizer, model, batch_size=300, max_length=1024):
    """
    Get embeddings for a list of text strings.

    Args:
        text (list): The list of text strings. If the list of text strings is too large, it will be split into batches.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model (transformers.PreTrainedModel): The model.

    Returns:
        torch.Tensor: The embeddings.
    """
    all_label_embeddings = []

    # Tokenize the labels in batches
    for batch in chunks(text, batch_size):
        # TODO: Consider identifying the max_length in the dataset to improve speed
        tokenized_labels = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(model.device)

        # Get the label embeddings (average across all tokens of the last hidden state)
        with autocast():
            outputs = model(**tokenized_labels)
            label_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_label_embeddings.append(label_embeddings)

    # Concatenate all the label embeddings
    return torch.cat(all_label_embeddings, dim=0)
