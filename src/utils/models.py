import torch
import logging
import numpy as np

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


def get_label_embeddings(tokenized_labels, model, batch_size_limit=300):
    """
    Get embeddings for a list of tokenized labels.

    Args:
        tokenized_labels (BatchEncoding): Tokenized labels.
        model (transformers.PreTrainedModel): The model.
        batch_size_limit (int): The maximum number of labels to process in a single batch.

    Returns:
        torch.Tensor: The embeddings.
    """
    all_label_embeddings = []

    # Move the entire tokenized data to GPU
    tokenized_labels = {key: value.to(model.device)
                        for key, value in tokenized_labels.items()}

    total_labels = len(tokenized_labels["input_ids"])

    for start_idx in range(0, total_labels, batch_size_limit):
        end_idx = min(start_idx + batch_size_limit, total_labels)
        batch = {key: value[start_idx:end_idx]
                 for key, value in tokenized_labels.items()}

        # Get the label embeddings (average across all tokens of the last hidden state)
        outputs = model(**batch)
        label_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_label_embeddings.append(label_embeddings)

    # Concatenate all the label embeddings
    return torch.cat(all_label_embeddings, dim=0)


def generate_label_embeddings_from_text(label_annotations, label_tokenizer, label_encoder, label_batch_size_limit):
    """Tokenize the labels and generate label embeddings."""
    tokenized_labels = tokenize_labels(label_annotations, label_tokenizer)
    with torch.no_grad():
        return get_label_embeddings(tokenized_labels, label_encoder, batch_size_limit=label_batch_size_limit).cpu()

def sigmoid_bias_from_prob(prior_prob):
    return -np.log((1 - prior_prob) / prior_prob)