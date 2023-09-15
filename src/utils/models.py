from transformers import AutoTokenizer, AutoModel
import torch
from torch.cuda.amp import autocast  # For AMP
import logging
from torch.utils.data import DataLoader


def count_parameters_by_layer(model):
    """
    Logs the number of total and trainable parameters for each layer of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which parameters are to be counted.

    Outputs:
        Logs layer names along with their total and trainable parameters.
    """
    total_params = 0
    trainable_params = 0

    logging.info("Layer-wise parameter details:")
    logging.info(
        f"{'Layer Name':<{max([len(name) for name, _ in model.named_parameters()])}} {'Total Parameters':<20} {'Trainable Parameters'}")

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

    assert trainable_params > 0, "No trainable parameters found. Check the config file to ensure that the model is not frozen."

    logging.info(
        f"{name:<{max([len(name) for name, _ in model.named_parameters()])}} {num_params:<20} {num_params if param.requires_grad else 0}")

    logging.info(
        f"{'TOTAL':<{max([len(name) for name, _ in model.named_parameters()])}} {total_params:<20} {trainable_params}")


def load_model_and_tokenizer(checkpoint, freeze_weights=False):
    """
    Load a tokenizer and model given a checkpoint string.

    Args:
        checkpoint (str): The checkpoint string for the model and tokenizer, based on the HuggingFace model hub.
        freeze_weights (bool, optional): If True, the model weights will be frozen. Defaults to False.

    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to('cuda')

    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    return tokenizer, model


def tokenize_inputs(tokenizer, labels, padding=True, truncation=True, max_length=512):
    """
    Tokenize inputs given a tokenizer.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        labels (list): The list of labels to tokenize.
        max_length (int, optional): Maximum length for tokenization. Defaults to 512.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: The tokenized inputs.
    """
    return tokenizer(labels, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt").to('cuda')


def get_embeddings_from_tokens(model, input_data, train_model=False):
    """
    Get embeddings given tokens using a model.

    Args:
        model (transformers.PreTrainedModel): The model.
        input_data (Union[DataLoader, transformers.tokenization_utils_base.BatchEncoding]): DataLoader or Tokenized inputs.
        train_model (bool, optional): If True, the model will be trained on the input tokens. Defaults to False.

    Returns:
        torch.Tensor: The embeddings tensor.
    """
    # Set the model to training mode if train_model is True
    model.train(train_model)

    if isinstance(input_data, DataLoader):
        #TODO: 768 is hard-coded here; this should be changed to a variable
        embeddings = torch.empty((len(input_data.dataset), 768), device='cuda')
        for i, batch in enumerate(input_data):
            batch = tuple(t.to('cuda') for t in batch)
            with torch.set_grad_enabled(train_model), autocast():
                outputs = model.forward(
                    input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True)
                batch_embeddings = outputs.last_hidden_state
                # Extract the [CLS] token embeddings (the first token for each sequence)
                cls_embeddings = batch_embeddings[:, 0, :]
                embeddings[i*input_data.batch_size:(i+1)
                           * input_data.batch_size] = cls_embeddings
        return embeddings

    else:  # Direct tokenized inputs
        with torch.set_grad_enabled(train_model), autocast():
            outputs = model.forward(
                input_ids=input_data['input_ids'].to('cuda'), attention_mask=input_data['attention_mask'].to('cuda'), output_hidden_states=True)
            embeddings = outputs.last_hidden_state
            # Extract the [CLS] token embeddings (the first token for each sequence)
            cls_embeddings = embeddings[:, 0, :]
        return cls_embeddings
