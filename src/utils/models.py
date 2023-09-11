from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import logging


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

        logging.info(
            f"{name:<{max([len(name) for name, _ in model.named_parameters()])}} {num_params:<20} {num_params if param.requires_grad else 0}")

    logging.info(
        f"{'TOTAL':<{max([len(name) for name, _ in model.named_parameters()])}} {total_params:<20} {trainable_params}")


def load_PubMedBERT(trainable=False):
    """
    Load the PubMedBERT tokenizer and model.

    Args:
        trainable (bool, optional): If True, allows the model weights to be trainable. 
                                    Defaults to False (weights are frozen).

    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    # Load PubMedBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModelForMaskedLM.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # If not trainable, freeze the model weights
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False

    # Move model to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return tokenizer, model


def get_PubMedBERT_embedding(tokenizer, model, text):
    """
    Obtain the embedding for a given text using the PubMedBERT model.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the PubMedBERT model.
        model (transformers.PreTrainedModel): The PubMedBERT model.
        text (str): The input text for which the embedding is to be obtained.

    Returns:
        torch.Tensor: The embedding tensor for the input text.
    """
    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Tokenize the text and obtain the output tensors
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)

    # Move the tensors to GPU if available
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    model = model.to(device)

    # Get the embeddings
    with torch.no_grad():
        # Set output_hidden_states to True
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Get the last hidden state

        # Get the [CLS] token embedding rather than the embedding for each token
        sequence_embedding = embeddings[:, 0, :]

    return sequence_embedding
