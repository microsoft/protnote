from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def load_PubMedBERT():
    # Load PubMedBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    
    # Move model to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return tokenizer, model


def get_PubMedBERT_embedding(tokenizer, model, text):
    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Tokenize the text and obtain the output tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move the tensors to GPU if available
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    model = model.to(device)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  # Set output_hidden_states to True
        embeddings = outputs.hidden_states[-1]  # Get the last hidden state

        # Get the [CLS] token embedding rather than the embedding for each token
        sequence_embedding = embeddings[:, 0, :]

    # Convert the tensor to a numpy array
    sequence_embedding = sequence_embedding.cpu().numpy()

    return sequence_embedding


