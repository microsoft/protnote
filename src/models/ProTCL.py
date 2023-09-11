import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
from src.utils.models import load_PubMedBERT, get_PubMedBERT_embedding


class ProTCL(nn.Module):
    def __init__(
            self,
            protein_embedding_dim,
            label_embedding_dim,
            latent_dim,
            temperature,
            sequence_embedding_matrix=None,
            train_label_embeddings=False,
            label_embedding_matrix=None,
            train_sequence_embeddings=False,):
        super().__init__()

        # Projection heads
        # TODO: Discuss whether or not to include bias in the projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # Temperature parameter
        self.t = temperature

        # If using a pre-trained sequence embedding matrix, create a nn.Embedding layer
        # TODO: Possibly allow this to train
        if sequence_embedding_matrix is not None:
            self.pretrained_sequence_embeddings = nn.Embedding.from_pretrained(
                sequence_embedding_matrix, freeze=not train_sequence_embeddings)
        # TODO: Support using ProteInfer here like we did with labels and PubMedBERT

        # If using a pre-trained label embedding matrix, create a nn.Embedding layer
        if label_embedding_matrix is not None:
            self.pretrained_label_embeddings = nn.Embedding.from_pretrained(
                label_embedding_matrix, freeze=not train_label_embeddings)
        # Otherwise, load the label pre-trained encoder and allow it to train
        else:
            # TODO: This doesn't work yet. Maybe we build an nn.Embedding but populate the matrix with the encoder? It would need to be re-calculated every pass, though, not just on init.
            self.label_tokenizer, self.label_encoder = load_PubMedBERT(
                trainable=train_label_embeddings)

        # Log the configurations
        logging.info(
            "################## Model encoder configurations ##################")

        logging.info(
            f"Using {'<<CACHED>>' if sequence_embedding_matrix is not None else '<<ProteInfer>>'} sequence embeddings with training {'<<ENABLED>>' if train_sequence_embeddings else '<<DISABLED>>'}."
        )

        logging.info(
            f"Using {'<<CACHED>>' if label_embedding_matrix is not None else '<<PubMedBERT>>'} label embeddings with training {'<<ENABLED>>' if train_label_embeddings else '<<DISABLED>>'}."
        )

        logging.info(
            "##################################################################")

    def forward(self, P, L):
        """
        Forward pass of the model.
        args:
            P: Tensor of protein sequences. Contains integer sequence IDs if using a pre-trained embedding matrix,
            otherwise contains one-hot encoded sequences.
            L: Tensor of labels of shape (batch_size, num_labels)
        """
        # Collapse labels to a single vector with an "any" operation
        collapsed_labels = torch.any(L, dim=0)

        # If using pre-trained label embeddings, convert labels to embeddings
        if hasattr(self, 'pretrained_label_embeddings'):
            L_f = self.pretrained_label_embeddings.weight[collapsed_labels]
        # If using a text encoder, convert labels to embeddings
        else:
            # Convert labels to embeddings using label encoder
            L_f = get_PubMedBERT_embedding(
                self.label_tokenizer, self.label_encoder, L)

        # If using pre-trained sequence embeddings, convert sequences to embeddings (since P is a tensor of sequence IDs)
        if hasattr(self, 'pretrained_sequence_embeddings'):
            P_f = self.pretrained_sequence_embeddings(P)
        # If using a protein sequence encoder, convert sequences to embeddings
        else:
            # TODO: Use ProteInfer here
            # Throw error
            raise ValueError(
                "Sequence embeddings not found. Please provide a pre-trained sequence embedding map or a protein encoder.")

        # Project protein and label embeddings to latent space
        P_e = F.normalize(self.W_p(P_f), dim=1)
        L_e = F.normalize(self.W_l(L_f), dim=1)

        return P_e, L_e
