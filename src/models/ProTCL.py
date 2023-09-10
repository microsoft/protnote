import torch.nn as nn
import torch.nn.functional as F
import torch


class ProTCL(nn.Module):
    def __init__(
            self,
            protein_embedding_dim,
            label_embedding_dim,
            latent_dim,
            temperature,
            sequence_embedding_matrix=None,
            label_embedding_matrix=None):
        super().__init__()

        # Projection heads
        # TODO: Discuss whether or not to include bias in the projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # Temperature parameter
        self.t = nn.Parameter(torch.tensor(temperature))

        # If using a pre-trained sequence embedding matrix, create a nn.Embedding layer
        # TODO: Possibly allow this to train
        if sequence_embedding_matrix is not None:
            self.pretrained_sequence_embeddings = nn.Embedding.from_pretrained(
                sequence_embedding_matrix, freeze=True)

        # If using a pre-trained label embedding matrix, create a nn.Embedding layer
        # TODO: Possibly allow this to train
        if label_embedding_matrix is not None:
            self.pretrained_label_embeddings = nn.Embedding.from_pretrained(
                label_embedding_matrix, freeze=True)

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
        if self.pretrained_label_embeddings is not None:
            L_f = self.pretrained_label_embeddings.weight[collapsed_labels]
        # If using a text encoder, convert labels to embeddings
        else:
            # Throw error
            raise ValueError(
                "Label embeddings not found. Please provide a pre-trained label embedding map or a text encoder.")

        # If using pre-trained sequence embeddings, convert sequences to embeddings (since P is a tensor of sequence IDs)
        if self.pretrained_sequence_embeddings is not None:
            P_f = self.pretrained_sequence_embeddings(P)
        # If using a protein sequence encoder, convert sequences to embeddings
        else:
            # Throw error
            raise ValueError(
                "Sequence embeddings not found. Please provide a pre-trained sequence embedding map or a protein encoder.")

        # Project protein and label embeddings to latent space
        P_e = F.normalize(self.W_p(P_f), dim=1)
        L_e = F.normalize(self.W_l(L_f), dim=1)

        return P_e, L_e
