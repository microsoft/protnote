import torch.nn as nn
import torch.nn.functional as F
import torch


class ProTCL(nn.Module):
    def __init__(
            self,
            protein_embedding_dim,
            label_embedding_dim,
            latent_dim, temperature,
            sequence_to_embeddings_dict=None,
            ordered_label_embeddings=None):
        super().__init__()

        # Projection heads
        # TODO: Discuss whether or not to include bias in the projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # Temperature parameter
        self.t = nn.Parameter(torch.tensor(temperature))

        # TODO: Have __init__ take the arguments train_sequence_encoder and train_label_encoder and handle here
        # Pre-trained sequence embedding map
        self.sequence_to_embeddings_dict = sequence_to_embeddings_dict

        # Pre-trained label embedding map
        self.ordered_label_embeddings = ordered_label_embeddings

    def forward(self, P, L):
        # Collapse labels to a single vector with an "any" operation
        collapsed_labels = torch.any(L, dim=0)

        # If using pre-trained label embeddings, convert labels to embeddings
        if self.ordered_label_embeddings is not None:
            L_f = self.ordered_label_embeddings[collapsed_labels]
        # If using a text encoder, convert labels to embeddings
        else:
            # Throw error
            raise ValueError(
                "Label embeddings not found. Please provide a pre-trained label embedding map or a text encoder.")

        # If using pre-trained sequence embeddings, convert sequences to embeddings
        if self.sequence_to_embeddings_dict is not None:
            P_f = torch.stack(
                [self.sequence_to_embeddings_dict[seq] for seq in P])
        # If using a protein sequence encoder, convert sequences to embeddings
        else:
            # Throw error
            raise ValueError(
                "Sequence embeddings not found. Please provide a pre-trained sequence embedding map or a protein encoder.")

        # Move P_f to GPU
        P_f = P_f.to(L_f.device)

        # Project protein and label embeddings to latent space
        P_e = F.normalize(self.W_p(P_f), dim=1)
        L_e = F.normalize(self.W_l(L_f), dim=1)

        return P_e, L_e
