import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.models import get_label_embeddings


class ProTCL(nn.Module):
    def __init__(
        self,
        protein_embedding_dim=1100,
        label_embedding_dim=1024,
        latent_dim=1024,
        label_encoder=None,
        sequence_encoder=None,
        train_label_encoder=False,
        train_sequence_encoder=False,
        output_dim=1024,
        output_num_layers=2
    ):
        super().__init__()

        # Default label embedding cache
        self.cached_label_embeddings = None

        # Training options
        self.train_label_encoder, self.train_sequence_encoder = train_label_encoder, train_sequence_encoder

        # Encoders
        self.label_encoder, self.sequence_encoder = label_encoder, sequence_encoder

        # Projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # TODO: This could change. Currently keeping latent dim.
        self.output_layer = get_mlp(
            latent_dim*2, output_dim, output_num_layers)

    def _get_joint_embeddings(self, P_e, L_e):
        # Input stats
        num_sequences = P_e.shape[0]
        num_labels = L_e.shape[0]
        sequence_embedding_dim = P_e.shape[1]
        label_embedding_dim = L_e.shape[1]

        # Expand protein and label embeddings to all combinations of sequences and labels.
        P_e_expanded = P_e.unsqueeze(
            1).expand(-1, num_labels, -1).reshape(-1, sequence_embedding_dim)
        L_e_expanded = L_e.unsqueeze(0).expand(
            num_sequences, -1, -1).reshape(-1, label_embedding_dim)

        # Conatenate protein and label embeddings. Shape: (batch_size, latent_dim*2)
        joint_embeddings = torch.cat([P_e_expanded, L_e_expanded], dim=1)
        return joint_embeddings, num_sequences, num_labels

    def forward(
        self,
        sequence_onehots=None,
        sequence_embeddings=None,
        sequence_lengths=None,
        tokenized_labels=None,
        label_embeddings=None
    ):
        """
        Forward pass of the model.
        Returns a representation of the similarity between each sequence and each label.
        args:
            sequence_onehots (optional): Tensor of one-hot encoded protein sequences.
            sequence_embeddings (optional): Tensor of pre-trained sequence embeddings.
            sequence_lengths (optional): Tensor of sequence lengths.
            tokenized_labels (optional): List of tokenized label sequences.
            label_embeddings (optional): Tensor of pre-trained label embeddings.
        """

        # If label embeddings are provided and we're not training the laebel encoder, use them. Otherwise, compute them.
        if label_embeddings is not None and not self.train_label_encoder:
            L_f = label_embeddings
        elif tokenized_labels is not None:
            # Throw an error
            raise ValueError(
                "Training label encoder is not currently supported. ")

            # If in training loop or we haven't cached the label embeddings, compute the embeddings
            if self.training or self.cached_label_embeddings is None:
                # Get label embeddings from tokens
                L_f = get_label_embeddings(
                    tokenized_labels,
                    self.label_encoder,
                )
                # If not training, cache the label embeddings
                if not self.training:
                    # TODO: Rather than an nn.Embedding layer, this should be a mapping from token to embedding
                    self.cached_label_embeddings = nn.Embedding.from_pretrained(
                        L_f, freeze=True
                    )
        else:
            raise ValueError(
                "Incompatible label parameters passed to forward method.")

        # If sequence embeddings are provided and we're not training the sequence encoder, use them. Otherwise, compute them.
        if sequence_embeddings is not None and not self.train_sequence_encoder:
            P_f = sequence_embeddings
        elif sequence_onehots is not None and sequence_lengths is not None:
            # Throw an error
            raise ValueError(
                "Training sequence encoder is not currently supported. ")
            P_f = self.sequence_encoder.get_embeddings(
                sequence_onehots, sequence_lengths)
        else:
            raise ValueError(
                "Incompatible sequence parameters passed to forward method.")

        # Project protein and label embeddings to common latent space.
        P_e = self.W_p(P_f)
        L_e = self.W_l(L_f)

        # Get concatenated embeddings, representing all possible combinations of protein and label embeddings
        # (number proteins * number labels by latent_dim*2)
        joint_embeddings, num_sequences, num_labels = self._get_joint_embeddings(
            P_e, L_e)

        # Feed through MLP to get logits (which represent similarities)
        logits = self.output_layer(joint_embeddings)

        # Reshape for loss function
        logits = logits.reshape(num_sequences, num_labels)

        return logits

    def clear_label_embeddings_cache(self):
        """
        Clears the cached label embeddings, forcing the model to recompute them on the next forward pass.
        """
        self.cached_label_embeddings = None


def get_mlp(input_dim, output_dim, num_layers):
    """
    Creates a variable length MLP with ReLU activations.
    """
    layers = []
    layers.append(nn.Linear(input_dim, output_dim))
    layers.append(nn.ReLU())
    for _ in range(num_layers-1):
        layers.append(nn.Linear(output_dim, output_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(output_dim, 1))
    return nn.Sequential(*layers)
