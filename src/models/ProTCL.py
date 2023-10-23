import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.models import get_label_embeddings
import os


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
        output_mlp_hidden_dim_scale_factor=1024,
        output_mlp_num_layers=2,
        output_neuron_bias=None,
        label_batch_size_limit=float("inf"),
        sequence_batch_size_limit=float("inf"),
    ):
        super().__init__()

        # Training options
        self.train_label_encoder, self.train_sequence_encoder = train_label_encoder, train_sequence_encoder

        # Encoders
        self.label_encoder, self.sequence_encoder = label_encoder, sequence_encoder

        # Batch size limits
        self.label_batch_size_limit,  self.sequence_batch_size_limit = label_batch_size_limit, sequence_batch_size_limit

        # Projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # TODO: This could change. Currently keeping latent dim.
        self.output_layer = get_mlp(
            input_dim=latent_dim*2,
            hidden_dim=int(round(output_mlp_hidden_dim_scale_factor*latent_dim)),
            num_layers=output_mlp_num_layers,
            output_neuron_bias=output_neuron_bias
        )

    def _get_joint_embeddings(self, P_e, L_e):
        num_sequences = P_e.shape[0]
        num_labels = L_e.shape[0]
        sequence_embedding_dim = P_e.shape[1]
        label_embedding_dim = L_e.shape[1]

        # Use broadcasting so we don't have to expand the tensor dimensions
        joint_embeddings = torch.cat([
            P_e[:, None, :].expand(
                num_sequences, num_labels, sequence_embedding_dim),
            L_e[None, :, :].expand(
                num_sequences, num_labels, label_embedding_dim)
        ], dim=2).reshape(-1, sequence_embedding_dim + label_embedding_dim)

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
        if label_embeddings is not None and (not self.train_label_encoder or (self.train_label_encoder and not self.training)):
            L_f = label_embeddings
        elif tokenized_labels is not None:
            # Get label embeddings from tokens
            with torch.set_grad_enabled(self.train_label_encoder and self.training):
                L_f = get_label_embeddings(
                    tokenized_labels,
                    self.label_encoder,
                    batch_size_limit=self.label_batch_size_limit
                )
        else:
            raise ValueError(
                "Incompatible label parameters passed to forward method.")

        # If sequence embeddings are provided and we're not training the sequence encoder, use them. Otherwise, compute them.
        if sequence_embeddings is not None and not self.train_sequence_encoder:
            P_f = sequence_embeddings
        elif sequence_onehots is not None and sequence_lengths is not None:
            
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


def get_mlp(input_dim,
            hidden_dim,
            num_layers,
            output_neuron_bias=None):
    """
    Creates a variable length MLP with ReLU activations.
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    output_neuron = nn.Linear(hidden_dim, 1)
    if output_neuron_bias is not None:
        # Set the bias of the final linear layer
        output_neuron.bias.data.fill_(output_neuron_bias)
    layers.append(output_neuron)
    return nn.Sequential(*layers)
