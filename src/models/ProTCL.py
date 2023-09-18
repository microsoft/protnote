import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.models import get_embeddings


class ProTCL(nn.Module):
    def __init__(
        self,
        protein_embedding_dim,
        label_embedding_dim,
        latent_dim,
        temperature,
        label_encoder,
        label_tokenizer,
        label_annotation_map,
        int_label_id_map,
        sequence_encoder,
        sequence_embedding_matrix,
        label_embedding_matrix,
        train_projection_head,
        train_label_encoder,
        train_sequence_encoder,
    ):
        super().__init__()

        # Map integer indices to GO label ids (based on the vocabulary)
        self.int2label = int_label_id_map
        # Map GO label ids to label annotations
        self.label_annotation_map = label_annotation_map

        # Default label embedding cache
        self.cached_label_embeddings = None

        # Training options
        self.train_label_encoder = train_label_encoder
        self.train_sequence_encoder = train_sequence_encoder

        # Encoders and tokenizers
        self.label_encoder = label_encoder
        self.label_tokenizer = label_tokenizer
        self.sequence_encoder = sequence_encoder

        # Projection heads
        # TODO: Discuss whether or not to include bias in the projection heads
        self.W_p = nn.Linear(protein_embedding_dim, latent_dim, bias=False)
        self.W_l = nn.Linear(label_embedding_dim, latent_dim, bias=False)

        # Optionally freeze projection head weights
        if not train_projection_head:
            self.W_p.weight.requires_grad = False
            self.W_l.weight.requires_grad = False

        # Temperature parameter
        self.t = temperature

        # If using a cached sequence embedding matrix, create a nn.Embedding layer
        if sequence_embedding_matrix is not None and not train_sequence_encoder:
            self.cached_sequence_embeddings = nn.Embedding.from_pretrained(
                sequence_embedding_matrix, freeze=True
            )

        # If using a cached label embedding matrix, create a nn.Embedding layer
        if label_embedding_matrix is not None and not train_label_encoder:
            self.cached_label_embeddings = nn.Embedding.from_pretrained(
                label_embedding_matrix, freeze=True
            )

    def forward(self, P, L, sequence_lengths):
        """
        Forward pass of the model.
        args:
            P: Tensor of protein sequences. Contains integer sequence IDs if using a pre-trained embedding matrix,
            otherwise contains one-hot encoded sequences.
            L: Tensor of label indices of shape (num_labels)
        """

        # If not training the label encoder, used the cached label embeddings (passed in as a parameter)
        if not self.train_label_encoder or (self.cached_label_embeddings is not None and not self.training):
            L_f = self.cached_label_embeddings(L)
        # Otherwise, compute embeddings for the considered labels
        else:
            # For each of the labels, get the corresponding label id from int2label
            # TODO: This is an O(n) way to get the label ids. We should be able to do this with torch tensors in constant time.
            label_ids = [self.int2label[idx.item()] for idx in L]

            # Lookup the label strings for each label_id
            label_annotations = [self.label_annotation_map[label_id]
                                 for label_id in label_ids]

            # If in training loop or we haven't cached the label embeddings, compute the embeddings
            if self.training or self.cached_label_embeddings is None:
                L_f = get_embeddings(
                    label_annotations,
                    self.label_tokenizer,
                    self.label_encoder,
                )
                # If not training, cache the label embeddings
                if not self.training:
                    self.cached_label_embeddings = nn.Embedding.from_pretrained(
                        L_f, freeze=True
                    )

        # If using pre-trained sequence embeddings, convert sequences to embeddings (since P is a tensor of sequence IDs)
        if hasattr(self, "cached_sequence_embeddings"):
            P_f = self.cached_sequence_embeddings(P)
        # If using a protein sequence encoder, convert sequences to embeddings
        else:
            # TODO: Have not tested this.
            with torch.set_grad_enabled(True), torch.cuda.amp.autocast():
                P_f = self.sequence_encoder.get_embeddings(P, sequence_lengths)
            # Throw error
            raise ValueError(
                "Sequence embeddings not found. Please provide a pre-trained sequence embedding map or a protein encoder."
            )

        # Project protein and label embeddings to latent space and L2 normalize
        P_e = F.normalize(self.W_p(P_f), dim=1)
        L_e = F.normalize(self.W_l(L_f), dim=1)

        return P_e, L_e

    def clear_label_embeddings_cache(self):
        self.cached_label_embeddings = None
