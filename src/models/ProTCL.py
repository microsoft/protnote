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
        label_embedding_matrix,
        train_label_encoder,
        train_sequence_encoder,
        output_dim,
        output_num_layers
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

        # Temperature parameter
        self.t = temperature

        # Sequence encoder
        self.sequence_encoder = sequence_encoder

        # If using a cached label embedding matrix, create a nn.Embedding layer
        if label_embedding_matrix is not None and not train_label_encoder:
            self.cached_label_embeddings = nn.Embedding.from_pretrained(
                label_embedding_matrix, freeze=True
            )

        #TODO: This could change. Currently keeping latent dim.
        self.output_layer = get_mlp(latent_dim*2,output_dim,output_num_layers)
       
    def _get_joint_embeddings(self, P_e, L_e):
        # Input stats
        num_sequences = P_e.shape[0]
        num_labels = L_e.shape[0]
        sequence_embedding_dim = P_e.shape[1]
        label_embedding_dim = L_e.shape[1]

        
        # Expand protein and label embeddings to all combinations of sequences and labels.
        P_e_expanded = P_e.unsqueeze(1).expand(-1, num_labels, -1).reshape(-1, sequence_embedding_dim)
        L_e_expanded = L_e.unsqueeze(0).expand(num_sequences, -1, -1).reshape(-1, label_embedding_dim)

        # Conatenate protein and label embeddings. Shape: (batch_size, latent_dim*2)
        joint_embeddings = torch.cat([P_e_expanded, L_e_expanded], dim=1)
        return joint_embeddings,num_sequences,num_labels

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

        # Get sequence embeddings
        P_f = self.sequence_encoder.get_embeddings(P, sequence_lengths)

        # Project protein and label embeddings to common latent space.
        P_e = self.W_p(P_f)
        L_e = self.W_l(L_f)

        joint_embeddings,num_sequences,num_labels = self._get_joint_embeddings(P_e, L_e)

        #Feed through MLP
        logits = self.output_layer(joint_embeddings)

        # Reshape for loss function
        logits = logits.reshape(num_sequences, num_labels)
        
        return logits

    def clear_label_embeddings_cache(self):
        self.cached_label_embeddings = None

#Create variable length MLP that can change dimension and then all layers 
#are the same size.
def get_mlp(input_dim,output_dim,num_layers):
    layers = []
    layers.append(nn.Linear(input_dim,output_dim))
    layers.append(nn.ReLU())
    for _ in range(num_layers-1):
        layers.append(nn.Linear(output_dim,output_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(output_dim,1))
    return nn.Sequential(*layers)