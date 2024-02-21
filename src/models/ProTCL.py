import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils.models import get_label_embeddings
import os
from torchvision.ops import MLP

class ProTCL(nn.Module):
    def __init__(
        self,
        protein_embedding_dim=1100,
        label_embedding_dim=1024,
        label_embedding_pooling_method='mean',
        descriptions_per_label = 1,
        latent_dim=1024,
        label_encoder=None,
        sequence_encoder=None,
        label_encoder_num_trainable_layers=False,
        train_sequence_encoder=False,
        output_mlp_hidden_dim_scale_factor=1024,
        output_mlp_num_layers=2,
        output_neuron_bias=None,
        outout_mlp_add_batchnorm=True,
        dropout=0.0,
        sequence_embedding_dropout=0.0,
        label_embedding_dropout=0.0,
        projection_head_num_layers=1,
        projection_head_hidden_dim_scale_factor = 1,
        label_batch_size_limit=float("inf"),
        sequence_batch_size_limit=float("inf"),
        feature_fusion='concatenation',
        temperature=0.07,
    ):
        super().__init__()

        # Training options
        self.label_encoder_num_trainable_layers, self.train_sequence_encoder = label_encoder_num_trainable_layers, train_sequence_encoder

        # Encoders
        self.label_encoder, self.sequence_encoder = label_encoder, sequence_encoder
        self.descriptions_per_label = descriptions_per_label

        # Batch size limits
        self.label_batch_size_limit,  self.sequence_batch_size_limit = label_batch_size_limit, sequence_batch_size_limit

        self.feature_fusion = feature_fusion
        self.temperature = temperature
        self.label_embedding_pooling_method = label_embedding_pooling_method
        self.latent_dim = latent_dim

        # Projection heads
        self.W_p = MLP(protein_embedding_dim,
                                    [latent_dim*projection_head_hidden_dim_scale_factor]*(projection_head_num_layers-1) + [latent_dim],
                                    bias=False,
                                    norm_layer=torch.nn.BatchNorm1d,
                                    dropout=dropout)
        

        self.W_l = MLP(label_embedding_dim,
                       [latent_dim*projection_head_hidden_dim_scale_factor]*(projection_head_num_layers-1) + [latent_dim],
                       bias=False,
                       norm_layer=torch.nn.BatchNorm1d,
                       dropout=dropout
                       )
        #Doing it this way to avoid weight name changing even if dropout is 0 to avoid model loading issues.
        if sequence_embedding_dropout>0:
            self.W_p = nn.Sequential(nn.Dropout(sequence_embedding_dropout),self.W_p)
        if label_embedding_dropout>0:
            self.W_l = nn.Sequential(nn.Dropout(label_embedding_dropout),self.W_l)
        
        #MLP For raw attention score in case label embedding pooling method = all
        if self.label_embedding_pooling_method=='all':

            #TODO: this could be a simple mlp using get_mlp because it includes output neuron
            self.raw_attn_scorer = nn.Linear(label_embedding_dim,1, bias=True)

        if self.feature_fusion.startswith('concatenation'):
            # TODO: This could change. Currently keeping latent dim.
            self.output_layer = get_mlp(
                input_dim=self._get_concatenated_features_dim(),
                hidden_dim=int(round(output_mlp_hidden_dim_scale_factor*latent_dim)),
                num_layers=output_mlp_num_layers,
                output_neuron_bias=output_neuron_bias,
                batch_norm=outout_mlp_add_batchnorm,
                dropout=dropout,
            )

    def _get_concatenated_features_dim(self):
        
        dim = {'concatenation_diff':self.latent_dim*3,
               'concatenation_prod':self.latent_dim*3,
               'concatenation':self.latent_dim*2
               }
        return dim[self.feature_fusion]


    def _get_joint_embeddings(self, P_e, L_e, num_sequences,num_labels):

        sequence_embedding_dim = P_e.shape[1]
        label_embedding_dim = L_e.shape[1]

        # Use broadcasting so we don't have to expand the tensor dimensions

        joint_embeddings = torch.cat([
            P_e[:, None, :].expand(
                num_sequences, num_labels, sequence_embedding_dim),
            L_e[None, :, :].expand(
                num_sequences, num_labels, label_embedding_dim)
        ], dim=2).reshape(-1, sequence_embedding_dim + label_embedding_dim)

        if self.feature_fusion == 'concatenation_diff':
            joint_embeddings = torch.cat([joint_embeddings,
                                          (joint_embeddings[:,:sequence_embedding_dim] - joint_embeddings[:,sequence_embedding_dim:])],
                                          dim=-1)
        
        if self.feature_fusion == 'concatenation_prod':
            joint_embeddings = torch.cat([joint_embeddings,
                                          (joint_embeddings[:,:sequence_embedding_dim] * joint_embeddings[:,sequence_embedding_dim:])],
                                          dim=-1)

        return joint_embeddings
    
    def additive_attention(self,hidden_states,attention_mask):
        raw_attn_scores = self.raw_attn_scorer(hidden_states).squeeze(-1)
        
        #Masked scored for softmax
        raw_attn_scores = raw_attn_scores.masked_fill(attention_mask==0,float('-inf'))

        #Normalized attention weights
        attn_weights = torch.softmax(raw_attn_scores,dim=-1)

        #Get final label embedding
        return torch.bmm(attn_weights.unsqueeze(1),hidden_states).squeeze(1)

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
        
        # TODO: Remove sequence_embeddings and tokenized_labels from this code. They are not used. We always use the label_embeddings and sequence_onehots.
      
        #---------------------- LABEL EMBEDDING ----------------------# 
        if label_embeddings is not None and (self.label_encoder_num_trainable_layers == 0 or not self.training):
            # If label embeddings are provided and we don't need to propagate gradients (either because we aren't in training, or we didn't freeze the weights), use them.
            L_f = label_embeddings
        elif (tokenized_labels is not None) and self.training:
            # Otherwise, compute them on the fly (with or without gradients, depending on whether we are training the label encoder).
            if self.label_encoder_num_trainable_layers > 0:
                # Get label embeddings from tokens, with gradients
                L_f = get_label_embeddings(
                    tokenized_labels,
                    self.label_encoder,
                    method=self.label_embedding_pooling_method,
                    batch_size_limit=self.label_batch_size_limit
                )
            else:
                # Get label embeddings from tokens, without gradients
                with torch.no_grad():
                    L_f = get_label_embeddings(
                        tokenized_labels,
                        self.label_encoder,
                        method=self.label_embedding_pooling_method,
                        batch_size_limit=self.label_batch_size_limit
                    )
        else:
            raise ValueError(
                "Incompatible label parameters passed to forward method.")

        #---------------------- SEQUENCE EMBEDDINGS ----------------------# 
        if sequence_embeddings is not None and (not self.train_sequence_encoder or not self.training):
            # If sequence embeddings are provided and we don't need to propagate gradients (either because we aren't in training, or we didn't freeze the weights), use them.
            P_f = sequence_embeddings
        elif sequence_onehots is not None and sequence_lengths is not None: 
            # Otherwise, compute them on the fly (with or without gradients, depending on self.train_sequence_encoder).
            if self.train_sequence_encoder and self.training:
                # Compute embeddings with gradient calculations enabled
                P_f = self.sequence_encoder.get_embeddings(sequence_onehots, sequence_lengths)
            else:
                # Compute embeddings with gradient calculations disabled
                with torch.no_grad():
                    P_f = self.sequence_encoder.get_embeddings(sequence_onehots, sequence_lengths)
        else:
            raise ValueError(
                "Incompatible sequence parameters passed to forward method.")

        
        if self.label_embedding_pooling_method=='all':
            L_f = self.additive_attention(L_f,tokenized_labels['attention_mask'])

        # Project protein and label embeddings to common latent space.
        P_e = self.W_p(P_f)
        L_e = self.W_l(L_f)

        num_sequences = P_e.shape[0]
        num_labels = L_e.shape[0]

        # Get concatenated embeddings, representing all possible combinations of protein and label embeddings
        # (number proteins * number labels by latent_dim*2)

        if self.feature_fusion=='similarity':
            P_e = F.normalize(P_e,dim=-1,p=2)
            L_e = F.normalize(L_e,dim=-1,p=2)
            logits = torch.mm(P_e, L_e.t()) / self.temperature
            
        elif self.feature_fusion.startswith('concatenation'):
            joint_embeddings = self._get_joint_embeddings(
                P_e, L_e, num_sequences, num_labels)

            # Feed through MLP to get logits (which represent similarities)
            logits = self.output_layer(joint_embeddings)
        else:
            raise ValueError("feature fusion method not implemented")
        
        # Reshape for loss 
        if self.descriptions_per_label==1:
            logits = logits.reshape(num_sequences, num_labels)
        else:
            assert not self.training, "label ensembling shouldn't be done during training"
            # Get equivalent logit of averaging in probability space

            logits = torch.special.logit(torch.sigmoid(logits)\
                .reshape(num_sequences,
                         num_labels//self.descriptions_per_label,
                         self.descriptions_per_label).mean(axis=-1),eps=1e-7)

            # logits = logits\
            #             .reshape(num_sequences,
            #              num_labels//self.descriptions_per_label,
            #              self.descriptions_per_label).mean(axis=-1)
        return logits

def get_mlp(input_dim,
            hidden_dim,
            num_layers,
            input_dropout=0.0,
            dropout=0.0,
            batch_norm = False,
            output_neuron_bias=None):
    """
    Creates a variable length MLP with ReLU activations.
    """
    layers = []

    if input_dropout>0:
        layers.append(nn.Dropout(input_dropout))

    add_hidden_layers_bias =  not batch_norm
   
    for idx in range(num_layers):
        if idx ==0:
            layers.append(nn.Linear(input_dim, hidden_dim,bias=add_hidden_layers_bias))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim,bias=add_hidden_layers_bias))
 
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
 
        layers.append(nn.ReLU())
        
        # Add dropout after each ReLU activation (except the final layer)
        if idx < num_layers - 1:
            layers.append(nn.Dropout(dropout))
       
    output_neuron = nn.Linear(hidden_dim, 1)
    if output_neuron_bias is not None:
        # Set the bias of the final linear layer
        output_neuron.bias.data.fill_(output_neuron_bias)
    layers.append(output_neuron)
    return nn.Sequential(*layers)
