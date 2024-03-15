import torch
from typing import Literal, Text, Optional
import numpy as np
from ..data.datasets import set_padding_to_sentinel
from src.utils.proteinfer import transfer_tf_weights_to_torch
from transformers import AutoConfig, AutoTokenizer, EsmModel
import torch.nn as nn

class MaskedConv1D(torch.nn.Conv1d):
    def forward(self, x, sequence_lengths):
        '''
        Correct for padding before and after. Can be redundant
        but reduces overhead of setting padding to sentiel in other contexts.
        '''
        x = set_padding_to_sentinel(x, sequence_lengths, 0)
        x = super().forward(x)
        x = set_padding_to_sentinel(x, sequence_lengths, 0)
        return x

# ResNet-V2 https://arxiv.org/pdf/1602.07261v2.pdf


class Residual(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 kernel_size: int,
                 dilation: int,
                 bottleneck_factor: float,
                 activation=torch.nn.ReLU
                 ):
        super().__init__()

        bottleneck_out_channels = int(
            np.floor(input_channels * bottleneck_factor))
        self.bn_activation_1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_channels, eps=0.001, momentum=0.01),
            activation())

        self.masked_conv1 = MaskedConv1D(in_channels=input_channels,
                                         out_channels=bottleneck_out_channels,
                                         padding='same',
                                         kernel_size=kernel_size,
                                         stride=1,
                                         dilation=dilation
                                         )
        self.bn_activation_2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(bottleneck_out_channels,
                                 eps=0.001, momentum=0.01),
            activation())

        self.masked_conv2 = MaskedConv1D(in_channels=bottleneck_out_channels,
                                         out_channels=input_channels,
                                         padding='same',
                                         kernel_size=1,
                                         stride=1,
                                         dilation=1
                                         )

    def forward(self, x, sequence_lengths):
        out = self.bn_activation_1(x)
        out = self.masked_conv1(out, sequence_lengths)
        out = self.bn_activation_2(out)
        out = self.masked_conv2(out, sequence_lengths)
        out = out + x
        return out


class ProteInfer(torch.nn.Module):

    def __init__(self,
                 num_labels: int,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 activation,
                 dilation_base: int,
                 num_resnet_blocks: int,
                 bottleneck_factor: float
                 ):
        super().__init__()

        self.conv1 = MaskedConv1D(in_channels=input_channels,
                                  out_channels=output_channels,
                                  padding='same',
                                  kernel_size=kernel_size,
                                  stride=1,
                                  dilation=1
                                  )
        self.resnet_blocks = torch.nn.ModuleList()

        for i in range(num_resnet_blocks):
            self.resnet_blocks.append(
                Residual(input_channels=output_channels,
                         kernel_size=kernel_size,
                         dilation=dilation_base**i,
                         bottleneck_factor=bottleneck_factor,
                         activation=activation)
            )

        self.output_layer = torch.nn.Linear(
            in_features=output_channels, out_features=num_labels)

    def get_embeddings(self, x, sequence_lengths):
        features = self.conv1(x, sequence_lengths)
        # Sequential doesn't work here because of multiple inputs
        for idx, resnet_block in enumerate(self.resnet_blocks):
            features = resnet_block(features, sequence_lengths)
        features = set_padding_to_sentinel(features, sequence_lengths, 0)
        features = (torch.sum(features, dim=-1) /
                    sequence_lengths.unsqueeze(-1))  # Average pooling
        return features

    def forward(self, x, sequence_lengths):
        features = self.get_embeddings(x, sequence_lengths)
        logits = self.output_layer(features)
        return logits

    @classmethod
    def from_pretrained(cls,
                        weights_path: str,
                        num_labels: int,
                        input_channels: int,
                        output_channels: int,
                        kernel_size: int,
                        activation,
                        dilation_base: int,
                        num_resnet_blocks: int,
                        bottleneck_factor: float
                        ):
        '''
        Load a pretrained model from a path or url.
        '''
        model = cls(num_labels,
                    input_channels,
                    output_channels,
                    kernel_size,
                    activation,
                    dilation_base,
                    num_resnet_blocks,
                    bottleneck_factor
                    )
        transfer_tf_weights_to_torch(model, weights_path)

        return model

class SequenceEncoder(nn.Module):
    def __init__(self, encoder_type, pretrained, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'proteinfer':
            self._init_proteinfer(pretrained, **kwargs)
        elif encoder_type == 'esm-2':
            self._init_esm2(pretrained)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _init_proteinfer(self, pretrained, paths, config):
        assert paths and config, "Paths and config must be provided for ProteInfer"
        if pretrained:
            self.encoder = ProteInfer.from_pretrained(
                weights_path=paths["PROTEINFER_WEIGHTS_PATH"],
                num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
                input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
                output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
                kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
                activation=torch.nn.ReLU,
                dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
                num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
                bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
            )
        else:
            self.encoder = ProteInfer(
                num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
                input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
                output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
                kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
                activation=torch.nn.ReLU,
                dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
                num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
                bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
            )
            self.tokenizer = None

    def _init_esm2(self, pretrained):
        if not pretrained:
            raise ValueError("Pretrained model must be provided for ESM-2")
        esm_config = AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.encoder = EsmModel(esm_config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def get_embeddings(self, sequences=None, sequence_onehots=None, sequence_lengths=None):
        """
        Get embeddings for the given batch of sequences.
        :param sequence_onehots: One-hot encoded sequence (for ProteInfer)
        :param sequence_str: Sequence string (for ESM-2)
        :return: Embeddings for the given batch of sequences as a tensor 
        """
        if self.encoder_type == 'proteinfer':
            return self._get_proteinfer_embeddings(sequence_onehots, sequence_lengths)
        elif self.encoder_type == 'esm-2':
            return self._get_esm2_embeddings(sequences, sequence_lengths)
        else:
            raise ValueError("Incompatible encoder type")

    def _get_proteinfer_embeddings(self, sequence_onehots, sequence_lengths):
        if sequence_onehots is None or sequence_lengths is None:
            raise ValueError("One-hot encoded sequence must be provided for ProteInfer")
        return self.encoder.get_embeddings(sequence_onehots, sequence_lengths)

    def _mean_pool(iself, output, sequence_lengths):
        batch_size, _, embedding_dim = output.shape
        embeddings = torch.zeros(batch_size, embedding_dim, device=output.device)
        
        for i in range(batch_size):
            # Calculate mean from 1 to sequence_length + 1 for each sequence
            embeddings[i] = output[i, 1:sequence_lengths[i] + 1].mean(dim=0)
        
        return embeddings

    def _get_esm2_embeddings(self, sequences, sequence_lengths):
        if sequences is None:
            raise ValueError("Sequence string must be provided for Hugging Face model")
        inputs = self.tokenizer(sequences, return_tensors='pt', padding='max_length', truncation=True, max_length=None).to(self.encoder.device)
        output = self.encoder(**inputs).last_hidden_state
        # Perform mean pooling over the sequence length 
        # TODO: Add support for other pooling methods
        return self._mean_pool(output, sequence_lengths)
