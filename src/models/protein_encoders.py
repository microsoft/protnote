import torch
from typing import Literal, Text, Optional
import numpy as np
from ..data.datasets import set_padding_to_sentinel
from src.utils.proteinfer import transfer_tf_weights_to_torch


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
