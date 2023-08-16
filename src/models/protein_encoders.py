import torch
from typing import Literal,Text,Optional
import numpy as np


def set_padding_to_sentinel(padded_representations, sequence_lengths, sentinel):
    # Create a sequence mask
    seq_mask = torch.arange(padded_representations.size(1)).to(sequence_lengths.device) < sequence_lengths[:, None]

    # Use broadcasting to expand the mask to match the shape of padded_representations
    return torch.where(seq_mask.unsqueeze(-1), padded_representations, sentinel)

class SequentialMultiInput(torch.nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs

class MaskedConv1D(torch.nn.Conv1d):
    def forward(self,x,sequence_lengths):
        '''
        Correct for padding before and after. Can be redundant
        but reduces overhead of setting padding to sentiel in other contexts.
        '''
        x = set_padding_to_sentinel(x,sequence_lengths,0)
        x = super().forward(x)
        x = set_padding_to_sentinel(x,sequence_lengths,0)
        return x

#ResNet-V2 https://arxiv.org/pdf/1602.07261v2.pdf
class Residual(torch.nn.Module):
    def __init__(self,
                 input_channels:int,
                 kernel_size:int,
                 dilation: int,
                 bottleneck_factor:float,
                 activation = torch.nn.ReLU
                 ):
        super().__init__()

        bottleneck_out_channels = int(np.floor(input_channels * bottleneck_factor))
        self.bn_activation_1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_channels),
            activation())
        
        self.masked_conv1 = MaskedConv1D(in_channels=input_channels,
                            out_channels=bottleneck_out_channels,
                            padding= 'same',
                            kernel_size=kernel_size,
                            stride=1,
                            dilation=dilation
                            )
        self.bn_activation_2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(bottleneck_out_channels),
            activation())
            
        self.masked_conv2 = MaskedConv1D(in_channels=bottleneck_out_channels,
                            out_channels=input_channels,
                            padding='same',
                            kernel_size=1,
                            stride=1,
                            dilation=1
                            )
            

    def forward(self,x,sequence_lengths):
        out = self.bn_activation_1(x)
        out = self.masked_conv1(out,sequence_lengths)
        out = self.bn_activation_2(out)
        out = self.masked_conv2(out,sequence_lengths)

        return out + x
    
class ProteInfer(torch.nn.Module):

    def __init__(self,
                 num_labels,
                 input_channels,
                 output_channels,
                 kernel_size,
                 activation,
                 dilation_base,
                 num_resnet_blocks,
                 bottleneck_factor
                 ):
        super().__init__()

        self.conv1 = MaskedConv1D(in_channels=input_channels,
                                  out_channels=output_channels,
                                  padding= 'same',
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
                        activation = activation)
                     )
        self.output_layer = torch.nn.Linear(in_features=output_channels,out_features=num_labels)

    def forward(self,x,sequence_lengths):
        features = self.conv1(x,sequence_lengths)

        #Sequential doesn't work here because of multiple inputs
        for resnet_block in self.resnet_blocks:
            features = resnet_block(features,sequence_lengths)

        features = (torch.sum(features,dim=-1)/sequence_lengths.unsqueeze(-1)) #Works because convs are masked
        logits = self.output_layer(features)
        return logits

        