import torch
import torch.nn as nn
from .model_layers import STFTConvLayer


class ConvSTFT(nn.Module):
    def __init__(self, in_channel, filters, kernels, strides, dropout):
        super().__init__()
        err = "input length is not equal ({0}, {1})"
        assert len(filters) == len(kernels), err.format("filters", "kernels")
        assert len(filters) == len(strides), err.format("filters", "strides")
        assert len(kernels) == len(strides), err.format("kernels", "strides")

        stft = []
        in_ch = in_channel
        for out_ch, kernel, stride in zip(filters, kernels, strides):
            stft.append(STFTConvLayer(in_ch, out_ch, kernel, stride, dropout))
            in_ch = out_ch
        stft.append(nn.Flatten())
        self.stft = nn.ModuleList(stft)

        self.outputChannels = out_ch

    def forward(self, x):
        for module in self.stft:
            x = module(x)
        return x
