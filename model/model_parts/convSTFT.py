import torch.nn as nn
from model.model_parts import STFTConvLayer


class ConvSTFT(nn.Module):
    def __init__(self, in_channel, filters, kernals, strides, dropout):
        super().__init__()
        err = "input length is not equal ({0}, {1})"
        assert len(filters) == len(kernals), err.format("filters", "kernals")
        assert len(filters) == len(strides), err.format("filters", "strides")
        assert len(kernals) == len(strides), err.format("kernals", "strides")

        stft = []
        in_ch = in_channel
        for out_ch, kernal, stride in zip(filters, kernals, strides):
            stft.append(STFTConvLayer(in_ch, out_ch, kernal, stride, dropout))
            in_ch = out_ch
        stft.append(nn.Flatten())
        self.stft = nn.ModuleList(stft)

    def forward(self, x):
        for module in self.stft:
            x = module(x)
        return x
