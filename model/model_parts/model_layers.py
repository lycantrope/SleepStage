import torch.nn as nn


class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class STFTConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dropout):
        super().__init__()
        self.stft_conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.stft_conv(x)
