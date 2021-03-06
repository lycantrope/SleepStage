import torch.nn as nn

from .model_layers import Conv1DLayer


class ConvNet(nn.Module):
    def __init__(
        self, input_dims, in_channel, filters, kernels, strides, skip_by, dropout
    ):
        super().__init__()
        err = "input length is not equal ({0}, {1})"
        assert len(filters) == len(kernels), err.format("filters", "kernels")
        assert len(filters) == len(strides), err.format("filters", "strides")
        assert len(kernels) == len(strides), err.format("kernels", "strides")

        in_ch = in_channel
        skip_in_ch = in_ch
        acc_stride = 1

        dims_counts = int(input_dims)
        skip_outputDims = int(input_dims)
        convs = {}
        skips = {}
        for id, (out_ch, kernel, stride) in enumerate(zip(filters, kernels, strides)):
            convs[f"conv{id}"] = Conv1DLayer(in_ch, out_ch, kernel, stride, dropout)
            in_ch = out_ch
            acc_stride = acc_stride * stride
            # count dimensions
            dims_counts = (dims_counts - 1) // stride + 1

            if id % skip_by:
                skip_kernel_size = kernel

            if id % skip_by == (-1) % skip_by:

                skips[f"skip_conv{id}"] = nn.Conv1d(
                    skip_in_ch,
                    out_ch,
                    kernel_size=skip_kernel_size,
                    stride=acc_stride,
                    padding=int(skip_kernel_size - 1) // 2,
                    bias=False,
                )
                # count dimensions
                skip_outputDims = (skip_outputDims - 1) // acc_stride + 1
                acc_stride = 1
                skip_in_ch = out_ch

        self.skip_by = skip_by
        self.outputChannels = out_ch  # output channels
        self.outputDims = dims_counts  # output dimensions
        self.outputSize = out_ch * dims_counts  # channels * dimensions
        self.skip_outputDims = skip_outputDims

        self.convs = nn.ModuleDict(convs)
        self.skips = nn.ModuleDict(skips)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_copy = x
        for id, conv in self.convs.items():
            x = conv(x)
            skip_key = "skip_" + id
            if skip_key in self.skips:
                x = x + self.skips[skip_key](x_copy)
                x_copy = x
        return self.flatten(x)
