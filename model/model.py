import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from model.model_parts import ConvNet, ConvSTFT, ClassifyNet


class CNN_LSTM(BaseModel):
    def __init__(
        self,
        num_classes: int,
        data_params: dict,
        convnet_params: dict,
        stft_params: dict,
        classifier_params: dict,
        dropout: float,
    ):
        super().__init__()

        # data_params
        self.data_params = data_params
        self.FreqBoundary = self.data_params["FreqBoundary"]
        FreqBinWidth = self.data_params["FreqBinWidth"]
        FreqWidth = abs(self.FreqBoundary[0] - self.FreqBoundary[1])
        self.spectrumFeatureNums = round(FreqWidth / FreqBinWidth)

        # model HYPERPARAMS
        # ConvNet PARAMS
        self.convnet_params = convnet_params
        self.convnet_params["dropout"] = dropout

        # ConvSTFT PARAMS
        self.stft_params = stft_params
        self.stft_params["dropout"] = dropout

        # ClassifierNet PARAMS
        self.classifier_params = classifier_params
        self.classifier_params["dropout"] = dropout
        self.classifier_params["num_classes"] = num_classes

        # conv for rawdata
        self.convNet = ConvNet(**self.convnet_params)
        # conv for STFT
        if self.params.useSTFT:
            self.stsf = ConvSTFT(**self.stft_params)

        # print('$$$ self.additionalFeatureDim =', self.additionalFeatureDim)
        self.additionalFeatureDim = 0
        if params.useSTFT:
            FreqHisto = params.outputDim_cnn_for_stft
        else:
            FreqHisto = self.spectrumFeatureNums

        if self.params.useFreqHisto:
            self.additionalFeatureDim += FreqHisto
        if self.param.useTime:
            self.additionalFeatureDim += 1

        self.combined_size = self.additionalFeatureDim
        if self.useRaw:
            self.combined_size += self.convNet.outputSize

        self.classifier = ClassifyNet(
            input_size=self.combined_size, **self.classifier_params
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print('$%$%$%$%$% in forward, x.shape = (batchSize, seqLen, channelNum, featureNum) = ', x.shape)
        # normalized = self.batn_first(x)
        # connect two parts

        if self.useLSTM:
            batchSize, subseqLen, channelNum, featureNum = x.shape
        else:
            batchSize, channelNum, featureNum = x.shape
            subseqLen = 1
        # print('in forward(), before reshape: x.shape =', x.shape)
        x = x.reshape(-1, channelNum, featureNum)
        # print('$%$%$%$%$% in forward, after reshape, x.shape = (new_batchSize, channelNum, featureNum) = ', x.shape)

        raw = x[:, :, : self.rawDataDim]
        freq = x[:, :, self.rawDataDim : -1].reshape(
            batchSize, channelNum, self.spectrumFeatureNums, -1
        )
        time = self.flatten(x[:, :, -1])
        combined = None
        if self.useRaw and self.useFreq:
            rawFeature = self.convNet(raw)
            if self.useSTFT:
                freqFeature = self.stft(freq)
            else:
                freqFeature = self.flatten(freq)

            if self.useTime:
                combined = torch.cat((rawFeature, freqFeature, time), dim=1)
            else:
                combined = torch.cat((rawFeature, freqFeature), dim=1)

        elif self.useFreq:
            if self.useSTFT:
                combined = self.stft(freq)
            else:
                combined = self.flatten(freq)

            if self.useTime:
                combined = torch.cat((combined, time), dim=1)
        elif self.useRaw:
            combined = self.convNet(raw)
            if self.useTime:
                combined = torch.cat((combined, time), dim=1)
        else:
            if self.useTime:
                combined = time

        return self.classifier(combined, batchSize, subseqLen)
