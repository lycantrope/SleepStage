import torch.nn as nn


class ClassifyNet(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        dropout,
        use_lstm=False,
        lstm_inputDim=None,
        lstm_hidden_size=None,
        ltsm_num_layers=None,
        bidirectional=False,
    ):
        super().__init__()
        middle_size = num_classes
        self.use_lstm = use_lstm
        if use_lstm:
            middle_size = lstm_inputDim
            self.lstm = nn.LSTM(
                input_size=lstm_inputDim,
                hidden_size=lstm_hidden_size,
                num_layers=ltsm_num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
            num_directions = 2 if bidirectional else 1
            self.final_fc_lstm = nn.Linear(
                lstm_hidden_size * num_directions, num_classes
            )

        self.pre_fc_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(dropout),
            nn.Linear(input_size, middle_size),
        )

    def forward(self, x, batchsize=None, subseqlen=None):
        x = self.pre_fc_layer(x)
        if not self.use_lstm:
            return x
        else:
            x, _ = self.lstm(x.reshape(batchsize, subseqlen, -1))
            return self.final_fc_lstm(x)
