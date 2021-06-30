import torch.nn as nn


class ClassifyNet(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        dropout,
        useLSTM,
        lstm_params,
    ):
        super().__init__()
        middle_size = num_classes
        self.useLSTM = useLSTM
        if useLSTM:
            middle_size = lstm_params["input_size"]
            hidden_size = lstm_params["hidden_size"]
            bidirectional = lstm_params["bidirectional"]
            self.lstm = nn.LSTM(
                **lstm_params,
                dropout=dropout,
            )
            num_directions = 2 if bidirectional else 1
            self.final_fc_lstm = nn.Linear(hidden_size * num_directions, num_classes)

        self.pre_fc_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(dropout),
            nn.Linear(input_size, middle_size),
        )

    def forward(self, x, batchsize=None, subseqlen=None):
        x = self.pre_fc_layer(x)
        if not self.useLSTM:
            return x
        else:
            x, _ = self.lstm(x.reshape(batchsize, subseqlen, -1))
            return self.final_fc_lstm(x)
