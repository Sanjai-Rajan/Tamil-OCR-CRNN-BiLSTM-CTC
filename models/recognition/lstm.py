import torch.nn as nn


class SequenceModel(nn.Module):

    def __init__(self, input_size=8192):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        return output
