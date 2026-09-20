import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=256,
            bidirectional=True
        )

        self.fc = nn.Linear(512, 128)

    def forward(self, x):

        x = self.cnn(x)

        return x
