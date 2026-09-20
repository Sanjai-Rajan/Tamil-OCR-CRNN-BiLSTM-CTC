import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, classes):

        super().__init__()

        self.linear = nn.Linear(
            512,
            classes
        )

    def forward(self, x):

        return self.linear(x)
