import torch
import torch.nn as nn

from models.recognition.encoder import Encoder
from models.recognition.lstm import SequenceModel
from models.recognition.head import PredictionHead


class CRNN(nn.Module):

    def __init__(self, num_classes, feature_size=8192, less_downsample=False):

        super().__init__()

        self.encoder = Encoder(less_downsample=less_downsample)

        self.sequence = SequenceModel(input_size=feature_size)

        self.head = PredictionHead(num_classes)

    def forward(self, x):

        x = self.encoder(x)

        batch, channel, height, width = x.size()

        x = x.permute(0, 3, 1, 2)

        x = x.reshape(
            batch,
            width,
            channel * height
        )

        x = self.sequence(x)

        x = self.head(x)

        return x
