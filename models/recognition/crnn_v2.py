import torch
import torch.nn as nn

from models.recognition.encoder_v2 import EncoderV2
from models.recognition.lstm import SequenceModel
from models.recognition.head import PredictionHead

class CRNNV2(nn.Module):
    def __init__(self, num_classes, feature_size=4096, high_vertical_resolution=True):
        super().__init__()

        self.encoder = EncoderV2(high_vertical_resolution=high_vertical_resolution)
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
