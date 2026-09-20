import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, less_downsample=False):
        super().__init__()
        
        # Balanced temporal downsampling (T = W/4)
        pool1 = nn.MaxPool2d((2, 2))
        pool2 = nn.MaxPool2d((2, 2))
        pool3 = nn.MaxPool2d((2, 1))

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            pool1,

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            pool2,

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            pool3,

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)
