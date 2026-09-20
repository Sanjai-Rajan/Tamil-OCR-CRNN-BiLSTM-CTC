import torch.nn as nn

class EncoderV2(nn.Module):
    def __init__(self, high_vertical_resolution=True):
        super().__init__()
        
        # Original less_downsample logic applied to width
        pool1 = nn.MaxPool2d((2, 1))
        pool2 = nn.MaxPool2d(2) 
        
        # If high_vertical_resolution, do not downsample height in pool3
        # Original: (2, 1) -> H/8, W/2
        # New: (1, 1) -> H/4, W/2 (so for H=32, final H=8)
        pool3 = nn.MaxPool2d((1, 1)) if high_vertical_resolution else nn.MaxPool2d((2, 1))

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            pool1,

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            pool2,

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            pool3,

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)
