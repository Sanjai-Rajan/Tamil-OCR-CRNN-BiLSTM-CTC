import torch.nn as nn

criterion = nn.CTCLoss(
    blank=0,
    reduction="mean",
    zero_infinity=True,
)
