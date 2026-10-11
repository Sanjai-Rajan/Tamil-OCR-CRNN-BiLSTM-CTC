"""
CRNN v2: residual CNN + 2-layer BiLSTM + CTC, trained from scratch.

Changes from the old model, each addressing an observed problem:
  * input height 48 (was 32): Tamil vowel signs above/below the letter keep more pixels
  * residual blocks + dropout (old model: plain convs, no dropout, heavy overfitting)
  * height collapsed by a learned (3x1) conv, giving a compact 384-d feature per column
    (old model flattened 512x4 = 2048 features straight into the LSTM)
  * packed sequences: the BiLSTM ignores padding (old model read ~45% white padding,
    and the backward LSTM started inside it)
  * horizontal stride 4, as before (W/4 timesteps): the capacity diagnostic showed this is ample
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BasicBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = None if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + (x if self.short is None else self.short(x)))


class CRNNv2(nn.Module):
    STRIDE_W = 4

    def __init__(self, num_classes: int, in_h: int = 48, channels=(64, 128, 256, 384),
                 lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            BasicBlock(32, c1), nn.MaxPool2d(2, 2),                      # H/2,  W/2
            BasicBlock(c1, c2), BasicBlock(c2, c2), nn.MaxPool2d(2, 2),   # H/4,  W/4
            BasicBlock(c2, c3), BasicBlock(c3, c3), nn.MaxPool2d((2, 1)), # H/8
            BasicBlock(c3, c4), BasicBlock(c4, c4), nn.MaxPool2d((2, 1)), # H/16 (=3 for 48)
        )
        h_out = in_h // 16
        self.collapse = nn.Sequential(nn.Conv2d(c4, c4, (h_out, 1), bias=False), nn.BatchNorm2d(c4),
                                      nn.ReLU(inplace=True))
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(c4, lstm_hidden, num_layers=lstm_layers, bidirectional=True,
                           dropout=dropout if lstm_layers > 1 else 0.0)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)
        self.in_h = in_h

    @classmethod
    def output_lengths(cls, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, cls.STRIDE_W, rounding_mode="floor").clamp(min=1)

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None):
        """x: B x 1 x H x W  ->  (logits T x B x C, output lengths B)"""
        f = self.collapse(self.cnn(x))              # B x C x 1 x T
        f = f.squeeze(2).permute(2, 0, 1)           # T x B x C
        f = self.drop(f)
        T = f.shape[0]
        if widths is None:
            lengths = torch.full((x.shape[0],), T, dtype=torch.long)
        else:
            lengths = self.output_lengths(widths.cpu()).clamp(max=T)
        packed = pack_padded_sequence(f, lengths.cpu(), enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, total_length=T)
        return self.fc(self.drop(out)), lengths


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
