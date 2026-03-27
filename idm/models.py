import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class IDM(nn.Module):
    """Stacked-frame IDM: (frame_t || frame_t+1) -> action_t (normalised)"""

    def __init__(self, action_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock(6, 32, stride=2),     # 45x80
            ResBlock(32, 64, stride=2),    # 23x40
            ResBlock(64, 128, stride=2),   # 12x20
            ResBlock(128, 256, stride=2),  #  6x10
            ResBlock(256, 512, stride=1),  #  6x10  deeper before pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x).flatten(1))
