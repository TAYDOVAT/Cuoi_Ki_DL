import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + identity


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return self.prelu(x)


class SRResNet(nn.Module):
    def __init__(self, scale=4, num_blocks=16, channels=64):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, channels, 9, 1, 4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

        if scale == 4:
            up_blocks = [UpsampleBlock(channels, 2), UpsampleBlock(channels, 2)]
        elif scale == 2:
            up_blocks = [UpsampleBlock(channels, 2)]
        else:
            raise ValueError('scale must be 2 or 4')

        self.upsampler = nn.Sequential(*up_blocks)
        self.conv3 = nn.Conv2d(channels, 3, 9, 1, 4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1
        out = self.upsampler(out)
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, base_channels, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_ch = base_channels
        for i in range(1, 4):
            out_ch = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 2, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def load_pretrained(generator, weight_path, map_location='cpu'):
    state = torch.load(weight_path, map_location=map_location)
    generator.load_state_dict(state)
    return generator