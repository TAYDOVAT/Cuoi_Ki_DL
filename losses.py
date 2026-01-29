import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        return self.criterion(sr, hr)


class PerceptualLoss(nn.Module):
    def __init__(self, layer=35):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:layer]).eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_n = (sr - self.mean) / self.std
        hr_n = (hr - self.mean) / self.std
        sr_f = self.features(sr_n)
        hr_f = self.features(hr_n)
        return self.criterion(sr_f, hr_f)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, target_is_real, target_value=None):
        if target_value is None:
            target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        else:
            target = torch.full_like(logits, float(target_value))
        return self.criterion(logits, target)
