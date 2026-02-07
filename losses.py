import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        return self.criterion(sr, hr)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, max_val=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.max_val = max_val
        self.register_buffer("_kernel", self._create_kernel(window_size, sigma))

    @staticmethod
    def _create_kernel(window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_1d = g.view(1, 1, 1, -1)
        kernel_2d = kernel_1d.transpose(2, 3) @ kernel_1d
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def _get_kernel(self, channels, device, dtype):
        kernel = self._kernel.to(device=device, dtype=dtype)
        return kernel.repeat(channels, 1, 1, 1)

    def forward(self, sr, hr):
        c1 = (0.01 * self.max_val) ** 2
        c2 = (0.03 * self.max_val) ** 2

        channels = sr.size(1)
        kernel = self._get_kernel(channels, sr.device, sr.dtype)

        mu1 = F.conv2d(sr, kernel, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(hr, kernel, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(sr * sr, kernel, padding=self.window_size // 2, groups=channels)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(hr * hr, kernel, padding=self.window_size // 2, groups=channels)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(sr * hr, kernel, padding=self.window_size // 2, groups=channels)
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
        )
        return 1.0 - ssim_map.mean()


class LPIPSLoss(nn.Module):
    def __init__(self, net="vgg"):
        super().__init__()
        import lpips

        self.model = lpips.LPIPS(net=net)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        # LPIPS expects inputs in [-1, 1]
        sr_norm = sr * 2.0 - 1.0
        hr_norm = hr * 2.0 - 1.0
        return self.model(sr_norm, hr_norm).mean()


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
