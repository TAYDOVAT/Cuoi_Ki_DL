import torch
import torch.nn.functional as F


def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_val = 10 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr_val.mean().item()


def _gaussian_kernel(window_size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, 1, -1)
    kernel_2d = kernel_1d.transpose(2, 3) @ kernel_1d
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def ssim(sr, hr, max_val=1.0, window_size=11, sigma=1.5):
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    device = sr.device
    kernel = _gaussian_kernel(window_size, sigma, device=device)
    kernel = kernel.repeat(sr.size(1), 1, 1, 1)

    mu1 = F.conv2d(sr, kernel, padding=window_size // 2, groups=sr.size(1))
    mu2 = F.conv2d(hr, kernel, padding=window_size // 2, groups=hr.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, kernel, padding=window_size // 2, groups=sr.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, kernel, padding=window_size // 2, groups=hr.size(1)) - mu2_sq
    sigma12 = F.conv2d(sr * hr, kernel, padding=window_size // 2, groups=sr.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()