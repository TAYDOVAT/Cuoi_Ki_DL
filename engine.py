import torch
from metrics import psnr, ssim


def _maybe_postfix(loader, loss_val, psnr_val):
    if hasattr(loader, 'set_postfix'):
        loader.set_postfix({'loss': f'{loss_val:.4f}', 'psnr': f'{psnr_val:.2f}'})


def train_srresnet_epoch(model, loader, optimizer, device, pixel_criterion):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)
        loss = pixel_criterion(sr, hr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)

        batch_size = lr.size(0)
        total_loss += loss.item() * batch_size
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        count += batch_size

        _maybe_postfix(loader, loss.item(), total_psnr / max(count, 1))

    return {
        'loss': total_loss / max(count, 1),
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }


def val_srresnet_epoch(model, loader, device, pixel_criterion):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            sr = model(lr)
            loss = pixel_criterion(sr, hr)

            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)

            batch_size = lr.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            count += batch_size

            _maybe_postfix(loader, loss.item(), total_psnr / max(count, 1))

    return {
        'loss': total_loss / max(count, 1),
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }


def train_gan_epoch(generator, discriminator, loader, optimizer_g, optimizer_d, device,
                    pixel_criterion, perceptual_criterion, adversarial_criterion,
                    weights):
    generator.train()
    discriminator.train()

    total_g = 0.0
    total_d = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Train D
        with torch.no_grad():
            sr = generator(lr)
        d_real = discriminator(hr)
        d_fake = discriminator(sr.detach())
        loss_d_real = adversarial_criterion(d_real, True)
        loss_d_fake = adversarial_criterion(d_fake, False)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()

        # Train G
        sr = generator(lr)
        d_fake_for_g = discriminator(sr)
        loss_pixel = pixel_criterion(sr, hr)
        loss_perc = perceptual_criterion(sr, hr)
        loss_adv = adversarial_criterion(d_fake_for_g, True)
        loss_g = (weights['pixel'] * loss_pixel +
                  weights['perceptual'] * loss_perc +
                  weights['adversarial'] * loss_adv)

        optimizer_g.zero_grad(set_to_none=True)
        loss_g.backward()
        optimizer_g.step()

        with torch.no_grad():
            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)

        batch_size = lr.size(0)
        total_g += loss_g.item() * batch_size
        total_d += loss_d.item() * batch_size
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        count += batch_size

        if hasattr(loader, 'set_postfix'):
            loader.set_postfix({
                'loss_G': f'{loss_g.item():.4f}',
                'loss_D': f'{loss_d.item():.4f}',
                'psnr': f'{total_psnr / max(count, 1):.2f}',
            })

    return {
        'loss_g': total_g / max(count, 1),
        'loss_d': total_d / max(count, 1),
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }


def val_gan_epoch(generator, discriminator, loader, device,
                  pixel_criterion, perceptual_criterion, adversarial_criterion,
                  weights):
    generator.eval()
    discriminator.eval()

    total_g = 0.0
    total_d = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            sr = generator(lr)
            d_real = discriminator(hr)
            d_fake = discriminator(sr)

            loss_d_real = adversarial_criterion(d_real, True)
            loss_d_fake = adversarial_criterion(d_fake, False)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            loss_pixel = pixel_criterion(sr, hr)
            loss_perc = perceptual_criterion(sr, hr)
            loss_adv = adversarial_criterion(d_fake, True)
            loss_g = (weights['pixel'] * loss_pixel +
                      weights['perceptual'] * loss_perc +
                      weights['adversarial'] * loss_adv)

            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)

            batch_size = lr.size(0)
            total_g += loss_g.item() * batch_size
            total_d += loss_d.item() * batch_size
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            count += batch_size

            if hasattr(loader, 'set_postfix'):
                loader.set_postfix({
                    'loss_G': f'{loss_g.item():.4f}',
                    'loss_D': f'{loss_d.item():.4f}',
                    'psnr': f'{total_psnr / max(count, 1):.2f}',
                })

    return {
        'loss_g': total_g / max(count, 1),
        'loss_d': total_d / max(count, 1),
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }