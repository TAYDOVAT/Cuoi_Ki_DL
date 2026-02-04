import torch
import torch.distributed as dist
import os
import csv
from metrics import psnr, ssim


# ==================== Checkpoint Functions ====================


def save_gan_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    epoch,
    best_lpips,
    path="weights/gan_checkpoint.pth",
):
    """Save GAN checkpoint with all training states."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    gen = generator.module if hasattr(generator, "module") else generator
    disc = discriminator.module if hasattr(discriminator, "module") else discriminator

    checkpoint = {
        "epoch": epoch,
        "best_lpips": best_lpips,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict() if scheduler_g else None,
        "scheduler_d_state_dict": scheduler_d.state_dict() if scheduler_d else None,
    }
    torch.save(checkpoint, path)


def load_gan_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    scheduler_g=None,
    scheduler_d=None,
    path="weights/gan_checkpoint.pth",
    load_disc=True,
    device="cuda",
):
    """
    Load GAN checkpoint and restore all training states.

    Args:
        load_disc: If True, load discriminator weights. If False, only load generator.

    Returns:
        start_epoch: Epoch to resume from (1-indexed)
        best_psnr: Best PSNR achieved so far
    """
    if not os.path.exists(path):
        print(f"[Checkpoint] Not found at {path}. Starting from scratch.")
        return 1, 100.0

    print(f"[Checkpoint] Loading from {path}...")
    checkpoint = torch.load(path, map_location=device)

    # Load generator (always)
    gen = generator.module if hasattr(generator, "module") else generator
    disc = discriminator.module if hasattr(discriminator, "module") else discriminator

    gen.load_state_dict(checkpoint["generator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if scheduler_g and checkpoint.get("scheduler_g_state_dict"):
        scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
    print("[Checkpoint] Generator weights loaded.")

    # Load discriminator (optional)
    if load_disc:
        disc.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if scheduler_d and checkpoint.get("scheduler_d_state_dict"):
            scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
        print("[Checkpoint] Discriminator weights loaded.")
    else:
        print("[Checkpoint] Discriminator weights SKIPPED (load_disc=False).")

    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    best_lpips = checkpoint.get("best_lpips", 100.0)

    print(f"[Checkpoint] Resuming from epoch {start_epoch}, best LPIPS: {best_lpips:.4f}")
    return start_epoch, best_lpips


def load_gan_history_from_log(log_path, start_epoch):
    """
    Load existing training history from CSV log file.

    Args:
        log_path: Path to gan_log.csv
        start_epoch: Starting epoch (1-indexed), logs before this will be kept

    Returns:
        history dict with lists for each metric
    """
    history = {
        "loss_g": {"train": [], "val": []},
        "loss_d": {"train": [], "val": []},
        "psnr": {"train": [], "val": []},
        "ssim": {"train": [], "val": []},
        "lpips": {"train": [], "val": []},
        "d_real_prob": {"train": [], "val": []},
        "d_fake_prob": {"train": [], "val": []},
    }

    if not os.path.exists(log_path) or start_epoch <= 1:
        return history

    try:
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row["epoch"])
                if epoch < start_epoch:
                    history["loss_g"]["train"].append(float(row["train_loss_g"]))
                    history["loss_g"]["val"].append(float(row["val_loss_g"]))
                    history["loss_d"]["train"].append(float(row["train_loss_d"]))
                    history["loss_d"]["val"].append(float(row["val_loss_d"]))
                    history["d_real_prob"]["train"].append(
                        float(row["train_d_real_prob"])
                    )
                    history["d_real_prob"]["val"].append(float(row["val_d_real_prob"]))
                    history["d_fake_prob"]["train"].append(
                        float(row["train_d_fake_prob"])
                    )
                    history["d_fake_prob"]["val"].append(float(row["val_d_fake_prob"]))
                    history["psnr"]["train"].append(float(row["train_psnr"]))
                    history["psnr"]["val"].append(float(row["val_psnr"]))
                    history["ssim"]["train"].append(float(row["train_ssim"]))
                    history["ssim"]["val"].append(float(row["val_ssim"]))
                    if "train_lpips" in row and "val_lpips" in row:
                        history["lpips"]["train"].append(float(row["train_lpips"]))
                        history["lpips"]["val"].append(float(row["val_lpips"]))
                    else:
                        history["lpips"]["train"].append(0.0)
                        history["lpips"]["val"].append(0.0)
        print(
            f"[Log] Loaded {len(history['loss_g']['train'])} previous epochs from {log_path}"
        )
    except Exception as e:
        print(f"[Log] Error loading history: {e}. Starting fresh.")

    return history


def rewrite_log_up_to_epoch(log_path, history, start_epoch):
    """
    Rewrite CSV log file with only epochs before start_epoch.
    This ensures clean resume without duplicate entries.
    """
    expected_header = [
        "epoch",
        "train_loss_g",
        "val_loss_g",
        "train_loss_d",
        "val_loss_d",
        "train_d_real_prob",
        "val_d_real_prob",
        "train_d_fake_prob",
        "val_d_fake_prob",
        "train_psnr",
        "val_psnr",
        "train_ssim",
        "val_ssim",
        "train_lpips",
        "val_lpips",
    ]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(expected_header)

        num_entries = len(history["loss_g"]["train"])
        for i in range(num_entries):
            writer.writerow(
                [
                    i + 1,  # epoch (1-indexed)
                    history["loss_g"]["train"][i],
                    history["loss_g"]["val"][i],
                    history["loss_d"]["train"][i],
                    history["loss_d"]["val"][i],
                    history["d_real_prob"]["train"][i],
                    history["d_real_prob"]["val"][i],
                    history["d_fake_prob"]["train"][i],
                    history["d_fake_prob"]["val"][i],
                    history["psnr"]["train"][i],
                    history["psnr"]["val"][i],
                    history["ssim"]["train"][i],
                    history["ssim"]["val"][i],
                    history["lpips"]["train"][i],
                    history["lpips"]["val"][i],
                ]
            )


# ==================== Training Functions ====================


def _is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _maybe_postfix(loader, loss_val, psnr_val):
    if _is_main_process() and hasattr(loader, "set_postfix"):
        loader.set_postfix({"loss": f"{loss_val:.4f}", "psnr": f"{psnr_val:.2f}"})


def _ddp_reduce_totals(totals, device):
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor(totals, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()
    return totals


def train_srresnet_epoch(model, loader, optimizer, device, pixel_criterion, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    if scaler is None:
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    total_loss, total_psnr, total_ssim, count = _ddp_reduce_totals(
        [total_loss, total_psnr, total_ssim, float(count)], device
    )

    return {
        "loss": total_loss / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
    }


def val_srresnet_epoch(model, loader, device, pixel_criterion, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
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

    total_loss, total_psnr, total_ssim, count = _ddp_reduce_totals(
        [total_loss, total_psnr, total_ssim, float(count)], device
    )

    return {
        "loss": total_loss / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
    }


def _r1_penalty(d_real, real_imgs):
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).flatten(1).sum(1).mean()


def train_gan_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    device,
    pixel_criterion,
    perceptual_criterion,
    adversarial_criterion,
    weights,
    lpips_metric=None,
    g_steps=2,
    d_steps=1,
    r1_weight=0.0,
    use_amp=False,
    scaler=None,
):
    generator.train()
    discriminator.train()

    total_g = 0.0
    total_d = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    if scaler is None:
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Train D
        real_label = 0.9
        fake_label = 0.0
        loss_d = 0.0
        d_real_prob = 0.0
        d_fake_prob = 0.0
        for _ in range(d_steps):
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    sr = generator(lr)
            # Add Gaussian Noise to prevent D from overfitting
            noise_std = 0.05
            if r1_weight > 0.0:
                hr.requires_grad_(True)
            
            # Apply noise to inputs for D
            d_real_input = hr + torch.randn_like(hr) * noise_std
            d_fake_input = sr.detach() + torch.randn_like(sr) * noise_std
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                d_real = discriminator(d_real_input)
                d_fake = discriminator(d_fake_input)
                loss_d_real = adversarial_criterion(d_real, True, real_label)
                loss_d_fake = adversarial_criterion(d_fake, False, fake_label)
                loss_d_step = 0.5 * (loss_d_real + loss_d_fake)

            if r1_weight > 0.0:
                # Compute R1 penalty in full precision for stability
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    d_real_fp32 = discriminator(d_real_input.float())
                    r1_penalty = _r1_penalty(d_real_fp32, hr)
                    loss_d_step = loss_d_step + 0.5 * r1_weight * r1_penalty

            with torch.no_grad():
                d_real_prob = torch.sigmoid(d_real).mean()
                d_fake_prob = torch.sigmoid(d_fake).mean()

            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d_step).backward()
            scaler.step(optimizer_d)
            scaler.update()
            loss_d = loss_d_step
            if r1_weight > 0.0:
                hr = hr.detach()

        # Train G
        loss_g = 0.0
        for _ in range(g_steps):
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                sr = generator(lr)
                d_fake_for_g = discriminator(sr)
                loss_pixel = pixel_criterion(sr, hr)
                loss_perc = perceptual_criterion(sr, hr)
                loss_adv = adversarial_criterion(d_fake_for_g, True)
                loss_g_step = (
                    weights["pixel"] * loss_pixel
                    + weights["perceptual"] * loss_perc
                    + weights["adversarial"] * loss_adv
                )

            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g_step).backward()
            scaler.step(optimizer_g)
            scaler.update()
            loss_g = loss_g_step

        with torch.no_grad():
            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)
            
            if lpips_metric is not None:
                # Inputs to LPIPS should be [-1, 1]
                sr_norm = sr_clip * 2.0 - 1.0
                hr_norm = hr * 2.0 - 1.0
                batch_lpips = lpips_metric(sr_norm, hr_norm).mean().item()
            else:
                batch_lpips = 0.0

        batch_size = lr.size(0)
        total_g += loss_g.item() * batch_size
        total_d += loss_d.item() * batch_size
        total_d_real += d_real_prob.item() * batch_size
        total_d_fake += d_fake_prob.item() * batch_size
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        total_lpips += batch_lpips * batch_size
        count += batch_size

        if hasattr(loader, "set_postfix"):
            loader.set_postfix(
                {
                    "loss_G": f"{loss_g.item():.4f}",
                    "loss_D": f"{loss_d.item():.4f}",
                    "psnr": f"{total_psnr / max(count, 1):.2f}",
                    "lpips": f"{total_lpips / max(count, 1):.4f}",
                }
            )

    (
        total_g,
        total_d,
        total_d_real,
        total_d_fake,
        total_psnr,
        total_ssim,
        total_lpips,
        count,
    ) = _ddp_reduce_totals(
        [
            total_g,
            total_d,
            total_d_real,
            total_d_fake,
            total_psnr,
            total_ssim,
            total_lpips,
            float(count),
        ],
        device,
    )

    return {
        "loss_g": total_g / max(count, 1),
        "loss_d": total_d / max(count, 1),
        "d_real_prob": total_d_real / max(count, 1),
        "d_fake_prob": total_d_fake / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
        "lpips": total_lpips / max(count, 1),
    }


def val_gan_epoch(
    generator,
    discriminator,
    loader,
    device,
    pixel_criterion,
    perceptual_criterion,
    adversarial_criterion,
    weights,
    lpips_metric=None,
    use_amp=False,
):
    generator.eval()
    discriminator.eval()

    total_g = 0.0
    total_d = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        torch.cuda.empty_cache()
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                sr = generator(lr)
                d_real = discriminator(hr)
                d_fake = discriminator(sr)

                loss_d_real = adversarial_criterion(d_real, True)
                loss_d_fake = adversarial_criterion(d_fake, False)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

                d_real_prob = torch.sigmoid(d_real).mean()
                d_fake_prob = torch.sigmoid(d_fake).mean()

                loss_pixel = pixel_criterion(sr, hr)
                loss_perc = perceptual_criterion(sr, hr)
                loss_adv = adversarial_criterion(d_fake, True)
                loss_g = (
                    weights["pixel"] * loss_pixel
                    + weights["perceptual"] * loss_perc
                    + weights["adversarial"] * loss_adv
                )

            sr_clip = sr.clamp(0.0, 1.0)
            batch_psnr = psnr(sr_clip, hr)
            batch_ssim = ssim(sr_clip, hr)
            
            if lpips_metric is not None:
                sr_norm = sr_clip * 2.0 - 1.0
                hr_norm = hr * 2.0 - 1.0
                batch_lpips = lpips_metric(sr_norm, hr_norm).mean().item()
            else:
                batch_lpips = 0.0

            batch_size = lr.size(0)
            total_g += loss_g.item() * batch_size
            total_d += loss_d.item() * batch_size
            total_d_real += d_real_prob.item() * batch_size
            total_d_fake += d_fake_prob.item() * batch_size
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_lpips += batch_lpips * batch_size
            count += batch_size

            if hasattr(loader, "set_postfix"):
                loader.set_postfix(
                    {
                        "loss_G": f"{loss_g.item():.4f}",
                        "loss_D": f"{loss_d.item():.4f}",
                        "psnr": f"{total_psnr / max(count, 1):.2f}",
                        "lpips": f"{total_lpips / max(count, 1):.4f}",
                    }
                )

    (
        total_g,
        total_d,
        total_d_real,
        total_d_fake,
        total_psnr,
        total_ssim,
        total_lpips,
        count,
    ) = _ddp_reduce_totals(
        [
            total_g,
            total_d,
            total_d_real,
            total_d_fake,
            total_psnr,
            total_ssim,
            total_lpips,
            float(count),
        ],
        device,
    )

    return {
        "loss_g": total_g / max(count, 1),
        "loss_d": total_d / max(count, 1),
        "d_real_prob": total_d_real / max(count, 1),
        "d_fake_prob": total_d_fake / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
        "lpips": total_lpips / max(count, 1),
    }
