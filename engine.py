import torch
import torch.distributed as dist
import os
import csv
from datetime import datetime, timezone
from metrics import psnr, ssim

GAN_LOG_FIELDS = [
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
    "train_loss_adv",
    "val_loss_adv",
    "train_loss_lpips_core",
    "val_loss_lpips_core",
    "noise_std",
]

GAN_HISTORY_FIELDS = [
    "loss_g",
    "loss_d",
    "psnr",
    "ssim",
    "lpips",
    "d_real_prob",
    "d_fake_prob",
    "loss_adv",
    "loss_lpips_core",
]


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
    scaler=None,
    train_config=None,
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
        "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "train_config": train_config,
        "meta": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": str(torch.__version__),
        },
    }
    torch.save(checkpoint, path)


def load_gan_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    scheduler_g=None,
    scheduler_d=None,
    scaler=None,
    path="weights/gan_checkpoint.pth",
    device="cuda",
):
    """
    Load GAN checkpoint and restore all training states.

    Returns:
        start_epoch: Epoch to resume from (1-indexed)
        best_lpips: Best LPIPS achieved so far
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"[Checkpoint] Loading from {path}...")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=device)

    # Load generator (always)
    gen = generator.module if hasattr(generator, "module") else generator
    disc = discriminator.module if hasattr(discriminator, "module") else discriminator

    gen.load_state_dict(checkpoint["generator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if scheduler_g and checkpoint.get("scheduler_g_state_dict"):
        scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
    print("[Checkpoint] Generator weights loaded.")

    disc.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    if scheduler_d and checkpoint.get("scheduler_d_state_dict"):
        scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
    print("[Checkpoint] Discriminator weights loaded.")

    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("[Checkpoint] AMP scaler state loaded.")

    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    best_lpips = checkpoint.get("best_lpips", 100.0)

    print(f"[Checkpoint] Resuming from epoch {start_epoch}, best LPIPS: {best_lpips:.4f}")
    return start_epoch, best_lpips


def _empty_gan_history():
    history = {name: {"train": [], "val": []} for name in GAN_HISTORY_FIELDS}
    history["noise_std"] = []
    return history


def _row_float(row, key, default=0.0):
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _history_value(history, key, split, idx, default=0.0):
    series = history.get(key, {}).get(split, [])
    if idx < len(series):
        return series[idx]
    return default


def _history_scalar_value(history, key, idx, default=0.0):
    series = history.get(key, [])
    if idx < len(series):
        return series[idx]
    return default


def load_gan_history_from_log(log_path, start_epoch):
    """
    Load existing training history from CSV log file.

    Args:
        log_path: Path to gan_log.csv
        start_epoch: Starting epoch (1-indexed), logs before this will be kept

    Returns:
        history dict with lists for each metric
    """
    history = _empty_gan_history()

    if not os.path.exists(log_path) or start_epoch <= 1:
        return history

    try:
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row.get("epoch", 0))
                if epoch < start_epoch:
                    history["loss_g"]["train"].append(
                        _row_float(row, "train_loss_g", 0.0)
                    )
                    history["loss_g"]["val"].append(_row_float(row, "val_loss_g", 0.0))
                    history["loss_d"]["train"].append(
                        _row_float(row, "train_loss_d", 0.0)
                    )
                    history["loss_d"]["val"].append(_row_float(row, "val_loss_d", 0.0))
                    history["d_real_prob"]["train"].append(
                        _row_float(row, "train_d_real_prob", 0.0)
                    )
                    history["d_real_prob"]["val"].append(
                        _row_float(row, "val_d_real_prob", 0.0)
                    )
                    history["d_fake_prob"]["train"].append(
                        _row_float(row, "train_d_fake_prob", 0.0)
                    )
                    history["d_fake_prob"]["val"].append(
                        _row_float(row, "val_d_fake_prob", 0.0)
                    )
                    history["psnr"]["train"].append(_row_float(row, "train_psnr", 0.0))
                    history["psnr"]["val"].append(_row_float(row, "val_psnr", 0.0))
                    history["ssim"]["train"].append(_row_float(row, "train_ssim", 0.0))
                    history["ssim"]["val"].append(_row_float(row, "val_ssim", 0.0))
                    history["lpips"]["train"].append(_row_float(row, "train_lpips", 0.0))
                    history["lpips"]["val"].append(_row_float(row, "val_lpips", 0.0))
                    history["loss_adv"]["train"].append(
                        _row_float(row, "train_loss_adv", 0.0)
                    )
                    history["loss_adv"]["val"].append(
                        _row_float(row, "val_loss_adv", 0.0)
                    )
                    history["loss_lpips_core"]["train"].append(
                        _row_float(row, "train_loss_lpips_core", 0.0)
                    )
                    history["loss_lpips_core"]["val"].append(
                        _row_float(row, "val_loss_lpips_core", 0.0)
                    )
                    history["noise_std"].append(_row_float(row, "noise_std", 0.0))
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
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GAN_LOG_FIELDS)

        num_entries = len(history["loss_g"]["train"])
        for i in range(num_entries):
            writer.writerow(
                [
                    i + 1,  # epoch (1-indexed)
                    _history_value(history, "loss_g", "train", i),
                    _history_value(history, "loss_g", "val", i),
                    _history_value(history, "loss_d", "train", i),
                    _history_value(history, "loss_d", "val", i),
                    _history_value(history, "d_real_prob", "train", i),
                    _history_value(history, "d_real_prob", "val", i),
                    _history_value(history, "d_fake_prob", "train", i),
                    _history_value(history, "d_fake_prob", "val", i),
                    _history_value(history, "psnr", "train", i),
                    _history_value(history, "psnr", "val", i),
                    _history_value(history, "ssim", "train", i),
                    _history_value(history, "ssim", "val", i),
                    _history_value(history, "lpips", "train", i),
                    _history_value(history, "lpips", "val", i),
                    _history_value(history, "loss_adv", "train", i),
                    _history_value(history, "loss_adv", "val", i),
                    _history_value(history, "loss_lpips_core", "train", i),
                    _history_value(history, "loss_lpips_core", "val", i),
                    _history_scalar_value(history, "noise_std", i),
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


def _compute_discriminator_loss(
    d_real_logits,
    d_fake_logits,
    adversarial_criterion,
    real_label=1.0,
    fake_label=0.0,
):
    loss_d_real = adversarial_criterion(d_real_logits, True, real_label)
    loss_d_fake = adversarial_criterion(d_fake_logits, False, fake_label)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    return loss_d, loss_d_real, loss_d_fake


def _scheduled_noise_std(epoch_idx, start_std, end_std, decay_epochs):
    decay_epochs = int(max(decay_epochs, 1))
    if epoch_idx >= decay_epochs:
        return float(end_std)
    if decay_epochs == 1:
        return float(end_std)
    t = float(max(epoch_idx - 1, 0)) / float(decay_epochs - 1)
    return float(start_std) + t * (float(end_std) - float(start_std))


def train_srresnet_epoch(
    model, loader, optimizer, device, pixel_criterion, use_amp=False, scaler=None, lpips_metric=None
):
    model.train()
    total_loss = 0.0
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

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            sr_clip = sr.clamp(0.0, 1.0)
            sr_metric = sr_clip.float()
            hr_metric = hr.float()
            batch_psnr = psnr(sr_metric, hr_metric)
            batch_ssim = ssim(sr_metric, hr_metric)
            if lpips_metric is not None:
                sr_norm = sr_clip * 2.0 - 1.0
                hr_norm = hr * 2.0 - 1.0
                batch_lpips = lpips_metric(sr_norm, hr_norm).mean().item()
            else:
                batch_lpips = 0.0

        batch_size = lr.size(0)
        total_loss += loss.item() * batch_size
        total_psnr += batch_psnr * batch_size
        total_ssim += batch_ssim * batch_size
        total_lpips += batch_lpips * batch_size
        count += batch_size

        _maybe_postfix(loader, loss.item(), total_psnr / max(count, 1))

    total_loss, total_psnr, total_ssim, total_lpips, count = _ddp_reduce_totals(
        [total_loss, total_psnr, total_ssim, total_lpips, float(count)], device
    )

    return {
        "loss": total_loss / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
        "lpips": total_lpips / max(count, 1),
    }


def val_srresnet_epoch(model, loader, device, pixel_criterion, use_amp=False, lpips_metric=None):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
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
            sr_metric = sr_clip.float()
            hr_metric = hr.float()
            batch_psnr = psnr(sr_metric, hr_metric)
            batch_ssim = ssim(sr_metric, hr_metric)
            if lpips_metric is not None:
                sr_norm = sr_clip * 2.0 - 1.0
                hr_norm = hr * 2.0 - 1.0
                batch_lpips = lpips_metric(sr_norm, hr_norm).mean().item()
            else:
                batch_lpips = 0.0

            batch_size = lr.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_lpips += batch_lpips * batch_size
            count += batch_size

            _maybe_postfix(loader, loss.item(), total_psnr / max(count, 1))

    total_loss, total_psnr, total_ssim, total_lpips, count = _ddp_reduce_totals(
        [total_loss, total_psnr, total_ssim, total_lpips, float(count)], device
    )

    return {
        "loss": total_loss / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
        "lpips": total_lpips / max(count, 1),
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
    g_loss_mode="srgan",
    lpips_criterion=None,
    lpips_metric=None,
    epoch_idx=1,
    g_steps=1,
    d_steps=1,
    r1_weight=0.0,
    r1_interval=1,
    d_noise_std_start=0.0,
    d_noise_std_end=0.0,
    d_noise_decay_epochs=1,
    real_label=0.9,
    fake_label=0.0,
    use_amp=False,
    scaler=None,
):
    generator.train()
    discriminator.train()

    if r1_weight > 0.0:
        disc = discriminator.module if hasattr(discriminator, "module") else discriminator
        has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in disc.modules())
        if has_bn:
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                print("[WARN] R1 penalty with BatchNorm can cause autograd errors. Disabling R1 for stability.")
            r1_weight = 0.0

    total_g = 0.0
    total_d = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_adv = 0.0
    total_lpips_core = 0.0
    total_noise_std = 0.0
    count = 0

    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    if scaler is None:
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    if g_steps < 1 or d_steps < 1:
        raise ValueError(f"g_steps and d_steps must be >= 1, got g_steps={g_steps}, d_steps={d_steps}")
    noise_std = _scheduled_noise_std(
        epoch_idx=epoch_idx,
        start_std=d_noise_std_start,
        end_std=d_noise_std_end,
        decay_epochs=d_noise_decay_epochs,
    )
    r1_interval = max(int(r1_interval), 1)
    d_update_idx = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Train D
        loss_d = 0.0
        d_real_prob = 0.0
        d_fake_prob = 0.0
        for _ in range(d_steps):
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    sr = generator(lr)

            d_update_idx += 1
            apply_r1 = r1_weight > 0.0 and (d_update_idx % r1_interval == 0)
            hr_for_d = hr.requires_grad_(True) if apply_r1 else hr

            d_real_input = hr_for_d
            d_fake_input = sr.detach()
            if noise_std > 0.0:
                d_real_input = d_real_input + torch.randn_like(d_real_input) * noise_std
                d_fake_input = d_fake_input + torch.randn_like(d_fake_input) * noise_std

            if apply_r1:
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    d_real = discriminator(d_real_input.float())
                    d_fake = discriminator(d_fake_input.float())
                    loss_d_step, _, _ = _compute_discriminator_loss(
                        d_real,
                        d_fake,
                        adversarial_criterion,
                        real_label=real_label,
                        fake_label=fake_label,
                    )
                    r1_penalty = _r1_penalty(d_real, hr_for_d)
                    loss_d_step = loss_d_step + 0.5 * float(r1_weight) * r1_penalty
            else:
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    d_real = discriminator(d_real_input)
                    d_fake = discriminator(d_fake_input)
                    loss_d_step, _, _ = _compute_discriminator_loss(
                        d_real,
                        d_fake,
                        adversarial_criterion,
                        real_label=real_label,
                        fake_label=fake_label,
                    )

            with torch.no_grad():
                d_real_prob = torch.sigmoid(d_real).mean()
                d_fake_prob = torch.sigmoid(d_fake).mean()

            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d_step).backward()
            scaler.step(optimizer_d)
            scaler.update()
            loss_d = loss_d_step
            if apply_r1:
                hr = hr.detach()

        # Train G
        loss_g = 0.0
        loss_adv_for_log = None
        loss_lpips_core_for_log = None
        for _ in range(g_steps):
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                sr = generator(lr)
                d_fake_for_g = discriminator(sr)
                loss_adv = adversarial_criterion(d_fake_for_g, True)
                if g_loss_mode == "srgan":
                    loss_perc = perceptual_criterion(sr, hr)
                    loss_g_step = (
                        weights["perceptual"] * loss_perc
                        + weights["adversarial"] * loss_adv
                    )
                    loss_lpips_core = torch.zeros((), device=sr.device, dtype=sr.dtype)
                elif g_loss_mode == "lpips_adv":
                    if lpips_criterion is None:
                        raise RuntimeError(
                            "lpips_criterion is required when g_loss_mode='lpips_adv'."
                        )
                    with torch.amp.autocast(device_type=device_type, enabled=False):
                        loss_lpips = lpips_criterion(sr.float(), hr.float())
                    loss_g_step = (
                        weights["lpips"] * loss_lpips
                        + weights["adversarial"] * loss_adv
                    )
                    loss_lpips_core = loss_lpips
                else:
                    raise ValueError(f"Unsupported g_loss_mode: {g_loss_mode}")

            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g_step).backward()
            scaler.step(optimizer_g)
            scaler.update()
            loss_g = loss_g_step
            loss_adv_for_log = loss_adv
            loss_lpips_core_for_log = loss_lpips_core

        with torch.no_grad():
            sr_clip = sr.clamp(0.0, 1.0)
            sr_metric = sr_clip.float()
            hr_metric = hr.float()
            batch_psnr = psnr(sr_metric, hr_metric)
            batch_ssim = ssim(sr_metric, hr_metric)
            
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
        total_adv += loss_adv_for_log.item() * batch_size
        total_lpips_core += loss_lpips_core_for_log.item() * batch_size
        total_noise_std += noise_std * batch_size
        count += batch_size

        if hasattr(loader, "set_postfix"):
            loader.set_postfix(
                {
                    "loss_G": f"{loss_g.item():.4f}",
                    "loss_D": f"{loss_d.item():.4f}",
                    "psnr": f"{total_psnr / max(count, 1):.2f}",
                    "lpips": f"{total_lpips / max(count, 1):.4f}",
                    "adv": f"{total_adv / max(count, 1):.4f}",
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
        total_adv,
        total_lpips_core,
        total_noise_std,
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
            total_adv,
            total_lpips_core,
            total_noise_std,
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
        "loss_adv": total_adv / max(count, 1),
        "loss_lpips_core": total_lpips_core / max(count, 1),
        "noise_std": total_noise_std / max(count, 1),
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
    g_loss_mode="srgan",
    lpips_criterion=None,
    lpips_metric=None,
    real_label=0.9,
    fake_label=0.0,
    val_use_train_labels=True,
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
    total_adv = 0.0
    total_lpips_core = 0.0
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

                d_real_label = real_label if val_use_train_labels else 1.0
                d_fake_label = fake_label if val_use_train_labels else 0.0
                loss_d, _, _ = _compute_discriminator_loss(
                    d_real,
                    d_fake,
                    adversarial_criterion,
                    real_label=d_real_label,
                    fake_label=d_fake_label,
                )

                d_real_prob = torch.sigmoid(d_real).mean()
                d_fake_prob = torch.sigmoid(d_fake).mean()

                loss_adv = adversarial_criterion(d_fake, True)
                if g_loss_mode == "srgan":
                    loss_perc = perceptual_criterion(sr, hr)
                    loss_g = (
                        weights["perceptual"] * loss_perc
                        + weights["adversarial"] * loss_adv
                    )
                    loss_lpips_core = torch.zeros((), device=sr.device, dtype=sr.dtype)
                elif g_loss_mode == "lpips_adv":
                    if lpips_criterion is None:
                        raise RuntimeError(
                            "lpips_criterion is required when g_loss_mode='lpips_adv'."
                        )
                    with torch.amp.autocast(device_type=device_type, enabled=False):
                        loss_lpips = lpips_criterion(sr.float(), hr.float())
                    loss_g = (
                        weights["lpips"] * loss_lpips
                        + weights["adversarial"] * loss_adv
                    )
                    loss_lpips_core = loss_lpips
                else:
                    raise ValueError(f"Unsupported g_loss_mode: {g_loss_mode}")

            sr_clip = sr.clamp(0.0, 1.0)
            sr_metric = sr_clip.float()
            hr_metric = hr.float()
            batch_psnr = psnr(sr_metric, hr_metric)
            batch_ssim = ssim(sr_metric, hr_metric)
            
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
            total_adv += loss_adv.item() * batch_size
            total_lpips_core += loss_lpips_core.item() * batch_size
            count += batch_size

            if hasattr(loader, "set_postfix"):
                loader.set_postfix(
                    {
                        "loss_G": f"{loss_g.item():.4f}",
                        "loss_D": f"{loss_d.item():.4f}",
                        "psnr": f"{total_psnr / max(count, 1):.2f}",
                        "lpips": f"{total_lpips / max(count, 1):.4f}",
                        "adv": f"{total_adv / max(count, 1):.4f}",
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
        total_adv,
        total_lpips_core,
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
            total_adv,
            total_lpips_core,
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
        "loss_adv": total_adv / max(count, 1),
        "loss_lpips_core": total_lpips_core / max(count, 1),
        "noise_std": 0.0,
    }
