import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from data import build_loader
from original_model import SRResNet, DiscriminatorForVGG
from losses import PixelLoss, PerceptualLoss, AdversarialLoss, LPIPSLoss
from engine import (
    GAN_LOG_FIELDS,
    train_gan_epoch,
    val_gan_epoch,
    save_gan_checkpoint,
    load_gan_checkpoint,
    load_gan_history_from_log,
    rewrite_log_up_to_epoch,
)
try:
    import lpips
except ModuleNotFoundError:
    lpips = None


def parse_args():
    parser = argparse.ArgumentParser(description="DDP SRGAN training (torchrun)")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    return parser.parse_args()


def is_main_process():
    return dist.get_rank() == 0


def init_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_paths(cfg, base_dir):
    paths = cfg.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        p = Path(value)
        resolved[key] = str(p if p.is_absolute() else (base_dir / p).resolve())
    cfg["paths"] = resolved
    return cfg


def empty_history():
    return {
        "loss_g": {"train": [], "val": []},
        "loss_d": {"train": [], "val": []},
        "psnr": {"train": [], "val": []},
        "ssim": {"train": [], "val": []},
        "lpips": {"train": [], "val": []},
        "d_real_prob": {"train": [], "val": []},
        "d_fake_prob": {"train": [], "val": []},
        "loss_adv": {"train": [], "val": []},
        "loss_lpips_core": {"train": [], "val": []},
        "noise_std": [],
    }


def load_state_flexible(model, path, device):
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and sd and next(iter(sd)).startswith("module."):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=True)


def build_scheduler(optimizer, gan_cfg):
    scheduler_type = str(gan_cfg.get("scheduler_type", "multistep")).lower()
    gamma = float(gan_cfg.get("gamma", 0.5))

    if scheduler_type == "multistep":
        milestones = gan_cfg.get("milestones", [60, 90])
        if not isinstance(milestones, list) or not milestones:
            raise ValueError(
                f"gan.milestones must be a non-empty list for multistep scheduler, got {milestones}"
            )
        milestones = sorted(int(m) for m in milestones)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if scheduler_type == "step":
        step_size = int(gan_cfg.get("lr_step", 30))
        if step_size <= 0:
            raise ValueError(f"gan.lr_step must be > 0, got {step_size}")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_type in {"none", "constant"}:
        return None

    raise ValueError(
        f"Unsupported gan.scheduler_type='{scheduler_type}'. Expected one of ['multistep', 'step', 'none', 'constant']"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = Path(__file__).resolve().parent
    cfg = resolve_paths(cfg, base_dir)
    gan_cfg = cfg.get("gan", {})

    def _resolve_gan_path(key):
        val = gan_cfg.get(key)
        if not val:
            return
        p = Path(val)
        gan_cfg[key] = str(p if p.is_absolute() else (base_dir / p).resolve())

    _resolve_gan_path("init_gen_path")
    _resolve_gan_path("checkpoint_path")

    local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    world_size = dist.get_world_size()
    train_batch_size = cfg["gan"].get("train_batch_size", cfg["gan"]["batch_size"])
    if train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be > 0, got {train_batch_size}")

    raw_val_batch_size = cfg["gan"].get("val_batch_size", None)
    if raw_val_batch_size is None:
        val_batch_size = max(8, world_size * 4)
    else:
        if raw_val_batch_size <= 0:
            raise ValueError(f"val_batch_size must be > 0, got {raw_val_batch_size}")
        val_batch_size = int(raw_val_batch_size)
    val_batch_size = max(world_size, val_batch_size)
    val_batch_size = ((val_batch_size + world_size - 1) // world_size) * world_size

    real_label = float(cfg["gan"].get("real_label", 0.9))
    fake_label = float(cfg["gan"].get("fake_label", 0.0))
    if not (0.0 <= real_label <= 1.0 and 0.0 <= fake_label <= 1.0):
        raise ValueError(
            f"real_label and fake_label must be in [0, 1], got {real_label}, {fake_label}"
        )
    val_use_train_labels = bool(cfg["gan"].get("val_use_train_labels", True))
    r1_interval = int(cfg["gan"].get("r1_interval", 16))
    if r1_interval <= 0:
        raise ValueError(f"r1_interval must be > 0, got {r1_interval}")
    d_noise_std_start = float(cfg["gan"].get("d_noise_std_start", 0.02))
    d_noise_std_end = float(cfg["gan"].get("d_noise_std_end", 0.0))
    d_noise_decay_epochs = int(cfg["gan"].get("d_noise_decay_epochs", 40))
    if d_noise_decay_epochs <= 0:
        raise ValueError(f"d_noise_decay_epochs must be > 0, got {d_noise_decay_epochs}")
    if d_noise_std_start < 0.0 or d_noise_std_end < 0.0:
        raise ValueError(
            f"d_noise_std_start and d_noise_std_end must be >= 0, got {d_noise_std_start}, {d_noise_std_end}"
        )
    g_loss_mode = str(cfg["gan"].get("g_loss_mode", "srgan")).lower()
    valid_g_loss_modes = {"srgan", "lpips_adv"}
    if g_loss_mode not in valid_g_loss_modes:
        raise ValueError(
            f"Unsupported gan.g_loss_mode='{g_loss_mode}'. "
            f"Expected one of {sorted(valid_g_loss_modes)}"
        )

    train_dataset, train_loader = build_loader(
        cfg["paths"]["train_lr"],
        cfg["paths"]["train_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=train_batch_size,
        num_workers=cfg["gan"]["num_workers"],
        train=True,
        pin_memory=cfg["gan"].get("pin_memory", True),
        persistent_workers=cfg["gan"].get("persistent_workers", True),
    )
    val_dataset, val_loader = build_loader(
        cfg["paths"]["val_lr"],
        cfg["paths"]["val_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=val_batch_size,
        num_workers=cfg["gan"]["num_workers"],
        train=False,
        pin_memory=cfg["gan"].get("pin_memory", True),
        persistent_workers=cfg["gan"].get("persistent_workers", True),
    )

    generator = SRResNet(upscale=cfg["scale"]).to(device)
    discriminator = DiscriminatorForVGG().to(device)
    generator = torch.nn.parallel.DistributedDataParallel(
        generator, device_ids=[local_rank], output_device=local_rank
    )
    discriminator = torch.nn.parallel.DistributedDataParallel(
        discriminator, device_ids=[local_rank], output_device=local_rank
    )

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg["gan"]["lr_g"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg["gan"]["lr_d"])
    scheduler_g = build_scheduler(optimizer_g, cfg["gan"])
    scheduler_d = build_scheduler(optimizer_d, cfg["gan"])

    pixel_criterion = PixelLoss().to(device)
    perceptual_criterion = PerceptualLoss().to(device)
    adversarial_criterion = AdversarialLoss().to(device)
    lpips_criterion = None
    if g_loss_mode == "lpips_adv":
        try:
            lpips_criterion = LPIPSLoss(net="vgg").to(device)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "gan.g_loss_mode='lpips_adv' requires the 'lpips' package. "
                "Install it first (e.g. `pip install lpips`)."
            ) from e

    use_lpips = cfg["gan"].get("use_lpips", True)
    lpips_metric = None
    if use_lpips:
        if lpips is None:
            if is_main_process():
                print("[WARN] 'lpips' package not found. LPIPS metric will be disabled.")
        else:
            lpips_metric = lpips.LPIPS(net="vgg").to(device)

    weights = {
        "pixel": cfg["gan"]["pixel_weight"],
        "perceptual": cfg["gan"]["perc_weight"],
        "lpips": cfg["gan"].get("lpips_weight", 1.0),
        "adversarial": cfg["gan"]["adv_weight"],
    }

    use_amp = cfg["gan"].get("use_amp", False) and torch.cuda.is_available()
    device_type = "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    def write_log_header(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(GAN_LOG_FIELDS)

    log_path = os.path.join("logs", "gan_log.csv")
    if cfg["gan"]["resume"]:
        start_epoch, best_lpips = load_gan_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            scaler=scaler,
            path=cfg["gan"]["checkpoint_path"],
            device=device,
        )
        if is_main_process():
            history = load_gan_history_from_log(log_path, start_epoch)
            rewrite_log_up_to_epoch(log_path, history, start_epoch)
        else:
            history = empty_history()
    else:
        start_epoch = 1
        best_lpips = 100.0
        init_path = cfg["gan"].get("init_gen_path") or "weights/best_srresnet.pth"
        load_state_flexible(generator, init_path, device)
        if is_main_process():
            print(f"[INFO] Loaded Generator from '{init_path}'")
            print("[INFO] Initialized fresh Discriminator")
        history = empty_history()
        if is_main_process():
            write_log_header(log_path)

    if is_main_process():
        print("\n" + "=" * 50)
        print(f"Starting from epoch {start_epoch}, best LPIPS: {best_lpips:.4f}")
        print(f"Resume: {cfg['gan']['resume']}")
        print(
            f"Train batch size: {train_batch_size} | "
            f"Val batch size (effective): {val_batch_size}"
        )
        print(
            f"G loss mode: {g_loss_mode} | "
            f"weights(perceptual={weights['perceptual']}, "
            f"lpips={weights['lpips']}, adversarial={weights['adversarial']})"
        )
        print(
            f"Labels(train): real={real_label}, fake={fake_label} | "
            f"Val uses train labels: {val_use_train_labels}"
        )
        print(
            f"D noise schedule: start={d_noise_std_start}, end={d_noise_std_end}, "
            f"decay_epochs={d_noise_decay_epochs}"
        )
        print(
            f"R1: weight={cfg['gan'].get('r1_weight', 0.0)}, interval={r1_interval} | "
            f"G:D steps={cfg['gan'].get('g_steps', 1)}:{cfg['gan'].get('d_steps', 1)}"
        )
        print(f"Scheduler: {cfg['gan'].get('scheduler_type', 'multistep')}")
        print("=" * 50)

    epochs = cfg["gan"]["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            if is_main_process()
            else train_loader
        )
        train_stats = train_gan_epoch(
            generator,
            discriminator,
            train_pbar,
            optimizer_g,
            optimizer_d,
            device,
            pixel_criterion,
            perceptual_criterion,
            adversarial_criterion,
            weights,
            g_loss_mode=g_loss_mode,
            lpips_criterion=lpips_criterion,
            lpips_metric=lpips_metric,
            epoch_idx=epoch,
            g_steps=cfg["gan"].get("g_steps", 1),
            d_steps=cfg["gan"].get("d_steps", 1),
            r1_weight=cfg["gan"].get("r1_weight", 0.0),
            r1_interval=r1_interval,
            d_noise_std_start=d_noise_std_start,
            d_noise_std_end=d_noise_std_end,
            d_noise_decay_epochs=d_noise_decay_epochs,
            real_label=real_label,
            fake_label=fake_label,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_pbar = (
            tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            if is_main_process()
            else val_loader
        )
        val_stats = val_gan_epoch(
            generator,
            discriminator,
            val_pbar,
            device,
            pixel_criterion,
            perceptual_criterion,
            adversarial_criterion,
            weights,
            g_loss_mode=g_loss_mode,
            lpips_criterion=lpips_criterion,
            lpips_metric=lpips_metric,
            real_label=real_label,
            fake_label=fake_label,
            val_use_train_labels=val_use_train_labels,
            use_amp=use_amp,
        )

        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        if is_main_process():
            history["loss_g"]["train"].append(train_stats["loss_g"])
            history["loss_g"]["val"].append(val_stats["loss_g"])
            history["loss_d"]["train"].append(train_stats["loss_d"])
            history["loss_d"]["val"].append(val_stats["loss_d"])
            history["d_real_prob"]["train"].append(train_stats["d_real_prob"])
            history["d_real_prob"]["val"].append(val_stats["d_real_prob"])
            history["d_fake_prob"]["train"].append(train_stats["d_fake_prob"])
            history["d_fake_prob"]["val"].append(val_stats["d_fake_prob"])
            history["psnr"]["train"].append(train_stats["psnr"])
            history["psnr"]["val"].append(val_stats["psnr"])
            history["ssim"]["train"].append(train_stats["ssim"])
            history["ssim"]["val"].append(val_stats["ssim"])
            history["lpips"]["train"].append(train_stats["lpips"])
            history["lpips"]["val"].append(val_stats["lpips"])
            history["loss_adv"]["train"].append(train_stats["loss_adv"])
            history["loss_adv"]["val"].append(val_stats["loss_adv"])
            history["loss_lpips_core"]["train"].append(train_stats["loss_lpips_core"])
            history["loss_lpips_core"]["val"].append(val_stats["loss_lpips_core"])
            history["noise_std"].append(train_stats["noise_std"])

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        train_stats["loss_g"],
                        val_stats["loss_g"],
                        train_stats["loss_d"],
                        val_stats["loss_d"],
                        train_stats["d_real_prob"],
                        val_stats["d_real_prob"],
                        train_stats["d_fake_prob"],
                        val_stats["d_fake_prob"],
                        train_stats["psnr"],
                        val_stats["psnr"],
                        train_stats["ssim"],
                        val_stats["ssim"],
                        train_stats["lpips"],
                        val_stats["lpips"],
                        train_stats["loss_adv"],
                        val_stats["loss_adv"],
                        train_stats["loss_lpips_core"],
                        val_stats["loss_lpips_core"],
                        train_stats["noise_std"],
                    ]
                )

            if val_stats["lpips"] < best_lpips:
                best_lpips = val_stats["lpips"]
                gen_to_save = (
                    generator.module if hasattr(generator, "module") else generator
                )
                disc_to_save = (
                    discriminator.module
                    if hasattr(discriminator, "module")
                    else discriminator
                )
                torch.save(gen_to_save.state_dict(), "weights/best_gan.pth")
                torch.save(disc_to_save.state_dict(), "weights/best_disc.pth")
                print(f"[NEW BEST] LPIPS: {best_lpips:.4f}")

            epoch_dir = os.path.join("weights", f"srgan_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_gen_path = os.path.join(epoch_dir, f"gen_{epoch}.pth")
            epoch_ckpt_path = os.path.join(epoch_dir, f"checkpoint_srgan_{epoch}.pth")

            gen_to_save = generator.module if hasattr(generator, "module") else generator
            torch.save(gen_to_save.state_dict(), epoch_gen_path)
            save_gan_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                epoch=epoch,
                best_lpips=best_lpips,
                scaler=scaler,
                train_config=cfg,
                path=epoch_ckpt_path,
            )

            lr_g = optimizer_g.param_groups[0]["lr"]
            lr_d = optimizer_d.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{epochs} | LR_G: {lr_g:.6f} | LR_D: {lr_d:.6f}"
            )
            print(f"Best LPIPS: {best_lpips:.4f}")

    if is_main_process():
        print("\n" + "=" * 50)
        print("GAN Training Completed!")
        print(f"Best LPIPS: {best_lpips:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
