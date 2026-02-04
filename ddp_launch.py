import os
import csv
import random
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from ddp_utils import (
    init_distributed,
    get_device,
    maybe_wrap_ddp,
    is_main_process,
    cleanup_distributed,
)
from data import build_loader
from original_model import SRResNet, DiscriminatorForVGG
from losses import PixelLoss, PerceptualLoss, AdversarialLoss
from engine import (
    train_gan_epoch,
    val_gan_epoch,
    save_gan_checkpoint,
    load_gan_checkpoint,
    load_gan_history_from_log,
    rewrite_log_up_to_epoch,
)
import lpips


def _empty_history():
    return {
        "loss_g": {"train": [], "val": []},
        "loss_d": {"train": [], "val": []},
        "psnr": {"train": [], "val": []},
        "ssim": {"train": [], "val": []},
        "lpips": {"train": [], "val": []},
        "d_real_prob": {"train": [], "val": []},
        "d_fake_prob": {"train": [], "val": []},
    }


def _load_state_flexible(model, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and sd and next(iter(sd)).startswith("module."):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=True)


def _ddp_worker(rank, world_size, cfg):
    is_ddp, local_rank = init_distributed(
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT','29500')}",
    )
    device = get_device(local_rank)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_dataset, train_loader = build_loader(
        cfg["paths"]["train_lr"],
        cfg["paths"]["train_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=cfg["gan"]["batch_size"],
        num_workers=cfg["gan"]["num_workers"],
        train=True,
    )
    val_batch_size = max(8, world_size * 4)
    val_batch_size = (val_batch_size // world_size) * world_size
    val_dataset, val_loader = build_loader(
        cfg["paths"]["val_lr"],
        cfg["paths"]["val_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=val_batch_size,
        num_workers=cfg["gan"]["num_workers"],
        train=False,
    )

    generator = SRResNet(upscale=cfg["scale"]).to(device)
    discriminator = DiscriminatorForVGG().to(device)
    generator = maybe_wrap_ddp(generator, local_rank)
    discriminator = maybe_wrap_ddp(discriminator, local_rank)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg["gan"]["lr_g"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["gan"]["lr_d"])
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g, step_size=10000, gamma=0.5
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optimizer_d, step_size=10000, gamma=0.5
    )

    pixel_criterion = PixelLoss().to(device)
    perceptual_criterion = PerceptualLoss().to(device)
    adversarial_criterion = AdversarialLoss().to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)

    weights = {
        "pixel": cfg["gan"]["pixel_weight"],
        "perceptual": cfg["gan"]["perc_weight"],
        "adversarial": cfg["gan"]["adv_weight"],
    }

    use_amp = cfg["gan"].get("use_amp", False) and torch.cuda.is_available()
    device_type = device.type if hasattr(device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    log_path = os.path.join("logs", "gan_log.csv")
    if cfg["gan"]["resume"]:
        start_epoch, best_lpips = load_gan_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            path=cfg["gan"]["checkpoint_path"],
            load_disc=cfg["gan"]["load_disc"],
            device=device,
        )
        if is_main_process():
            history = load_gan_history_from_log(log_path, start_epoch)
            rewrite_log_up_to_epoch(log_path, history, start_epoch)
        else:
            history = _empty_history()
    else:
        start_epoch = 1
        best_lpips = 100.0
        _load_state_flexible(generator, "weights/best_srresnet.pth", device)
        if is_main_process():
            print("[INFO] Loaded Generator from 'weights/best_srresnet.pth'")
            print("[INFO] Initialized fresh Discriminator")
        history = _empty_history()
        if is_main_process():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
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
                )

    if is_main_process():
        print("\n" + "=" * 50)
        print(f"Starting from epoch {start_epoch}, best LPIPS: {best_lpips:.4f}")
        print(
            f"Resume: {cfg['gan']['resume']}, Load Disc: {cfg['gan']['load_disc']}"
        )
        print("=" * 50)

    epochs = cfg["gan"]["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader, "sampler") and hasattr(
            train_loader.sampler, "set_epoch"
        ):
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
            lpips_metric=lpips_metric,
            g_steps=cfg["gan"].get("g_steps", 1),
            d_steps=cfg["gan"].get("d_steps", 1),
            r1_weight=cfg["gan"].get("r1_weight", 0.0),
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
            lpips_metric=lpips_metric,
            use_amp=use_amp,
        )

        scheduler_g.step()
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
                    ]
                )

            save_gan_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                epoch=epoch,
                best_lpips=best_lpips,
                path=cfg["gan"]["checkpoint_path"],
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

            print(
                f"Epoch {epoch}/{epochs} | LR_G: {scheduler_g.get_last_lr()[0]:.6f} | "
                f"LR_D: {scheduler_d.get_last_lr()[0]:.6f}"
            )
            print(f"Best LPIPS: {best_lpips:.4f}")

    if is_main_process():
        print("\n" + "=" * 50)
        print("GAN Training Completed!")
        print(f"Best LPIPS: {best_lpips:.4f}")
        print("=" * 50)

    if is_ddp:
        cleanup_distributed()


def launch_ddp(cfg):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("DDP spawn requires at least 2 GPUs.")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["DISABLE_DP"] = "1"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    world_size = torch.cuda.device_count()
    mp.spawn(_ddp_worker, args=(world_size, cfg), nprocs=world_size, join=True)
