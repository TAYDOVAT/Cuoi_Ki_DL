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
from original_model import SRResNet
from losses import PixelLoss, SSIMLoss, LPIPSLoss
import lpips
from engine import (
    train_srresnet_epoch,
    val_srresnet_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DDP SRResNet training (torchrun)")
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


def load_state_flexible(model, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if isinstance(sd, dict) and sd:
        for prefix in ("module.", "_orig_mod."):
            if all(k.startswith(prefix) for k in sd.keys()):
                sd = {k[len(prefix) :]: v for k, v in sd.items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=True)


def format_path(template, loss_name):
    if "{loss}" in template:
        return template.format(loss=loss_name)
    return template


def empty_history():
    return {
        "loss": {"train": [], "val": []},
        "psnr": {"train": [], "val": []},
        "ssim": {"train": [], "val": []},
        "lpips": {"train": [], "val": []},
    }


def save_srresnet_checkpoint(model, optimizer, scheduler, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, path)


def load_srresnet_checkpoint(
    model,
    optimizer,
    scheduler=None,
    path="weights/srresnet_checkpoint.pth",
    device="cuda",
):
    if not os.path.exists(path):
        print(f"[Checkpoint] Not found at {path}. Starting from scratch.")
        return 1

    print(f"[Checkpoint] Loading from {path}...")
    checkpoint = torch.load(path, map_location=device)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"[Checkpoint] Resuming from epoch {start_epoch}")
    return start_epoch


def load_srresnet_history_from_log(log_path, start_epoch):
    history = empty_history()
    if not os.path.exists(log_path) or start_epoch <= 1:
        return history

    try:
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row["epoch"])
                if epoch < start_epoch:
                    history["loss"]["train"].append(float(row["train_loss"]))
                    history["loss"]["val"].append(float(row["val_loss"]))
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
        print(f"[Log] Loaded {len(history['loss']['train'])} previous epochs from {log_path}")
    except Exception as e:
        print(f"[Log] Error loading history: {e}. Starting fresh.")

    return history


def rewrite_log_up_to_epoch(log_path, history, start_epoch):
    expected_header = [
        "epoch",
        "train_loss",
        "val_loss",
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
        num_entries = len(history["loss"]["train"])
        for i in range(num_entries):
            writer.writerow(
                [
                    i + 1,
                    history["loss"]["train"][i],
                    history["loss"]["val"][i],
                    history["psnr"]["train"][i],
                    history["psnr"]["val"][i],
                    history["ssim"]["train"][i],
                    history["ssim"]["val"][i],
                    history["lpips"]["train"][i],
                    history["lpips"]["val"][i],
                ]
            )


def build_loss(loss_name):
    if loss_name == "l1":
        return PixelLoss()
    if loss_name == "ssim":
        return SSIMLoss()
    if loss_name == "lpips":
        return LPIPSLoss(net="vgg")
    raise ValueError(f"Unsupported loss: {loss_name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = Path(__file__).resolve().parent
    cfg = resolve_paths(cfg, base_dir)
    pretrained_path = cfg.get("train", {}).get("pretrained_path")
    if pretrained_path:
        p = Path(pretrained_path)
        cfg["train"]["pretrained_path"] = str(p if p.is_absolute() else (base_dir / p).resolve())

    loss_name = cfg.get("train", {}).get("loss", "l1").lower()

    local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs("weights", exist_ok=True)
    epoch_weights_dir = os.path.join("weights", "srresnet", loss_name)
    os.makedirs(epoch_weights_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_dataset, train_loader = build_loader(
        cfg["paths"]["train_lr"],
        cfg["paths"]["train_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        train=True,
    )

    val_batch_size = cfg["train"].get("val_batch_size", 32)
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        val_batch_size = max(val_batch_size, world_size)
        val_batch_size = (val_batch_size // world_size) * world_size
        if val_batch_size == 0:
            val_batch_size = world_size

    val_dataset, val_loader = build_loader(
        cfg["paths"]["val_lr"],
        cfg["paths"]["val_hr"],
        scale=cfg["scale"],
        hr_crop=cfg["hr_crop"],
        batch_size=val_batch_size,
        num_workers=cfg["train"]["num_workers"],
        train=False,
    )

    model = SRResNet(upscale=cfg["scale"]).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"].get("lr_step", 50),
        gamma=cfg["train"].get("lr_gamma", 0.5),
    )

    criterion = build_loss(loss_name).to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)

    use_amp = cfg["train"].get("use_amp", False) and torch.cuda.is_available()
    device_type = "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    log_path = os.path.join("logs", f"srresnet_{loss_name}_log.csv")
    checkpoint_path = format_path(
        cfg["train"].get("checkpoint_path", "weights/srresnet_{loss}_checkpoint.pth"),
        loss_name,
    )

    if cfg["train"].get("resume", False):
        start_epoch = load_srresnet_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=checkpoint_path,
            device=device,
        )
        if is_main_process():
            history = load_srresnet_history_from_log(log_path, start_epoch)
            rewrite_log_up_to_epoch(log_path, history, start_epoch)
        else:
            history = empty_history()
    else:
        start_epoch = 1
        history = empty_history()
        if is_main_process():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_psnr",
                        "val_psnr",
                        "train_ssim",
                        "val_ssim",
                        "train_lpips",
                        "val_lpips",
                    ]
                )

        if cfg["train"].get("load_pretrained_model", False):
            pre_path = cfg["train"].get("pretrained_path")
            if pre_path:
                load_state_flexible(model, pre_path, device)
                if is_main_process():
                    print(f"[INFO] Loaded pretrained SRResNet from '{pre_path}'")

    if is_main_process():
        print("\n" + "=" * 50)
        print(f"Loss: {loss_name}")
        print(f"Starting from epoch {start_epoch}")
        print(f"Resume: {cfg['train'].get('resume', False)}")
        print("=" * 50)

    epochs = cfg["train"]["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_pbar = (
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            if is_main_process()
            else train_loader
        )
        train_stats = train_srresnet_epoch(
            model,
            train_pbar,
            optimizer,
            device,
            criterion,
            use_amp=use_amp,
            scaler=scaler,
            lpips_metric=lpips_metric,
        )

        val_pbar = (
            tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            if is_main_process()
            else val_loader
        )
        val_stats = val_srresnet_epoch(
            model,
            val_pbar,
            device,
            criterion,
            use_amp=use_amp,
            lpips_metric=lpips_metric,
        )

        scheduler.step()

        if is_main_process():
            history["loss"]["train"].append(train_stats["loss"])
            history["loss"]["val"].append(val_stats["loss"])
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
                        train_stats["loss"],
                        val_stats["loss"],
                        train_stats["psnr"],
                        val_stats["psnr"],
                        train_stats["ssim"],
                        val_stats["ssim"],
                        train_stats["lpips"],
                        val_stats["lpips"],
                    ]
                )

            save_srresnet_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path=checkpoint_path,
            )
            save_srresnet_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path=os.path.join(
                    epoch_weights_dir,
                    f"srresnet_{loss_name}_checkpoint_epoch_{epoch}.pth",
                ),
            )

            torch.save(
                (model.module if hasattr(model, "module") else model).state_dict(),
                f"weights/last_srresnet_{loss_name}.pth",
            )
            torch.save(
                (model.module if hasattr(model, "module") else model).state_dict(),
                os.path.join(
                    epoch_weights_dir,
                    f"srresnet_{loss_name}_epoch_{epoch}.pth",
                ),
            )

            print(
                f"Epoch {epoch}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    if is_main_process():
        print("\n" + "=" * 50)
        print("SRResNet Training Completed!")
        print("=" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
