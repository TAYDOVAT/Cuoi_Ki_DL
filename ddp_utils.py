import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed():
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def get_device(local_rank=0):
    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_wrap_ddp(model, local_rank=0):
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            return DDP(model, device_ids=[local_rank])
        return DDP(model)
    return model


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
