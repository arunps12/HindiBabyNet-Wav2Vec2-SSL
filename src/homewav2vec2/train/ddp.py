"""DDP / distributed training helpers."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

log = logging.getLogger("homewav2vec2.ddp")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def global_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return global_rank() == 0


def setup_distributed() -> None:
    """Initialize the process group for DDP (torchrun sets env vars)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank())
        log.info(
            "DDP initialized: rank=%d/%d  local_rank=%d",
            global_rank(), world_size(), local_rank(),
        )
    else:
        log.info("Running single-process (no DDP)")


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
