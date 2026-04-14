"""Minimal distributed helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext(False, rank=0, world_size=1, local_rank=0)

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return DistributedContext(True, rank=rank, world_size=world_size, local_rank=local_rank)


def destroy_distributed(context: DistributedContext) -> None:
    if not context.distributed:
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
