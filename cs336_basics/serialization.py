import os
import typing
from typing import Optional, Any, Dict

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    *,
    config: Optional[Dict[str, Any]] = None,
):
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "iteration": int(iteration),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
) -> int:
    checkpoint = torch.load(src, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint.get("iteration", 0))
