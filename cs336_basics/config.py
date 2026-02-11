from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    train_data_path: Path
    val_data_path: Optional[Path] = None
    data_dtype: str = "uint16"            
    context_length: int = 256
    batch_size: int = 64

    # Model
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1344
    num_layers: int = 8
    rope_theta: float = 10000.0

    # Optimization
    max_iters: int = 50_000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    cosine_cycle_iters: int = 50_000

    betas1: float = 0.9
    betas2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Logging / eval / ckpt
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 1000
    ckpt_dir: Path = Path("checkpoints")
    resume_from: Optional[Path] = None

    # System
    seed: int = 1337
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "lm-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list):
                d[k] = [str(x) if isinstance(x, Path) else x for x in v]
        return d
