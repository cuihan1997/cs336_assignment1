from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from config import TrainingConfig
from data import get_batch
from model import TransformerLM
from optimizer import AdamW, learning_rate_schedule
from serialization import save_checkpoint, load_checkpoint
from utils import cross_entropy, gradient_clipping


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {x!r}")


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Training configuration.")

    # Data
    p.add_argument(
        "--train_data_path",
        type=Path,
        required=True,
        help="Path to 1D token-id binary file for training (memmap).",
    )
    p.add_argument(
        "--val_data_path",
        type=Path,
        default=None,
        help="Path to 1D token-id binary file for validation (memmap).",
    )
    p.add_argument(
        "--data_dtype",
        type=str,
        default="uint16",
        help="Numpy dtype for memmap files (e.g., uint16, int32).",
    )
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)

    # Model
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimization
    p.add_argument("--max_iters", type=int, default=50_000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--cosine_cycle_iters", type=int, default=50_000)

    p.add_argument("--betas1", type=float, default=0.9)
    p.add_argument("--betas2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Logging / eval / ckpt
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    p.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to a checkpoint .pt to resume from.",
    )

    # System
    p.add_argument("--seed", type=int, default=31415926)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_wandb", type=_str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="lm-training")
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (user or team). If None, use default logged-in entity.",
    )
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()
    return TrainingConfig(**vars(args))


def open_memmap_1d(path: Path, dtype: str) -> np.memmap:
    dt = np.dtype(dtype)
    n_items = os.path.getsize(path) // dt.itemsize
    if n_items <= 0:
        raise ValueError(f"Empty dataset file: {path}")
    if os.path.getsize(path) % dt.itemsize != 0:
        raise ValueError(
            f"File size is not divisible by dtype itemsize ({dt.itemsize}). "
            f"Got file_size={os.path.getsize(path)} for {path}."
        )
    return np.memmap(path, dtype=dt, mode="r", shape=(n_items,))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    *,
    device: str,
    batch_size: int,
    context_length: int,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(
            dataset, batch_size=batch_size, context_len=context_length, device=device
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    cfg = parse_args()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # wandb
    wandb = None
    if cfg.use_wandb:
        try:
            import wandb as _wandb  

            wandb = _wandb
            init_kwargs = dict(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg.to_dict(),
            )

            if cfg.wandb_entity is not None:
                init_kwargs["entity"] = cfg.wandb_entity

            wandb.init(**init_kwargs)
        except Exception as e:
            print(
                f"[warn] wandb requested but unavailable/failed to init: {e}. Continuing without wandb."
            )
            wandb = None

    train_data = open_memmap_1d(cfg.train_data_path, cfg.data_dtype)
    val_data = (
        open_memmap_1d(cfg.val_data_path, cfg.data_dtype)
        if cfg.val_data_path is not None
        else None
    )

    # Model
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        device=device,
        dtype=torch.float32,
    ).to(device)

    # Optimizer
    optim = AdamW(
        model.parameters(),
        lr=cfg.max_lr,  # will be overridden each iter by schedule
        betas=(cfg.betas1, cfg.betas2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Resume
    start_iter = 0
    if cfg.resume_from is not None:
        start_iter = load_checkpoint(cfg.resume_from, model=model, optimizer=optim)
        print(f"[info] Resumed from {cfg.resume_from} at iteration {start_iter}.")

    # Checkpoint dir
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    t0 = time.time()
    running_loss = 0.0
    for it in range(start_iter, cfg.max_iters):
        lr = learning_rate_schedule(
            it=it,
            max_learning_rate=cfg.max_lr,
            min_learning_rate=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad(set_to_none=True)

        # Gradient accumulation
        micro_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(
                train_data,
                batch_size=cfg.batch_size,
                context_len=cfg.context_length,
                device=str(device),
            )
            logits = model(x)
            loss = cross_entropy(logits, y) / cfg.grad_accum_steps
            loss.backward()
            micro_loss += loss.item()

        # Clip and step
        if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
            gradient_clipping(model.parameters(), cfg.max_grad_norm)

        optim.step()

        running_loss += micro_loss

        # Logging
        if (it + 1) % cfg.log_every == 0:
            dt = time.time() - t0
            avg_loss = running_loss / cfg.log_every
            tokens_per_iter = cfg.batch_size * cfg.context_length * cfg.grad_accum_steps
            tok_per_sec = tokens_per_iter * cfg.log_every / max(dt, 1e-9)

            msg = (
                f"iter {it+1:>7d}/{cfg.max_iters} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.3e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            print(msg)
            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "lr": lr,
                        "iter": it + 1,
                        "throughput_tok_s": tok_per_sec,
                    }
                )
            running_loss = 0.0
            t0 = time.time()

        # Eval
        if val_data is not None and (it + 1) % cfg.eval_every == 0:
            val_loss = estimate_loss(
                model,
                val_data,
                device=str(device),
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                eval_batches=cfg.eval_batches,
            )
            print(f"           val_loss {val_loss:.4f}")
            if wandb is not None:
                wandb.log({"val/loss": val_loss, "iter": it + 1})

        # Checkpoint
        if (it + 1) % cfg.ckpt_every == 0 or (it + 1) == cfg.max_iters:
            ckpt_path = cfg.ckpt_dir / f"ckpt_iter_{it+1}.pt"
            save_checkpoint(
                model=model,
                optimizer=optim,
                iteration=it + 1,
                out=str(ckpt_path),
                config=cfg.to_dict(),
            )
            print(f"[info] Saved checkpoint: {ckpt_path}")
            if wandb is not None:
                wandb.log({"ckpt/iter": it + 1})

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()




"""
python lm_training.py \
  --train_data_path ./data_bin/TinyStories/train.bin \
  --val_data_path ./data_bin/TinyStories/valid.bin \
  --data_dtype uint16 \
  --context_length 256 \
  --batch_size 32 \
  --vocab_size 10000 \
  --d_model 512 \
  --d_ff 1344 \
  --num_heads 16 \
  --num_layers 4 \
  --rope_theta 10000 \
  --max_iters 40000 \
  --max_grad_norm 1.0 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 200 \
  --cosine_cycle_iters 40000 \
  --betas1 0.9 \
  --betas2 0.95 \
  --eps 1e-8 \
  --weight_decay 0.1 \
  --log_every 100 \
  --eval_every 1000 \
  --eval_batches 50 \
  --ckpt_every 5000 \
  --ckpt_dir ./checkpoints/tinystories \
  --seed 31415926 \
  --device cuda\
  --use_wandb true\
  --wandb_entity hancui-cs336 \
  --wandb_project tinystories \
  --wandb_run_name tinystories_run1
"""




"""
python lm_training.py \
  --train_data_path ./data_bin/owt/train.bin \
  --val_data_path ./data_bin/owt/valid.bin \
  --data_dtype uint16 \
  --context_length 512 \
  --batch_size 64 \
  --vocab_size 32000 \
  --d_model 768 \
  --d_ff 2048 \
  --num_heads 12 \
  --num_layers 8 \
  --rope_theta 10000 \
  --max_iters 9000 \
  --grad_accum_steps 1\
  --max_grad_norm 1.0 \
  --max_lr 8e-4 \
  --min_lr 8e-5 \
  --warmup_iters 400 \
  --cosine_cycle_iters 9000 \
  --betas1 0.9 \
  --betas2 0.95 \
  --eps 1e-8 \
  --weight_decay 0.1 \
  --log_every 50 \
  --eval_every 500 \
  --eval_batches 10 \
  --ckpt_every 2000 \
  --ckpt_dir ./checkpoints/owt \
  --seed 31415926 \
  --device cuda \
  --use_wandb true \
  --wandb_entity hancui-cs336 \
  --wandb_project owt \
  --wandb_run_name owt_run1
"""