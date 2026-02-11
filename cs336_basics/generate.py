from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Any, Dict

import torch

from model import TransformerLM
from bpe_tokenizer_hf import BPETokenizer


@torch.no_grad()
def top_p_sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D (vocab_size,), got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1].")

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=probs.device))
        cutoff = int(cutoff.item())

        keep_probs = sorted_probs[: cutoff + 1]
        keep_idx = sorted_idx[: cutoff + 1]

        keep_probs = keep_probs / keep_probs.sum()
        sampled_in_keep = torch.multinomial(keep_probs, num_samples=1).item()
        return int(keep_idx[sampled_in_keep].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: List[int],
    *,
    max_new_tokens: int,
    end_token_id: Optional[int],
    temperature: float = 1.0,
    top_p: float = 1.0,
    context_length: int,
    device: torch.device,
) -> List[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    model.eval()
    ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        window = ids[-context_length:] if len(ids) > context_length else ids
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        logits = model(x)               # (1, T, vocab)
        next_logits = logits[0, -1, :]  # (vocab,)

        next_id = top_p_sample(next_logits, temperature=temperature, top_p=top_p)
        ids.append(next_id)

        if end_token_id is not None and next_id == end_token_id:
            break

    return ids


def _get_cfg_value(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if key in cfg:
        return cfg[key]
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict) and key in model_cfg:
        return model_cfg[key]
    return default


def _resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    if device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_built()):
            print("[warn] MPS requested but this PyTorch build has no MPS support; falling back to CPU.")
            return torch.device("cpu")
        if not torch.backends.mps.is_available():
            print("[warn] MPS requested but not available on this machine; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"Unknown --device {device_str!r}. Use one of: cpu, cuda, mps, auto.")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint.")

    p.add_argument("--ckpt_path", type=Path, required=True)
    p.add_argument("--vocab_json", type=Path, required=True)
    p.add_argument("--merges_txt", type=Path, required=True)

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--end_token", type=str, default="<|endoftext|>")

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: cpu, cuda, mps, or auto.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=31415926,
        help="Random seed for sampling (torch RNG). Same seed + same prompt + same ckpt => same output.",
    )

    # -------------------------------------------------------------------------
    # If ckpt contains a saved config, load hparams from ckpt.
    # -------------------------------------------------------------------------
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    args = p.parse_args()

    device = _resolve_device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)


    tokenizer = BPETokenizer.from_files(
        str(args.vocab_json),
        str(args.merges_txt),
        special_tokens=[args.end_token] if args.end_token else [],
    )
    end_token_id = tokenizer.encode(args.end_token)[0] if args.end_token else None


    ckpt = torch.load(str(args.ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict produced by save_checkpoint().")

    cfg: Dict[str, Any] = {}
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]
        print("[info] Loaded model hyperparameters from checkpoint config.")
    else:
        print("[warn] Checkpoint has no 'config'. Falling back to CLI model hyperparameters.")

    vocab_size = int(_get_cfg_value(cfg, "vocab_size", args.vocab_size))
    context_length = int(_get_cfg_value(cfg, "context_length", args.context_length))
    d_model = int(_get_cfg_value(cfg, "d_model", args.d_model))
    num_heads = int(_get_cfg_value(cfg, "num_heads", args.num_heads))
    d_ff = int(_get_cfg_value(cfg, "d_ff", args.d_ff))
    num_layers = int(_get_cfg_value(cfg, "num_layers", args.num_layers))
    rope_theta = float(_get_cfg_value(cfg, "rope_theta", args.rope_theta))

    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        device=torch.device("cpu"),   
        dtype=torch.float32,
    )

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint is missing 'model_state_dict'.")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model = model.to(device)  

    prompt_ids = tokenizer.encode(args.prompt)
    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        end_token_id=end_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=context_length,
        device=device,
    )

    completion_ids = out_ids[len(prompt_ids):]
    if end_token_id is not None and completion_ids and completion_ids[-1] == end_token_id:
        completion_ids = completion_ids[:-1]

    print(tokenizer.decode(prompt_ids + completion_ids))


if __name__ == "__main__":
    main()



"""
Example:
python generate.py \
  --ckpt_path ./checkpoints/tinystories/ckpt_iter_40000.pt \
  --vocab_json ./bpe_out/TinyStories/vocab.json \
  --merges_txt ./bpe_out/TinyStories/merges.txt \
  --prompt "Once upon a time" \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.9 \
  --device "mps"\
  --seed 31415926
"""




"""
python generate.py \
  --ckpt_path ./checkpoints/owt/ckpt_iter_9000.pt \
  --vocab_json ./bpe_out/owt/vocab.json \
  --merges_txt ./bpe_out/owt/merges.txt \
  --prompt "Over the past decade, researchers have discovered" \
  --max_new_tokens 512 \
  --temperature 0.8 \
  --top_p 0.9 \
  --device "mps"\
  --seed 31415926
"""
