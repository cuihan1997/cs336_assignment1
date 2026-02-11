from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


from tokenizers import ByteLevelBPETokenizer

Merge = Tuple[bytes, bytes]


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {x!r}")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    *,
    show_progress: bool = True,
) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[input_path],
        vocab_size=int(vocab_size),
        special_tokens=list(special_tokens or []),
        show_progress=bool(show_progress),
    )
    return tok


def save_vocab_and_merges(out_dir: str, tokenizer: ByteLevelBPETokenizer) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out))  # creates vocab.json + merges.txt
    print(f"Saved merges to: {out / 'merges.txt'}")
    print(f"Saved vocab  to: {out / 'vocab.json'}")


def find_longest_token_from_vocab_json(vocab_path: str | Path) -> tuple[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: Dict[str, int] = json.load(f)
    best_tok = ""
    best_id = -1
    for t, i in token_to_id.items():
        if len(t) > len(best_tok) or (len(t) == len(best_tok) and (best_id == -1 or int(i) < best_id)):
            best_tok = t
            best_id = int(i)
    return best_tok, best_id


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")

    parser.add_argument("--input-path", type=str, required=True, help="Path to training corpus (.txt)")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Final vocabulary size")
    parser.add_argument("--out-dir", type=str, default="./bpe_out", help="Directory to save vocab.json and merges.txt")
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens to add to vocab. Example: --special-tokens "<|endoftext|>" "<|pad|>"',
    )


    parser.add_argument(
        "--print-longest-token",
        type=_str2bool,
        default=False,
        help="If true, print the longest vocab token details after training (true/false).",
    )

    parser.add_argument(
        "--progress",
        type=_str2bool,
        default=True,
        help="Show progress bars (true/false).",
    )

    args = parser.parse_args()

    tokenizer = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        show_progress=args.progress,
    )

    save_vocab_and_merges(args.out_dir, tokenizer)

    if args.print_longest_token:
        vocab_path = Path(args.out_dir) / "vocab.json"
        tok, tid = find_longest_token_from_vocab_json(vocab_path)
        b = tok.encode("utf-8", errors="replace")
        print(f"Longest token: id={tid}, str_len={len(tok)}, utf8_bytes_len={len(b)}")
        print(f"Longest token details: token={tok!r} utf8_bytes={b!r}")

    print("Done.")


if __name__ == "__main__":
    main()




"""
python3.13 bpe_hf.py \
  --input-path ./data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --out-dir ./bpe_out/TinyStories \
  --special-tokens "<|endoftext|>" \
  --print-longest-token true\
  --progress true
"""






"""
python3.13 bpe_hf.py \
  --input-path ./data/owt_train.txt \
  --vocab-size 32000 \
  --out-dir ./bpe_out/owt \
  --special-tokens "<|endoftext|>" \
  --print-longest-token true\
  --progress true
"""