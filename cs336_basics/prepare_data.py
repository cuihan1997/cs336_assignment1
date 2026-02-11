from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from bpe_tokenizer_hf import BPETokenizer


def infer_num_tokens_from_file(bin_path: str, dtype: np.dtype) -> int:
    size = os.path.getsize(bin_path)
    if size % dtype.itemsize != 0:
        raise ValueError(
            f"File size {size} is not divisible by dtype itemsize {dtype.itemsize}. "
            f"bin_path={bin_path}"
        )
    return size // dtype.itemsize


def _dtype_from_arg(dtype_str: str) -> tuple[np.dtype, str]:
    s = dtype_str.strip().lower()
    if s == "uint16":
        return np.dtype("<u2"), "uint16"
    if s in {"int32_le", "int32"}:
        return np.dtype("<i4"), "int32_le"
    raise ValueError(f"Unsupported --dtype {dtype_str!r}. Use 'uint16' or 'int32_le'.")


def _max_token_id_from_vocab_json(vocab_path: str | Path) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    # token_to_id: Dict[str, int]
    return max(int(i) for i in token_to_id.values()) if token_to_id else -1


def write_tokens_bin_with_progress(
    tokenizer: BPETokenizer,
    input_path: str,
    out_bin_path: str,
    dtype: np.dtype,
    chunk_tokens: int,
    *,
    check_uint16_range: bool,
    show_token_rate: bool = True,
) -> int:
    out_path = Path(out_bin_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_size_bytes = os.path.getsize(input_path)
    total_tokens = 0
    buf: List[int] = []

    max_id_seen = -1

    with open(input_path, "r", encoding="utf-8") as f_in, open(out_path, "wb") as f_out:
        pbar = tqdm(
            total=file_size_bytes,
            unit="B",
            unit_scale=True,
            desc="Encoding",
            dynamic_ncols=True,
            smoothing=0.1,
        )

        try:
            for line in f_in:
                pbar.update(len(line.encode("utf-8")))

                ids = tokenizer.encode(line)

                if check_uint16_range and ids:
                    local_max = max(ids)
                    if local_max > max_id_seen:
                        max_id_seen = local_max

                buf.extend(ids)

                if len(buf) >= chunk_tokens:
                    if check_uint16_range and max_id_seen > 65535:
                        raise ValueError(
                            f"uint16 overflow: saw token id {max_id_seen} (> 65535). "
                            "Fix by using --dtype int32_le OR reduce vocab size / ensure ids fit uint16."
                        )

                    arr = np.asarray(buf, dtype=dtype)
                    arr.tofile(f_out)
                    total_tokens += len(buf)
                    buf.clear()

                    if show_token_rate:
                        pbar.set_postfix(tokens=total_tokens)

            if buf:
                if check_uint16_range and max_id_seen > 65535:
                    raise ValueError(
                        f"uint16 overflow: saw token id {max_id_seen} (> 65535). "
                        "Fix by using --dtype int32_le OR reduce vocab size / ensure ids fit uint16."
                    )

                arr = np.asarray(buf, dtype=dtype)
                arr.tofile(f_out)
                total_tokens += len(buf)
                buf.clear()

                if show_token_rate:
                    pbar.set_postfix(tokens=total_tokens)

        finally:
            pbar.close()

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-encode a text file into a binary token stream (uint16 or int32_le) with a progress bar."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Path to input UTF-8 .txt file")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocab.json (from bpe_hf.py)")
    parser.add_argument("--merges-path", type=str, required=True, help="Path to merges.txt (from bpe_hf.py)")

    parser.add_argument("--out-bin", type=str, required=True, help="Output .bin path (flat token stream)")

    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help="Optional output meta.json path (default: out-bin with .meta.json suffix)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens (must match your training). Example: --special-tokens "<|endoftext|>" "<|pad|>"',
    )

    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1_000_000,
        help="How many token ids to buffer before writing to disk",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32_le"],
        help="Output dtype for the token stream. Use uint16 for smaller files; use int32_le if ids may exceed 65535.",
    )

    args = parser.parse_args()

    dtype, meta_dtype = _dtype_from_arg(args.dtype)
    check_uint16_range = (meta_dtype == "uint16")

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_tokens,
    )

    if check_uint16_range:
        vocab_max_id = _max_token_id_from_vocab_json(args.vocab_path)
        if vocab_max_id > 65535:
            raise ValueError(
                f"uint16 overflow: vocab has max id {vocab_max_id} (> 65535). "
                "Use --dtype int32_le or reduce vocab size."
            )

    total = write_tokens_bin_with_progress(
        tokenizer=tokenizer,
        input_path=args.input_path,
        out_bin_path=args.out_bin,
        dtype=dtype,
        chunk_tokens=args.chunk_tokens,
        check_uint16_range=check_uint16_range,
        show_token_rate=True,
    )

    total2 = infer_num_tokens_from_file(args.out_bin, dtype=dtype)
    if total2 != total:
        raise RuntimeError(f"Token count mismatch: streamed={total}, filesize-derived={total2}")

    meta = {
        "input_path": args.input_path,
        "vocab_path": args.vocab_path,
        "merges_path": args.merges_path,
        "special_tokens": args.special_tokens,
        "dtype": meta_dtype,
        "num_tokens": total,
    }

    meta_path = args.meta_path
    if meta_path is None:
        meta_path = str(Path(args.out_bin).with_suffix(".meta.json"))
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote tokens bin: {args.out_bin}")
    print(f"Num tokens    : {total}")
    print(f"Dtype         : {meta_dtype}")
    print(f"Wrote meta    : {meta_path}")


if __name__ == "__main__":
    main()








"""
python3.13 prepare_data.py \
  --input-path ./data/TinyStoriesV2-GPT4-train.txt \
  --vocab-path ./bpe_out/TinyStories/vocab.json \
  --merges-path ./bpe_out/TinyStories/merges.txt \
  --out-bin ./data_bin/TinyStories/train.bin \
  --dtype uint16 \
  --special-tokens "<|endoftext|>" \
  --chunk-tokens 1000000
"""


"""
python3.13 prepare_data.py \
  --input-path ./data/TinyStoriesV2-GPT4-valid.txt \
  --vocab-path ./bpe_out/TinyStories/vocab.json \
  --merges-path ./bpe_out/TinyStories/merges.txt \
  --out-bin ./data_bin/TinyStories/valid.bin \
  --dtype uint16 \
  --special-tokens "<|endoftext|>" \
  --chunk-tokens 1000000
"""


"""
python3.13 prepare_data.py \
  --input-path ./data/owt_train.txt \
  --vocab-path ./bpe_out/owt/vocab.json \
  --merges-path ./bpe_out/owt/merges.txt \
  --out-bin ./data_bin/owt/train.bin \
  --dtype uint16 \
  --special-tokens "<|endoftext|>" \
  --chunk-tokens 1000000
"""


"""
python3.13 prepare_data.py \
  --input-path ./data/owt_valid.txt \
  --vocab-path ./bpe_out/owt/vocab.json \
  --merges-path ./bpe_out/owt/merges.txt \
  --out-bin ./data_bin/owt/valid.bin \
  --dtype uint16 \
  --special-tokens "<|endoftext|>" \
  --chunk-tokens 1000000
"""

