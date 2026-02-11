from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


@dataclass
class BPETokenizer:
    tokenizer: Tokenizer
    special_tokens: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.special_tokens is None:
            self.special_tokens = []

        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        if self.special_tokens:
            self.tokenizer.add_special_tokens(list(self.special_tokens))

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        model = BPE.from_file(str(vocab_filepath), str(merges_filepath))
        tok = Tokenizer(model)
        return cls(tokenizer=tok, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        return list(enc.ids)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        tail_keep = max(256, max((len(s) for s in (self.special_tokens or [])), default=0) + 16)
        buf = ""

        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk

            if len(buf) > 4096:
                cut = len(buf) - tail_keep
                ws = max(buf.rfind("\n", 0, cut), buf.rfind(" ", 0, cut), buf.rfind("\t", 0, cut))
                if ws != -1 and ws > 0:
                    cut = ws + 1

                prefix, buf = buf[:cut], buf[cut:]
                for tid in self.encode(prefix):
                    yield tid

        if buf:
            for tid in self.encode(buf):
                yield tid

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)
