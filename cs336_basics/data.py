from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt
import torch


ArrayLike1D = Union[npt.NDArray[np.integer], np.memmap]





def get_batch(
    dataset: ArrayLike1D,
    batch_size: int,
    context_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(dataset, "ndim", None) != 1:
        raise ValueError("dataset must be a 1D array of token IDs.")

    n = int(dataset.shape[0])
    if n < context_len + 1:
        raise ValueError(f"dataset is too small: needs at least {context_len + 1} tokens.")

    max_start_id = n - context_len - 1
    start_ids = np.random.randint(0, max_start_id + 1, size=batch_size)

    x_np = np.empty((batch_size, context_len), dtype=dataset.dtype)
    y_np = np.empty((batch_size, context_len), dtype=dataset.dtype)

    for i, s in enumerate(start_ids):
        x_np[i, :] = dataset[s : s + context_len]
        y_np[i, :] = dataset[s + 1 : s + context_len + 1]

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y
