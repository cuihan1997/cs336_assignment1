from jaxtyping import Float, Int
from collections.abc import Iterable
from torch import Tensor
import torch


def cross_entropy(
    predicted_logits: Float[Tensor, "batch_size seq_len vocab_size"],
    target_token_ids: Int[Tensor, "batch_size seq_len"],
):
    x = predicted_logits - predicted_logits.max(dim=-1, keepdim=True).values
    x_exp = x.exp()
    x_exp_sum = x_exp.sum(dim=-1, keepdim=True)
    log_probs = x - x_exp_sum.log()  # Shape: (batch_size, seq_len, vocab_size)
    target_log_probs = log_probs.gather(
        dim=-1, index=target_token_ids.unsqueeze(-1)
    ).squeeze(
        -1
    )  # Shape: (batch_size, seq_len)
    loss = -target_log_probs.mean()
    return loss


def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
):
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            grad = p.grad.data
            total_norm_sq += grad.norm(2).item() ** 2

    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)




