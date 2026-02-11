from typing import Optional
from collections.abc import Callable, Iterable
import torch
import math



class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["second_moment"] = torch.zeros_like(p.data)

                first_moment, second_moment = (
                    state["first_moment"],
                    state["second_moment"],
                )
                state["step"] += 1
                step = state["step"]

                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                adjusted_denominator = 1 - beta1**step
                adjusted_numerator = math.sqrt(1 - beta2**step)

                adjusted_lr = lr * adjusted_numerator / adjusted_denominator

                p.data -= adjusted_lr * first_moment / (second_moment.sqrt() + eps)
                p.data -= lr * weight_decay * p.data

        return loss
    

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        lr = max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr
