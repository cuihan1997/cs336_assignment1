import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int, Bool
import math


class Linear(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights: Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty((d_out, d_in), device=device, dtype=dtype)
        )
        std = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights: Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            torch.empty((vocab_size, d_model), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(
        self, token_ids: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        return self.weights[token_ids]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(
        self, x: Float[Tensor, "batch_size sequence_length d_model"]
    ) -> Float[Tensor, "batch_size sequence_length d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            reduce(
                x.square(),
                "batch_size sequence_length d_model -> batch_size sequence_length 1",
                "mean",
            )
            + self.eps
        )

        result = x / rms * rearrange(self.weights, "d_model -> 1 1 d_model")

        return result.to(in_dtype)


def silu(x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def _rotate(x: Float[Tensor, "... seq_len d_k"]):
    """
    Rotate: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)
    """
    x_even = x[..., ::2]  # (..., seq_len, d_k/2)
    x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)  # (..., seq_len, d_k)


class RoPE(nn.Module):

    def __init__(
        self,
        rope_theta: float,
        d_k: int,  # dimension of query and key vectors
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, but got d_k={d_k}")

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )  # Shape: (d_k/2, )

        positions = torch.arange(
            max_seq_len, device=device
        ).float()  # Shape: (max_seq_len, )
        angles = einsum(
            positions, inv_freq, "seq_len, d -> seq_len d"
        )  # Shape: (max_seq_len, d_k/2)

        cos = torch.cos(angles)  # Shape: (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # Shape: (max_seq_len, d_k/2)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        seq_len = x.shape[-2]
        pos = token_positions[..., :seq_len].long()  # (..., seq_len)
        cos_pos = self.cos[pos]  # (..., seq_len, d_k/2)
        sin_pos = self.sin[pos]  # (..., seq_len, d_k/2)

        cos_pos = cos_pos.repeat_interleave(2, dim=-1)  # (..., seq_len, d_k)
        sin_pos = sin_pos.repeat_interleave(2, dim=-1)  # (..., seq_len, d_k)

        x_rotated = _rotate(x)
        return x * cos_pos + x_rotated * sin_pos


def softmax(x: Float[Tensor, "..."], dim: int):
    """
    Numerically stable softmax function.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    attn_weights = softmax(scores / math.sqrt(d_k), dim=-1)
    output = einsum(
        attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )
    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, but got d_model={d_model}, num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(
            Q,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )

        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )  # Shape: (seq_len, seq_len)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  #

        attn_output = rearrange(
            attn_output,
            "batch_size num_heads seq_len d_k -> batch_size seq_len (num_heads d_k)",
        )  # Shape: (batch_size, seq_len, d_model)

        output = self.o_proj(attn_output)

        return output


class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, but got d_model={d_model}, num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_in=d_model, d_out=d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, self.d_k, max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(
            Q,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k",
            num_heads=self.num_heads,
        )

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )  # Shape: (seq_len, seq_len)

        attn_output = scaled_dot_product_attention(
            Q, K, V, mask=causal_mask
        )  # Shape: (batch_size, num_heads, seq_len, d_k)

        attn_output = rearrange(
            attn_output,
            "batch_size num_heads seq_len d_k -> batch_size seq_len (num_heads d_k)",
        )  # Shape: (batch_size, seq_len, d_model)

        output = self.o_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        # pre-norm attention
        x = x + self.attn(self.norm1(x), token_positions)
        # pre-norm ffn
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size=vocab_size, d_model=d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_in=d_model, d_out=vocab_size, device=device, dtype=dtype
        )

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        x = self.token_embeddings(in_indices)  # Shape: (batch_size, seq_len, d_model)
        token_positions = torch.arange(
            in_indices.shape[-1], device=in_indices.device
        ).unsqueeze(
            0
        )  # Shape: (1, seq_len)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # Shape: (batch_size, seq_len, vocab_size)
        return logits
