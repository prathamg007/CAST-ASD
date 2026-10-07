"""Post-LN transformer blocks shaped like TalkNet-ASD's `attentionLayer`.

Attention over time is causal unless `causal: false`. A key-padding mask is
added to the causal mask, not substituted for it. There is no positional
encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    """A stack of `TransformerLayer`s built from a config dict.

    Reads `d_model`, `m_ff`, `n_layers`, `n_heads`, `dropout`, and optionally
    `causal` (default True), `bias` (default False) and `init` (`custom`:
    N(0, 0.02^2) re-initialisation, or `pytorch_default`).
    """

    def __init__(self, cfg, cross_attention=False):
        super().__init__()
        self.dim = cfg["d_model"]
        self.dff = cfg["m_ff"] * self.dim
        self.num_layers = cfg["n_layers"]
        self.num_heads = cfg["n_heads"]
        self.dropout = cfg["dropout"]
        self.cross_attention = cross_attention
        self.causal = bool(cfg.get("causal", True))
        self.bias = bool(cfg.get("bias", False))
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    cross_attention=self.cross_attention,
                    bias=self.bias,
                    causal=self.causal,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.init = cfg.get("init", "custom")
        if self.init not in ("custom", "pytorch_default"):
            raise ValueError(f"unknown init: {self.init!r}")
        if self.init == "custom":
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, src=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src, key_padding_mask=key_padding_mask)
        return x


class TransformerLayer(nn.Module):
    """Self-attention + FFN, or (cross_attention) MHA(Q=x, K=V=src) + FFN.

    Post-LN: each sublayer is `norm(x + dropout(fn(x)))`.
    """

    def __init__(
        self,
        dim=256,
        ffn_dim=768,
        num_heads=4,
        dropout=0.1,
        cross_attention=False,
        bias=False,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.cross_attention = cross_attention
        self.self_attention = not cross_attention
        self.dropout = nn.Dropout(dropout)

        if self.self_attention:
            self.ln_self_attn = nn.LayerNorm(dim)
            self.mha = CausalMultiHeadAttention(
                dim=dim, num_heads=num_heads, dropout=dropout, bias=bias, causal=causal,
            )

        if cross_attention:
            self.ln_cross_query = nn.LayerNorm(dim)
            self.mha_cross = CausalMultiHeadAttention(
                dim=dim, num_heads=num_heads, dropout=dropout, bias=bias, causal=causal,
            )

        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.ffnetwork = ffn_block(dim, ffn_dim, dropout=dropout, bias=bias)

    def forward(self, x, src=None, key_padding_mask=None):
        if self.self_attention:
            x = self.ln_self_attn(
                x + self.dropout(self.mha(Q=x, K=x, V=x, key_padding_mask=key_padding_mask))
            )

        if self.cross_attention and src is not None:
            x = self.ln_cross_query(
                x
                + self.dropout(
                    self.mha_cross(Q=x, K=src, V=src, key_padding_mask=key_padding_mask)
                )
            )

        return self.ln_ffnetwork(x + self.dropout(self.ffnetwork(x)))


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with an optional additive float mask.

    Pass a float mask (0 / -inf); SDPA would read a bool mask with the opposite
    meaning (True = attend). The output projection has no dropout; the caller
    applies residual dropout.
    """

    def __init__(self, dim, num_heads, dropout, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def _project(self, Q, K, V):
        B, T, _ = Q.size()
        Tk = K.size(1)
        q = self.q_proj(Q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(K).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(V).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _combine(self, y, B, T, C):
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward(self, Q, K, V, mask=None):
        B, T, C = Q.size()
        q, k, v = self._project(Q, K, V)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self._combine(y, B, T, C)


class CausalMultiHeadAttention(MultiHeadAttention):
    """Attention over time with a causal mask and an optional key-padding mask.

    `causal=False` drops only the triangular mask, so a non-causal model uses
    the identical module.
    """

    def __init__(self, dim, num_heads, dropout, bias=False, causal=True):
        super().__init__(dim, num_heads, dropout, bias)
        self.causal = causal
        # Caches only the causal part, never a padded composite.
        self._causal = None

    @staticmethod
    def prepare_causal_mask(T, device="cpu", dtype=torch.float32):
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype))
        return mask.view(1, 1, T, T).requires_grad_(False)

    def _causal_mask(self, T, device, dtype):
        """[1, 1, T, T]: 0 where attendable, -inf where masked."""
        base = self.prepare_causal_mask(T, device, dtype)
        return base.masked_fill(base == 0, float("-inf")).masked_fill(base == 1, 0.0)

    def _cached_causal_mask(self, span, device, dtype):
        if (
            self._causal is None
            or self._causal.shape[-1] < span
            or self._causal.device != device
            or self._causal.dtype != dtype
        ):
            self._causal = self._causal_mask(span, device, dtype)
        return self._causal

    def forward(self, Q, K, V, key_padding_mask=None):
        B, T, C = Q.size()
        Tk = K.size(1)
        q, k, v = self._project(Q, K, V)

        if self.causal:
            mask = self._cached_causal_mask(max(T, Tk), Q.device, Q.dtype)[..., :T, :Tk]
        else:
            mask = Q.new_zeros(1, 1, T, Tk)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, Tk):
                raise ValueError(
                    f"key_padding_mask must be [batch, keys], got "
                    f"{tuple(key_padding_mask.shape)} for keys {Tk}"
                )
            # True marks a real key. A query with no real key at or before it
            # gets an all -inf row; SDPA (torch >= 2.5) returns zeros for it.
            bias = torch.zeros(
                key_padding_mask.shape, device=Q.device, dtype=mask.dtype
            )
            bias.masked_fill_(~key_padding_mask, float("-inf"))
            mask = mask + bias[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self._combine(y, B, T, C)


def ffn_block(din, dff, dropout=0.0, bias=False):
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dff, din, bias=bias),
    )
