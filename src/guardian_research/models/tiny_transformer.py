"""A tiny, inspectable decoder-only transformer.

Hand-rolled (rather than ``nn.TransformerEncoder``) so the causal attention and
parameter count are obvious and stable across torch versions. Small enough to
train on a CPU in seconds for the smoke test; the same class scales up via
config. Depth-over-width is the default, matching the catapult essay's "skinny
net" intuition.

Positional scheme is configurable (``pos_encoding``):

* ``learned`` — absolute learned position embeddings (the classic default).
  Cannot extrapolate: positions beyond the training length are never trained,
  so out-of-distribution (longer) inputs land on untrained position vectors.
* ``none`` — NoPE: no positional signal at all; the causal mask is the only
  source of order. Surprisingly strong at *length generalization* on algorithmic
  tasks, which is exactly the catapult/grokking regime of interest.
* ``rope`` — rotary embeddings applied to q/k. Relative-by-construction, so the
  same circuit applies at any length — the standard choice when you want the
  HARD out-of-distribution-length split to even be solvable in principle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_ENCODINGS = ("learned", "none", "rope")


@dataclass
class TinyTransformerConfig:
    vocab_size: int = 17
    max_len: int = 32
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    tie_weights: bool = True
    pos_encoding: str = "learned"  # learned | none | rope
    rope_base: float = 10000.0


def _rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (seq_len, head_dim/2) for rotary embeddings."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, half)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x of shape (B, n_heads, T, head_dim) (rotate-half)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.pos_encoding = cfg.pos_encoding
        self.rope_base = cfg.rope_base
        if self.pos_encoding == "rope":
            assert self.head_dim % 2 == 0, "rope requires an even head_dim"
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if self.pos_encoding == "rope":
            cos, sin = _rope_cos_sin(t, self.head_dim, self.rope_base, x.device, x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        if cfg.pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}, got {cfg.pos_encoding!r}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Absolute position table only exists for the 'learned' scheme.
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model) if cfg.pos_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Count tied embedding once.
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if self.pos_emb is not None and t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len} (learned positions)")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> list[int]:
        """Greedy-decode from a 1-D prompt tensor; stop at eos or max_new_tokens."""
        self.eval()
        ids = prompt_ids.tolist()
        device = next(self.parameters()).device
        # Only the learned scheme is hard-limited by max_len; rope/none extrapolate.
        window_cap = self.cfg.max_len if self.pos_emb is not None else 10**9
        for _ in range(max_new_tokens):
            window = ids[-window_cap:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = self(x)[0, -1]
            nxt = int(torch.argmax(logits).item())
            ids.append(nxt)
            if nxt == eos_id:
                break
        return ids[len(prompt_ids):]


def build_model(cfg: TinyTransformerConfig) -> TinyTransformer:
    return TinyTransformer(cfg)
