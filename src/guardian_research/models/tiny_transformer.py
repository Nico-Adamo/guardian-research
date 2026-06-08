"""A tiny, inspectable decoder-only transformer.

Hand-rolled (rather than ``nn.TransformerEncoder``) so the causal attention and
parameter count are obvious and stable across torch versions. Small enough to
train on a CPU in seconds for the smoke test; the same class scales up via
config for a GPU run. Depth-over-width is the default, matching the catapult
essay's "skinny net" intuition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
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
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
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
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate_greedy(
        self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int
    ) -> list[int]:
        """Greedy-decode from a 1-D prompt tensor; stop at eos or max_new_tokens."""
        self.eval()
        ids = prompt_ids.tolist()
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            window = ids[-self.cfg.max_len :]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = self(x)[0, -1]
            nxt = int(torch.argmax(logits).item())
            ids.append(nxt)
            if nxt == eos_id:
                break
        return ids[len(prompt_ids):]


def build_model(cfg: TinyTransformerConfig) -> TinyTransformer:
    return TinyTransformer(cfg)
