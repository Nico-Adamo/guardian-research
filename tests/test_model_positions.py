"""Positional-scheme tests: rope/none must be able to handle OOD lengths."""

import pytest
import torch

from guardian_research.models.tiny_transformer import TinyTransformer, TinyTransformerConfig


def _cfg(pos: str, max_len: int = 16) -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocab_size=17, max_len=max_len, d_model=32, n_layers=2, n_heads=4, pos_encoding=pos
    )


@pytest.mark.parametrize("pos", ["learned", "none", "rope"])
def test_forward_shapes(pos):
    model = TinyTransformer(_cfg(pos))
    x = torch.randint(0, 17, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 17)


def test_learned_cannot_exceed_max_len():
    model = TinyTransformer(_cfg("learned", max_len=8))
    with pytest.raises(ValueError):
        model(torch.randint(0, 17, (1, 12)))  # 12 > max_len 8


@pytest.mark.parametrize("pos", ["none", "rope"])
def test_rope_none_extrapolate_beyond_max_len(pos):
    # No absolute position table => longer-than-"max_len" sequences are allowed,
    # which is what makes the hard out-of-distribution-length split solvable.
    model = TinyTransformer(_cfg(pos, max_len=8))
    logits = model(torch.randint(0, 17, (1, 20)))  # 20 >> 8
    assert logits.shape == (1, 20, 17)


def test_rope_requires_even_head_dim():
    # head_dim must be even for rotate-half; d_model=30/n_heads=4 -> 7.5 invalid,
    # use an odd head_dim to confirm the guard fires.
    with pytest.raises(AssertionError):
        TinyTransformer(TinyTransformerConfig(vocab_size=17, max_len=8, d_model=12, n_heads=4, pos_encoding="rope"))
        # d_model=12, heads=4 -> head_dim=3 (odd)


@pytest.mark.parametrize("pos", ["learned", "none", "rope"])
def test_generate_runs(pos):
    model = TinyTransformer(_cfg(pos))
    out = model.generate_greedy(torch.tensor([1, 5, 6, 7]), max_new_tokens=5, eos_id=2)
    assert isinstance(out, list)
