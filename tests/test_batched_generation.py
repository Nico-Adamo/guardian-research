"""Verify that batched greedy generation matches serial output exactly."""

import pytest
import torch

from guardian_research.data.arithmetic import ArithmeticConfig, CharTokenizer, build_splits
from guardian_research.models.tiny_transformer import TinyTransformer, TinyTransformerConfig


def _make_model(pos: str = "rope") -> TinyTransformer:
    cfg = TinyTransformerConfig(
        vocab_size=17, max_len=32, d_model=32, n_layers=2, n_heads=4,
        d_ff=64, pos_encoding=pos,
    )
    return TinyTransformer(cfg)


@pytest.mark.parametrize("pos", ["learned", "none", "rope"])
def test_batch_matches_serial(pos):
    torch.manual_seed(42)
    model = _make_model(pos)
    tok = CharTokenizer()

    prompts_str = ["1+2=", "99+1=", "123+456=", "7+8="]
    prompts = [[tok.bos_id] + tok.encode(s) for s in prompts_str]

    # Serial generation.
    serial_results = []
    for p in prompts:
        p_tensor = torch.tensor(p, dtype=torch.long)
        gen = model.generate_greedy(p_tensor, max_new_tokens=10, eos_id=tok.eos_id)
        serial_results.append(gen)

    # Batched generation.
    batch_results = model.generate_greedy_batch(
        prompts, max_new_tokens=10, eos_id=tok.eos_id, pad_id=tok.pad_id
    )

    for i, (serial, batched) in enumerate(zip(serial_results, batch_results, strict=True)):
        assert serial == batched, (
            f"Mismatch at index {i} (prompt={prompts_str[i]!r}): "
            f"serial={serial}, batched={batched}"
        )


def test_batch_early_stopping():
    """Sequences that hit EOS at different times still produce correct output."""
    torch.manual_seed(0)
    model = _make_model()
    tok = CharTokenizer()

    # Use varied-length prompts to stress left-padding.
    prompts_str = ["1+1=", "99999+11111=", "5+3="]
    prompts = [[tok.bos_id] + tok.encode(s) for s in prompts_str]

    results = model.generate_greedy_batch(
        prompts, max_new_tokens=20, eos_id=tok.eos_id, pad_id=tok.pad_id
    )
    assert len(results) == 3
    for r in results:
        assert tok.eos_id not in r
        assert tok.pad_id not in r


def test_eval_accuracy_uses_batched_path():
    """Integration: _eval_accuracy produces a float in [0, 1] without hanging."""
    from guardian_research.experiments.arithmetic.train import _eval_accuracy

    torch.manual_seed(7)
    cfg = ArithmeticConfig(
        n_train=50, n_easy_eval=20, n_hard_eval=20,
        train_min_digits=1, train_max_digits=2,
        hard_min_digits=3, hard_max_digits=4, seed=0,
    )
    tok = CharTokenizer()
    splits = build_splits(cfg)
    model = _make_model()

    acc = _eval_accuracy(model, splits["easy_eval"], tok, n=20, max_new=32, batch_size=8)
    assert 0.0 <= acc <= 1.0
