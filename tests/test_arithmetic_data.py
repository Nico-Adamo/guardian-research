"""Arithmetic dataset tests: tokenizer, OOD/disjoint hard split, loss masking."""

from guardian_research.data.arithmetic import (
    ArithmeticConfig,
    CharTokenizer,
    build_splits,
    make_example,
)


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    s = "123+45="
    assert tok.decode(tok.encode(s)) == s


def test_make_example_reverse():
    prompt, answer = make_example(12, 34, "+", reverse_answer=True)
    assert prompt == "12+34="
    assert answer == "64"  # 46 reversed (least-significant digit first)
    prompt, answer = make_example(12, 34, "+", reverse_answer=False)
    assert answer == "46"


def test_hard_split_mixes_ood_and_carry_and_is_disjoint():
    cfg = ArithmeticConfig(
        n_train=600, n_easy_eval=64, n_hard_eval=64,
        train_min_digits=1, train_max_digits=2, hard_min_digits=3, hard_max_digits=4,
        hard_ood_frac=0.5, seed=0,
    )
    splits = build_splits(cfg)
    train_pairs = set(splits["train"].pairs)
    hard = splits["hard_eval"]
    # The whole hard split is disjoint from train (different examples).
    assert all(p not in train_pairs for p in hard.pairs)
    # It contains BOTH out-of-distribution lengths AND in-distribution carry cases.
    lengths = [max(len(str(a)), len(str(b))) for a, b in hard.pairs]
    assert any(le >= cfg.hard_min_digits for le in lengths), "expected some OOD-length examples"
    assert any(le <= cfg.train_max_digits for le in lengths), "expected some in-range carry examples"
    assert splits["hard_ood_eval"] is not None and splits["hard_carry_eval"] is not None


def test_item_shapes_and_answer_only_mask():
    cfg = ArithmeticConfig(n_train=50, n_easy_eval=8, n_hard_eval=8, seed=1)
    ds = build_splits(cfg)["train"]
    x, y, m = ds[0]
    assert x.shape == y.shape == m.shape
    assert m.sum() > 0  # the answer region is scored
    assert float(m[0]) == 0.0  # the very first (prompt) position is not scored


def test_carry_heavy_present():
    cfg = ArithmeticConfig(n_hard_eval=64, carry_heavy_frac=1.0, hard_min_digits=4, hard_max_digits=4, seed=3)
    hard = build_splits(cfg)["hard_eval"]
    # With carry_heavy_frac=1.0, many pairs should produce a carry out (result
    # has more digits than the longer operand) at least sometimes.
    carries = sum(1 for a, b in hard.pairs if len(str(a + b)) > max(len(str(a)), len(str(b))))
    assert carries > 0
