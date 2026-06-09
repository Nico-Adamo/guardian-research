"""Modular-arithmetic grokking testbed: dataset + a fast end-to-end run."""

from guardian_research.common.artifacts import load_result
from guardian_research.common.hydra_utils import compose_config, to_container
from guardian_research.data.modular import ModularConfig, build_modular_splits
from guardian_research.launchers.local import run_local


def test_modular_splits_disjoint_and_correct():
    splits = build_modular_splits(ModularConfig(p=23, op="+", train_frac=0.5, seed=0))
    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xval = splits["X_val"]
    # train + val partition all p*p pairs, disjoint.
    assert len(Xtr) + len(Xval) == 23 * 23
    # answers are correct modular sums (target predicted at the eq position).
    a, b = int(Xtr[0][0]), int(Xtr[0][2])
    assert int(ytr[0]) == (a + b) % 23
    assert splits["vocab_size"] == 23 + 2


def test_modular_grokking_runs_and_logs_curves():
    cfg = to_container(
        compose_config(
            [
                "+exp=arithmetic_modular_grok",
                "schedule=cyclic_weight_decay",
                "seed=0",
                "device=cpu",
                "data.p=23",
                "data.train_frac=0.5",
                "model.d_model=32",
                "model.n_layers=1",
                "train.max_steps=150",
                "train.eval_every=50",
                "train.log_every=50",
            ]
        )
    )
    out = run_local(cfg)
    r = load_result(out)
    assert r.status == "completed"
    assert r.metrics.get("train_acc") and r.metrics.get("val_acc")
    assert "final_val_acc" in r.final_metrics
    assert "grok_step" in r.final_metrics
    # Memorization-first: at the first eval, train_acc should be >= val_acc.
    assert r.metrics["train_acc"][0].value >= r.metrics["val_acc"][0].value - 1e-6
