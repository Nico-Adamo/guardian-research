"""CPU test for the dynamic-grokking (dynamic-evaluation) toy.

Runs the harness at *tiny* sizes and asserts that a valid RunResult is written
with: inner-step metrics, output-diversity metrics, and a static-vs-dynamic
comparison. It does NOT assert that dynamic beats static — that would be a
scientific claim the toy is not entitled to make.
"""

import pytest

from guardian_research.common.artifacts import load_result
from guardian_research.common.hydra_utils import compose_config, to_container
from guardian_research.experiments.dynamic_grokking.run import (
    UPDATE_TARGETS,
    _select_inner_params,
)
from guardian_research.launchers.local import run_local
from guardian_research.models.tiny_transformer import TinyTransformer, TinyTransformerConfig


def _tiny_overrides(update_target: str = "full") -> list[str]:
    return [
        "+exp=dynamic_grokking",
        "model=tiny_transformer",
        "seed=0",
        "device=cpu",
        # Tiny everything for a fast CPU test.
        "data.n_train=200",
        "data.n_easy_eval=24",
        "data.n_hard_eval=24",
        "train.max_steps=20",
        "train.batch_size=64",
        "dynamic.n_problems=3",
        "dynamic.inner_steps=2",
        "dynamic.samples_per_step=2",
        f"dynamic.update_target={update_target}",
    ]


def test_config_composes_to_dynamic_runner():
    cfg = to_container(compose_config(["+exp=dynamic_grokking", "model=tiny_transformer", "seed=0"]))
    assert cfg["experiment"] == "dynamic_grokking"
    assert cfg["runner"] == "dynamic_grokking"
    assert cfg["dynamic"]["update_target"] in UPDATE_TARGETS
    assert cfg["dynamic"]["inner_steps"] > 0


def test_run_writes_comparison_and_inner_metrics():
    cfg = to_container(compose_config(_tiny_overrides("full")))
    out = run_local(cfg)
    r = load_result(out)

    assert r.status == "completed"
    assert r.experiment == "dynamic_grokking"

    # Static-vs-dynamic comparison is present in final metrics.
    for key in (
        "dynamic_success",
        "static_success",
        "dynamic_minus_static",
        "dynamic_unique_mean",
        "static_unique_mean",
        "n_problems",
    ):
        assert key in r.final_metrics, f"missing final metric {key}"

    # Success rates are valid fractions.
    assert 0.0 <= r.final_metrics["dynamic_success"] <= 1.0
    assert 0.0 <= r.final_metrics["static_success"] <= 1.0

    # Per-problem comparison series were logged.
    assert r.metrics.get("dyn_success"), "expected a per-problem dyn_success series"
    assert r.metrics.get("static_success"), "expected a per-problem static_success series"
    assert r.metrics.get("dyn_unique"), "expected output-diversity (dyn_unique) series"

    # At least one problem produced an inner-step loss series (unless the base
    # solved every sampled problem on the first try, which is unlikely here).
    inner_series = [name for name in r.metrics if name.endswith("_inner_loss")]
    if r.final_metrics["dynamic_solved"] < r.final_metrics["n_problems"]:
        assert inner_series, "expected at least one per-problem inner-step loss series"

    assert r.params["update_target"] == "full"
    assert r.params["num_params"] > 0


def test_last_layer_target_runs():
    cfg = to_container(compose_config(_tiny_overrides("last_layer")))
    out = run_local(cfg)
    r = load_result(out)
    assert r.status == "completed"
    assert r.params["update_target"] == "last_layer"
    assert "dynamic_minus_static" in r.final_metrics


def test_last_layer_selects_fewer_params_than_full():
    mcfg = TinyTransformerConfig(vocab_size=17, max_len=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model = TinyTransformer(mcfg)
    full = _select_inner_params(model, "full")
    last = _select_inner_params(model, "last_layer")
    n_full = sum(p.numel() for p in full)
    n_last = sum(p.numel() for p in last)
    assert 0 < n_last <= n_full


def test_lora_target_is_honest_stub():
    mcfg = TinyTransformerConfig(vocab_size=17, max_len=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model = TinyTransformer(mcfg)
    with pytest.raises(NotImplementedError):
        _select_inner_params(model, "lora")
