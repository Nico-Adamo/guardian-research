"""End-to-end CPU smoke: a tiny arithmetic run writes a valid RunResult."""

from guardian_research.common.artifacts import load_result
from guardian_research.common.hydra_utils import compose_config, to_container
from guardian_research.launchers.local import run_local


def test_cpu_smoke_train():
    cfg = to_container(
        compose_config(
            [
                "+exp=arithmetic_catapult",
                "model=tiny_transformer",
                "schedule=baseline_cosine",
                "seed=0",
                "device=cpu",
                "train.max_steps=30",
                "train.eval_every=15",
                "train.eval_n=16",
                "train.log_every=10",
                "data.n_train=400",
                "data.n_easy_eval=32",
                "data.n_hard_eval=32",
            ]
        )
    )
    out = run_local(cfg)
    r = load_result(out)
    assert r.status == "completed"
    assert "final_hard_acc" in r.final_metrics
    assert "final_easy_acc" in r.final_metrics
    assert "final_memorization_gap" in r.final_metrics
    assert r.metrics.get("train_loss"), "expected a train_loss history"
    assert r.params["num_params"] > 0
    assert r.params["schedule"] == "baseline_cosine"
