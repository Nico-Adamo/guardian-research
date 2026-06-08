"""Config-loading tests: every schedule composes, sweep + gpu configs load."""

import pytest

from guardian_research.common.hydra_utils import compose_config, to_container

SCHEDULES = ["baseline_cosine", "onecycle_high_lr", "cyclic_lr", "cyclic_weight_decay"]


@pytest.mark.parametrize("sched", SCHEDULES)
def test_compose_arithmetic(sched):
    cfg = to_container(
        compose_config(["+exp=arithmetic_catapult", "model=tiny_transformer", f"schedule={sched}", "seed=0"])
    )
    assert cfg["experiment"] == "arithmetic_catapult"
    assert cfg["runner"] == "arithmetic"
    assert cfg["model"]["d_model"] == 64
    assert cfg["schedule"]["name"] == sched
    # Interpolation ${train.lr} resolves to the (possibly exp-overridden) value.
    assert cfg["schedule"]["base_lr"] == cfg["train"]["lr"]
    assert cfg["train"]["max_steps"] > 0


def test_compose_sweep():
    cfg = to_container(compose_config(["+exp=arithmetic_catapult", "sweep=arith_lr_wd_seed_v0"]))
    assert cfg["sweep"]["name"] == "arith_lr_wd_seed_v0"
    assert "schedule" in cfg["sweep"]["axes"]
    assert "train.lr" in cfg["sweep"]["axes"]
    assert len(cfg["sweep"]["seeds"]) == 2


def test_gpu_config_present_but_bigger():
    cfg = to_container(compose_config(["+exp=arithmetic_catapult_gpu", "model=small_transformer"]))
    assert cfg["model"]["d_model"] == 256
    assert cfg["train"]["max_steps"] >= 10000  # GPU-scale, not a CPU toy
