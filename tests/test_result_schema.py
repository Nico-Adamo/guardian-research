"""Result schema + RunWriter round-trip tests."""

from guardian_research.common.artifacts import RunWriter, load_result, new_run_id
from guardian_research.common.schemas import SCHEMA_VERSION, RunResult


def test_runwriter_roundtrip():
    rid = new_run_id("test")
    w = RunWriter("test_schema", rid, seed=0)
    w.set_config({"a": 1, "nested": {"b": 2}})
    w.set_params(schedule="baseline_cosine", lr=1e-3)
    w.log_metrics(0, train_loss=2.0, easy_acc=0.1)
    w.log_metrics(1, train_loss=1.5, easy_acc=0.2)
    w.set_final(final_easy_acc=0.5)
    out = w.finish()

    r = load_result(out)
    assert r.schema_version == SCHEMA_VERSION
    assert r.experiment == "test_schema"
    assert r.status == "completed"
    assert r.final_metrics["final_easy_acc"] == 0.5
    assert len(r.metrics["train_loss"]) == 2
    assert r.metrics["train_loss"][0].value == 2.0
    assert r.git.sha  # provenance recorded
    assert r.env.python


def test_failed_run_records_error():
    rid = new_run_id("test")
    try:
        with RunWriter("test_schema", rid, seed=0) as w:
            w.log_metrics(0, train_loss=1.0)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    r = load_result(w.dir / "results.json")
    assert r.status == "failed"
    assert "boom" in (r.error or "")


def test_schema_versioned():
    r = RunResult(
        run_id="x",
        experiment="e",
        seed=0,
        created_at="2026-01-01T00:00:00+00:00",
        git={"sha": "abc", "dirty": False},
        env={"python": "3.11", "platform": "test", "hostname": "h"},
    )
    assert r.schema_version == SCHEMA_VERSION
