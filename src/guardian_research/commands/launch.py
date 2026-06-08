"""`ga launch` — render/dry-run a cloud job, or (gated) submit a real one.

Safety invariants enforced here:
  * ``--dry-run`` spends nothing and records that a dry-run happened.
  * a REAL launch requires ``--yes`` AND passes the budget preflight (per-job +
    daily caps, allowed provider, allowed data class, clean tree, exact SHA,
    and a prior dry-run).
  * even when all gates pass, actual submission only happens when
    ``GUARDIAN_ALLOW_REAL_LAUNCH=1`` and the ``[cloud]`` extra is installed;
    otherwise the exact ``sky launch`` command is printed for a human to run.
"""

from __future__ import annotations

import argparse
import json

from ..common.budget import BudgetPolicy, estimate_cost
from ..common.env_info import current_sha, git_info
from ..common.hydra_utils import compose_config, split_overrides, to_container
from ..common.logging import console
from ..common.paths import runs_dir
from ..launchers.skypilot import LaunchSpec, dry_run_text, render_task_yaml

NAME = "launch"
HELP = "Dry-run (default-safe) or gated real cloud launch: --dry-run --provider skypilot +exp=... sweep=..."


def _marker_path():
    return runs_dir() / "_last_dry_run.json"


def _record_dry_run(key: str) -> None:
    path = _marker_path()
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    data[key] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _dry_run_done(key: str) -> bool:
    path = _marker_path()
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get(key))
    except Exception:
        return False


def run(argv: list[str]) -> int:
    flags, overrides = split_overrides(argv)
    policy = BudgetPolicy.load()

    ap = argparse.ArgumentParser(prog="ga launch", add_help=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provider", default="skypilot", help="launcher: skypilot|local")
    ap.add_argument("--cloud", default=None, help="cloud provider (runpod|lambda|modal)")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--max-cost-usd", type=float, default=policy.max_daily_cost_usd)
    ap.add_argument("--data-class", default="synthetic")
    ap.add_argument("--repo-url", default="${GUARDIAN_REPO_URL}")
    ap.add_argument("--yes", action="store_true", help="confirm a REAL (money-spending) launch")
    args = ap.parse_args(flags)

    cfg = to_container(compose_config(overrides))
    experiment = cfg.get("experiment")
    if not experiment:
        console.print("[red]No experiment.[/red] Pass +exp=<name> (e.g. +exp=arithmetic_catapult).")
        return 2

    sweep = cfg.get("sweep") or {}
    axes = {k: list(v) for k, v in (sweep.get("axes") or {}).items()}
    seeds = list(sweep.get("seeds") or [int(cfg.get("seed", 0))])
    sweep_name = sweep.get("name") if sweep.get("name") not in (None, "none") else None
    gpu = args.gpu or sweep.get("gpu", "l40s")
    cloud = args.cloud or sweep.get("provider") or (policy.allowed_providers[0] if policy.allowed_providers else "runpod")
    hours_per_job = float(sweep.get("hours_per_job", 0.25))

    fixed_overrides = [o for o in overrides if not o.split("=")[0].lstrip("+~") in ("exp", "sweep")]

    spec = LaunchSpec(
        experiment=experiment,
        overrides=fixed_overrides,
        provider=cloud,
        gpu=gpu,
        num_gpus=args.num_gpus,
        git_sha=current_sha(),
        repo_url=args.repo_url,
        sweep_name=sweep_name,
        sweep_axes=axes,
        seeds=seeds,
        hours_per_job=hours_per_job,
    )
    per_job = estimate_cost(1, hours_per_job=hours_per_job, gpu=gpu)
    total = round(per_job * spec.grid_size(), 2)
    marker_key = f"{experiment}:{sweep_name or 'single'}"

    if args.dry_run:
        console.print(dry_run_text(spec, per_job, total, args.max_cost_usd))
        _record_dry_run(marker_key)
        return 0

    # ---- REAL launch path: gated ---------------------------------------- #
    from ..common.budget import BudgetGuard

    report = BudgetGuard(policy).preflight_launch(
        provider=cloud,
        data_class=args.data_class,
        estimated_cost_usd=total,
        max_cost_usd=args.max_cost_usd,
        git_dirty=git_info().dirty,
        git_sha=current_sha(),
        dry_run_done=_dry_run_done(marker_key),
    )
    _print_report(report)
    if not report.passed:
        console.print("[red]✗ budget/safety preflight failed — nothing launched.[/red]")
        return 1
    if not args.yes:
        console.print("[yellow]Preflight passed, but --yes was not provided. Nothing launched.[/yellow]")
        console.print("Re-run with [bold]--yes[/bold] to confirm a real (money-spending) launch.")
        return 1

    return _maybe_submit(spec)


def _maybe_submit(spec: LaunchSpec) -> int:
    import os

    yaml_text = render_task_yaml(spec)
    allow = os.environ.get("GUARDIAN_ALLOW_REAL_LAUNCH") == "1"
    try:
        import sky  # noqa: F401

        have_sky = True
    except Exception:
        have_sky = False

    if not (allow and have_sky):
        console.print("[yellow]Real submission is disabled in this environment.[/yellow]")
        console.print("Gates passed. To submit, install the cloud extra and opt in explicitly:")
        console.print("  [bold]uv sync --extra cloud[/bold]")
        console.print("  [bold]GUARDIAN_ALLOW_REAL_LAUNCH=1 ga launch --yes ...[/bold]")
        console.print("\n--- task YAML (write to sky_task.yaml, then `sky launch -y sky_task.yaml`) ---")
        console.print(yaml_text)
        return 0

    # Opt-in real submission.
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    console.print(f"[bold]submitting:[/bold] sky launch -y {path}")
    return subprocess.call(["sky", "launch", "-y", path])


def _print_report(report) -> None:
    from rich.table import Table

    t = Table(title=f"preflight: {report.target}")
    t.add_column("check")
    t.add_column("ok")
    t.add_column("detail")
    for c in report.checks:
        t.add_row(c.name, "✓" if c.passed else "✗", c.detail)
    console.print(t)
