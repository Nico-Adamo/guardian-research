"""`ga collect [run_id]` — ingest run artifacts into the local control plane.

Local runs already live under ``runs/``; this command verifies/ingests them and
(for remote workers) documents the rsync/object-store pull that brings results
home. Result *ingestion* itself is just reading ``results.json`` — that is the
whole portability contract.
"""

from __future__ import annotations

from ..common.logging import console
from ..common.paths import runs_dir
from ..tracking.ingest import load_runs

NAME = "collect"
HELP = "Ingest/verify run artifacts locally: ga collect [run_id]"


def run(argv: list[str]) -> int:
    run_id = argv[0] if argv else None
    runs = load_runs()
    if run_id:
        runs = [r for r in runs if r.run_id == run_id or r.run_id.endswith(run_id)]

    console.print(f"[bold]runs/ root:[/bold] {runs_dir()}")
    console.print(f"ingestable runs: {len(runs)}")
    for r in runs[:20]:
        console.print(f"  - {r.experiment}/{r.run_id} [{r.status}] sha={r.git.sha[:10]} "
                      f"final_hard_acc={r.final_metrics.get('final_hard_acc')}")
    console.print(
        "\n[dim]Remote workers: results are pulled via rsync or an object store, e.g.:[/dim]\n"
        "  GUARDIAN_ARTIFACT_URI=s3://bucket/run-xyz  # set on the worker to upload\n"
        "  aws s3 sync s3://bucket/run-xyz runs/<exp>/<run_id>  # pull locally, then `ga collect`"
    )
    return 0
