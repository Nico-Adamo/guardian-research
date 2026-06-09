"""`ga collect` — pull worker results home and ingest them.

Closes the cloud round-trip: a disposable worker uploads ``runs/`` to
``$GUARDIAN_ARTIFACT_URI/<git_sha>/runs/`` (see launchers/skypilot.py); this
command syncs that prefix back into the local ``runs/`` and then ingests.
Result *ingestion* is just reading ``results.json`` — the portability contract.

Supports ``s3://`` (aws cli), ``gs://`` (gsutil), and rsync/local targets. If the
needed CLI isn't installed, it prints the exact command instead of failing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

from ..common.env_info import current_sha
from ..common.logging import console
from ..common.paths import ensure_dir, runs_dir
from ..tracking.ingest import load_runs

NAME = "collect"
HELP = "Pull worker results home + ingest: ga collect [--from URI] [--sha SHA] [run_id]"


def _pull_cmd(uri: str, sha: str, dest: str) -> tuple[list[str], str | None]:
    src = f"{uri.rstrip('/')}/{sha}/runs/"
    if uri.startswith("s3://"):
        return ["aws", "s3", "sync", src, dest], "aws"
    if uri.startswith("gs://"):
        return ["gsutil", "-m", "rsync", "-r", src, dest], "gsutil"
    return ["rsync", "-az", src, dest], "rsync"


def run(argv: list[str]) -> int:
    import os

    ap = argparse.ArgumentParser(prog="ga collect", add_help=True)
    ap.add_argument("run_id", nargs="?", default=None)
    ap.add_argument("--from", dest="uri", default=os.environ.get("GUARDIAN_ARTIFACT_URI", ""))
    ap.add_argument("--sha", default=None, help="commit whose results to pull (default: HEAD)")
    ap.add_argument("--dry-run", action="store_true", help="print the sync command, do not run it")
    args = ap.parse_args(argv)

    if args.uri:
        sha = args.sha or current_sha()
        dest = str(ensure_dir(runs_dir())) + "/"
        cmd, tool = _pull_cmd(args.uri, sha, dest)
        printable = " ".join(cmd)
        if args.dry_run:
            console.print(f"[bold]would run:[/bold] {printable}")
        elif tool and shutil.which(tool) is None:
            console.print(f"[yellow]{tool} not installed[/yellow] — run this yourself to pull results:")
            console.print(f"  [bold]{printable}[/bold]")
        else:
            console.print(f"[bold]pulling:[/bold] {printable}")
            rc = subprocess.call(cmd)
            if rc != 0:
                console.print(f"[red]sync exited {rc}[/red] (check creds/path); ingesting whatever is local")
    else:
        console.print("[dim]No --from / GUARDIAN_ARTIFACT_URI set; ingesting local runs only.[/dim]")

    runs = load_runs()
    if args.run_id:
        runs = [r for r in runs if r.run_id == args.run_id or r.run_id.endswith(args.run_id)]
    console.print(f"[bold]runs/ root:[/bold] {runs_dir()}")
    console.print(f"ingestable runs: {len(runs)}")
    for r in runs[:20]:
        console.print(f"  - {r.experiment}/{r.run_id} [{r.status}] sha={r.git.sha[:10]} "
                      f"final_hard_acc={r.final_metrics.get('final_hard_acc')}")
    return 0
