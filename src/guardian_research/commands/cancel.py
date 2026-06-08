"""`ga cancel <run_id|job>` — cancel a cloud job (stub).

Cancellation talks to the launcher's job controller (e.g. `sky cancel`). Real
cloud jobs are not enabled in this scaffold, so this is an explicit, honest stub
rather than a silent no-op: it tells you exactly what it *would* run.
"""

from __future__ import annotations

from ..common.logging import console

NAME = "cancel"
HELP = "Cancel a cloud job (stub — prints the command it would run)"


def run(argv: list[str]) -> int:
    target = argv[0] if argv else "<job-id>"
    console.print("[yellow]cancel is a stub in this scaffold.[/yellow]")
    console.print("With the cloud extra installed and a live job, this would run:")
    console.print(f"  [bold]sky cancel {target}[/bold]   # or `sky jobs cancel {target}` for managed jobs")
    console.print("Local in-process runs (`ga train`) are cancelled with Ctrl-C.")
    return 0
