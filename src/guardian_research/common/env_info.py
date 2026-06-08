"""Capture the exact provenance of a run: git SHA, dirty state, versions, device.

The git SHA requirement is a safety gate, not a nicety: cloud jobs must run from
an exact commit (see common/budget.py and docs/security.md). We surface the SHA
and dirty flag everywhere so a result can always be traced back to code.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from functools import lru_cache

from .paths import repo_root
from .schemas import EnvInfo, GitInfo


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


@lru_cache(maxsize=1)
def git_info() -> GitInfo:
    sha = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    dirty = bool(status) if status is not None else True
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return GitInfo(sha=sha, dirty=dirty, branch=branch)


def _torch_device() -> tuple[str, bool, str | None]:
    """Return (device, cuda_available, torch_version)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True, torch.__version__
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", False, torch.__version__
        return "cpu", False, torch.__version__
    except Exception:
        return "cpu", False, None


@lru_cache(maxsize=1)
def env_info() -> EnvInfo:
    device, cuda, torch_version = _torch_device()
    return EnvInfo(
        python=sys.version.split()[0],
        torch=torch_version,
        platform=platform.platform(),
        hostname=socket.gethostname(),
        device=device,
        cuda_available=cuda,
    )


def is_clean_git_tree() -> bool:
    return not git_info().dirty


def current_sha() -> str:
    return git_info().sha
