"""Repository root and cwd-independent path resolution for example drivers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    for _p in (
        REPO_ROOT / "examples",
        REPO_ROOT / "src",
        REPO_ROOT / "DINGO-BNS" / "dingo",
        REPO_ROOT,
    ):
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))


_ensure_sys_path()


def resolve_under_repo(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve ``path`` against the repository root, not the process cwd.

    Absolute paths are returned unchanged (after ``resolve()``). Relative
    paths, including ``.``, are joined onto ``repo_root`` so Appendix C
    commands such as ``--outdir results/...`` work from any working directory.
    """
    root = repo_root or REPO_ROOT
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()
