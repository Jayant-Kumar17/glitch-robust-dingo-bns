"""Repository-root paths, independent of the process working directory."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str | Path) -> Path:
    """Join ``parts`` onto the repository root and return an absolute path."""
    return REPO_ROOT.joinpath(*parts)


def resolve_under_repo(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve ``path`` against the repository root, not the process cwd."""
    root = repo_root or REPO_ROOT
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()
