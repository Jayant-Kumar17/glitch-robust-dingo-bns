"""Cwd-independent path helpers and archived paper artefacts."""

from __future__ import annotations

from adapt.paths import REPO_ROOT, resolve_under_repo

ARCHIVE_FILES = (
    "results/dingo_official_control/comparison_report.json",
    "results/dingo_official_control/control_summary.json",
    "results/dingo_official_control/poison_summary.json",
    "results/dingo_official_control/gated_glitchy_summary.json",
    "results/excision_honest/excision_report.json",
    "results/stress_test_excision_v1/results.csv",
    "results/stress_test_excision_v1/summary.json",
    "results/stress_test_synthetic_bns_v1/results.csv",
    "results/stress_test_synthetic_bns_v1/summary.json",
    "results/journal_method_hardening_v1/ablation_summary.json",
    "results/journal_method_hardening_v1/ablation_results.csv",
    "results/clean_gate_control_v1/results.csv",
    "results/clean_gate_control_v1/summary.json",
    "results/stress_real_glitches_v2/summary.json",
    "results/stress_real_glitches_v2/results.csv",
    "results/collapse_threshold_v1/summary.json",
    "results/collapse_threshold_v1/results.csv",
    "results/loudness_audit_v1/loudness.csv",
    "checkpoints/glitch_detector_v1/best_glitch_detector.pt",
    "LICENSE",
)


def test_repo_root_has_project_files() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "adapt").is_dir()
    assert (REPO_ROOT / "environment.yml").is_file()


def test_resolve_relative_ignores_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    got = resolve_under_repo("results/dingo_official_control")
    assert got == (REPO_ROOT / "results" / "dingo_official_control").resolve()


def test_no_machine_local_paths() -> None:
    roots = [REPO_ROOT / "examples", REPO_ROOT / "paper", REPO_ROOT / "src"]
    offenders = []
    for root in roots:
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if "/Users/" in text or "/home/" in text:
                offenders.append(str(py.relative_to(REPO_ROOT)))
    assert offenders == []


def test_archived_paper_summaries_present() -> None:
    missing = [rel for rel in ARCHIVE_FILES if not (REPO_ROOT / rel).is_file()]
    assert missing == []


def test_checkpoint_is_lightweight() -> None:
    ckpt = REPO_ROOT / "checkpoints" / "glitch_detector_v1" / "best_glitch_detector.pt"
    assert ckpt.stat().st_size < 10 * 1024 * 1024
