"""Verification suite for ADAPTPipelineManager.

Plain runnable script (also pytest-compatible test function names).
Does not modify any core src modules.
"""

from __future__ import annotations

import numpy as np

# Public class name is ADAPTPipelineManager (project acronym). Alias matches
# the ADATP* spelling used in the test request without changing src/.
from adapt.pipeline_manager import ADAPTPipelineManager as ADATPPipelineManager
from adapt.noise_analytics import GlobalNoiseHub
from adapt.router import MatchedFilterRouter


def test_pipeline_manager_initialization():
    """Default construction falls back to H1 + L1 hub and mass classifier."""
    manager = ADATPPipelineManager()
    assert manager is not None
    assert isinstance(manager.noise_hub, GlobalNoiseHub)
    assert isinstance(manager.classifier, MatchedFilterRouter)
    assert manager.noise_hub.detectors == ["H1", "L1"]
    print("  test_pipeline_manager_initialization => PASS")


def test_prepare_dingo_context_clean_data():
    """Finite profile vector marks the DINGO stub as valid and ready."""
    manager = ADATPPipelineManager()
    clean = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
    context = manager.prepare_dingo_context(clean)
    assert context["is_valid"] is True
    assert context["dingo_ready"] is True
    assert context["n_nan"] == 0
    assert context["n_inf"] == 0
    assert np.all(np.isfinite(context["context_vector"]))
    print("  test_prepare_dingo_context_clean_data => PASS")


def test_prepare_dingo_context_with_nans():
    """NaNs are caught, invalidate the context, and are zeroed in a copy."""
    manager = ADATPPipelineManager()
    dirty = np.ones(16, dtype=np.float64)
    dirty[3] = np.nan
    dirty[7] = np.nan
    original = dirty.copy()

    context = manager.prepare_dingo_context(dirty)

    assert context["is_valid"] is False
    assert context["dingo_ready"] is False
    assert context["n_nan"] == 2
    assert np.all(np.isfinite(context["context_vector"]))
    assert context["context_vector"][3] == 0.0
    assert context["context_vector"][7] == 0.0
    # Caller array must not be mutated.
    assert np.isnan(dirty[3]) and np.isnan(dirty[7])
    assert np.array_equal(np.isnan(dirty), np.isnan(original))
    print("  test_prepare_dingo_context_with_nans => PASS")


def run_pipeline_tests():
    print("--- Starting ADAPT Pipeline Manager Verification Tests ---")
    test_pipeline_manager_initialization()
    test_prepare_dingo_context_clean_data()
    test_prepare_dingo_context_with_nans()
    print("\nALL PIPELINE MANAGER TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_pipeline_tests()
