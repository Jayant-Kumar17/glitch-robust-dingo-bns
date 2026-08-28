"""Verification suite for ObservatoryNode rolling chunk stream.

Plain runnable script (also pytest-compatible test function names).
Does not modify any core src modules.
"""

from __future__ import annotations

import numpy as np

from adapt.node_stream import ObservatoryNode

SAMPLE_RATE = 4096.0
STRIDE_SECONDS = 2.0
N_RAW = 40960  # 10 s at 4096 Hz
CHUNK_SAMPLES = int(round(STRIDE_SECONDS * SAMPLE_RATE))  # 8192
REQUIRED_KEYS = ("detector", "strain", "is_simulated", "sample_rate")


def _make_node(seed: int = 1) -> ObservatoryNode:
    raw = np.zeros(N_RAW, dtype=np.float64)
    return ObservatoryNode(
        "H1",
        raw,
        sample_rate=SAMPLE_RATE,
        stride_seconds=STRIDE_SECONDS,
        seed=seed,
    )


def _assert_required_keys(payload: dict) -> None:
    for key in REQUIRED_KEYS:
        assert key in payload, f"Error: missing payload key '{key}'"


def test_observatory_node_real_data():
    """First chunk from a 10 s dummy array is real (not simulated)."""
    print("\n[Test] test_observatory_node_real_data...")
    node = _make_node(seed=1)
    chunk = node.get_next_chunk()
    _assert_required_keys(chunk)
    assert chunk["is_simulated"] is False, "Error: first chunk should be real data!"
    assert chunk["detector"] == "H1"
    assert chunk["strain"].shape == (CHUNK_SAMPLES,)
    assert chunk["sample_rate"] == SAMPLE_RATE
    assert "start_sample" in chunk and chunk["start_sample"] == 0
    assert "stride_seconds" in chunk and chunk["stride_seconds"] == STRIDE_SECONDS
    print("  => PASS")


def test_observatory_node_fallback_to_simulated():
    """After exhausting real samples, chunks stay fixed-shape and become simulated."""
    print("\n[Test] test_observatory_node_fallback_to_simulated...")
    node = _make_node(seed=2)
    # 5 exact real chunks (40960 / 8192), then 1 fully simulated.
    n_calls = 6
    payloads = []
    for _ in range(n_calls):
        payloads.append(node.get_next_chunk())

    shapes = [p["strain"].shape for p in payloads]
    assert all(s == (CHUNK_SAMPLES,) for s in shapes), (
        f"Error: chunk shape not constant; got {shapes}"
    )
    assert payloads[0]["is_simulated"] is False, "Error: first chunk should be real!"
    assert payloads[-1]["is_simulated"] is True, (
        "Error: chunk after exhaust should be simulated!"
    )
    _assert_required_keys(payloads[-1])
    assert payloads[-1]["detector"] == "H1"
    assert payloads[-1]["sample_rate"] == SAMPLE_RATE
    print(f"  shapes={shapes[0]} x {n_calls}, last is_simulated=True")
    print("  => PASS")


def run_node_stream_tests():
    print("--- Starting ADAPT ObservatoryNode Verification Tests ---")
    test_observatory_node_real_data()
    test_observatory_node_fallback_to_simulated()
    print("\nALL OBSERVATORY NODE STREAM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_node_stream_tests()
