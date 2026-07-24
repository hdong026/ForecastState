"""Unit tests for capacity-constrained graph resolution."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastStateGraphResolution.arch.graph_resolution import (
    build_capacity_constrained_clusters,
    build_structural_graph,
    num_clusters_for_capacity,
    validate_assignment,
)


def _make_chain_adj(n: int = 20) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    return a


def test_capacity_constraint_and_deterministic():
    adj = _make_chain_adj(24)
    cap = 4
    m_expected = num_clusters_for_capacity(24, cap)
    a = build_capacity_constrained_clusters(adj, capacity=cap, seed=1, distance_mode="shortest_path")
    b = build_capacity_constrained_clusters(adj, capacity=cap, seed=1, distance_mode="shortest_path")
    assert a["num_clusters"] == m_expected
    assert np.array_equal(a["labels"], b["labels"])
    report = validate_assignment(a["labels"], capacity=cap)
    assert report["over_capacity_count"] == 0
    assert report["empty_cluster_count"] == 0
    assert report["max_cluster_size"] <= cap
    # structural graph shape
    a_str = build_structural_graph(a["P"], a["W"], a["C"])
    assert a_str.shape == (m_expected, m_expected)
    assert np.isfinite(a_str).all()


def test_capacity_one_identity():
    adj = _make_chain_adj(10)
    meta = build_capacity_constrained_clusters(adj, capacity=1, seed=0)
    assert meta["num_clusters"] == 10
    assert np.allclose(meta["C"], np.eye(10))
    assert np.allclose(meta["P"], np.eye(10))


if __name__ == "__main__":
    test_capacity_constraint_and_deterministic()
    test_capacity_one_identity()
    print("test_graph_resolution: OK")
