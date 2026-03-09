"""tests/test_metrics.py — unit tests for metric computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import EpisodeResult, EvalStats, compute_gap


def _make_results(n=10):
    return [
        EpisodeResult(
            total_reward=float(i),
            total_cost=float(100 - i),
            max_tardiness=float(i) / n,
            service_rate=float(i) / n,
            customers_served=i,
            steps=20,
        )
        for i in range(n)
    ]


def test_eval_stats_shape():
    results = _make_results()
    stats = EvalStats.from_results(results)
    assert stats.n_episodes == 10


def test_service_rate_bounds():
    results = _make_results()
    stats = EvalStats.from_results(results)
    assert 0.0 <= stats.mean_service_rate <= 1.0
    assert 0.0 <= stats.min_service_rate <= 1.0
    assert 0.0 <= stats.max_service_rate <= 1.0


def test_pct_full_service():
    results = _make_results(10)
    # Only the last result has service_rate = 9/10 < 1.0; none is exactly 1.0
    stats = EvalStats.from_results(results)
    assert stats.pct_full_service == 0.0


def test_compute_gap():
    assert compute_gap(110, 100) == pytest.approx(10.0)
    assert compute_gap(100, 100) == pytest.approx(0.0)


import pytest


def test_compute_gap_zero_baseline():
    result = compute_gap(50, 0)
    assert result == float("inf")
