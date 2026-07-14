"""Tests for the from-scratch PSI drift module."""
import numpy as np

from src.api.drift import classify_psi, load_reference, psi_from_reference


def test_load_reference_has_expected_keys():
    ref = load_reference()
    assert "bin_edges" in ref
    assert "bin_proportions" in ref
    assert len(ref["bin_edges"]) == len(ref["bin_proportions"]) + 1
    assert abs(sum(ref["bin_proportions"]) - 1.0) < 1e-6


def test_psi_not_significant_when_current_approximates_reference_shape():
    # A normal approximation to the (skewed) reference distribution won't
    # match bin-for-bin, but shouldn't register as a *significant* shift.
    ref = load_reference()
    rng = np.random.default_rng(1)
    sample = np.clip(rng.normal(ref["reference_mean"], ref["reference_std"], 2000), 0, 1)
    value = psi_from_reference(sample)
    assert value < 0.25


def test_psi_large_when_current_is_a_degenerate_low_distribution():
    current = np.full(200, 0.01)
    value = psi_from_reference(current)
    assert value > 0.25


def test_classify_psi_bands():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.30) == "significant"


def test_classify_psi_boundary_values():
    assert classify_psi(0.0999) == "stable"
    assert classify_psi(0.10) == "moderate"
    assert classify_psi(0.2499) == "moderate"
    assert classify_psi(0.25) == "significant"
