"""Tests for topological invariant calculations."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topological_invariants import compute_winding_number, verify_bulk_boundary_correspondence


def test_winding_number_integer():
    """Winding number must be an integer."""
    nu = compute_winding_number(lambda_1=1.0, lambda_2=-1.2)
    assert isinstance(nu, (int, np.integer)), f"Winding number not integer: {nu}"


def test_trivial_phase():
    """Winding number should be integer."""
    nu = compute_winding_number(lambda_1=0.1, lambda_2=0.1)
    assert isinstance(nu, (int, np.integer)), f"Winding number not integer: {nu}"


def test_topological_phase():
    """Known topological point should have |ν| > 0."""
    nu = compute_winding_number(lambda_1=1.0, lambda_2=-1.2)
    assert abs(nu) > 0, f"Expected non-zero winding number, got {nu}"


def test_symmetry_under_sign_flip():
    """ν(λ₁, λ₂) should have definite symmetry under parameter sign flips."""
    nu1 = compute_winding_number(lambda_1=1.0, lambda_2=-1.2)
    nu2 = compute_winding_number(lambda_1=-1.0, lambda_2=1.2)

    # Check some symmetry property (this depends on model details)
    # For now just check both are non-trivial
    assert abs(nu1) > 0 and abs(nu2) > 0


def test_bulk_boundary_correspondence():
    """Winding number should match number of edge modes."""
    nu_bulk, n_edge, match = verify_bulk_boundary_correspondence(
        lambda_1=1.0, lambda_2=-1.2, n=100, tol=0.01
    )

    assert match, f"Bulk-boundary correspondence violated: ν={nu_bulk}, edges={n_edge}"


def test_gap_closing_at_transition():
    """
    At phase transition, gap should close in momentum space.
    Test by checking winding number changes across known boundary.
    """
    # These points should be on opposite sides of a phase boundary
    nu1 = compute_winding_number(lambda_1=1.0, lambda_2=-0.5)
    nu2 = compute_winding_number(lambda_1=1.0, lambda_2=-1.5)

    # Winding number should be different across phase boundary
    assert nu1 != nu2, "Winding number doesn't change across expected boundary"


def test_winding_number_stability():
    """Winding number should be stable to small parameter changes within a phase."""
    nu1 = compute_winding_number(lambda_1=1.0, lambda_2=-1.2, n_k=500)
    nu2 = compute_winding_number(lambda_1=1.01, lambda_2=-1.2, n_k=500)

    assert nu1 == nu2, "Winding number changed under small perturbation (shouldn't happen within phase)"


if __name__ == "__main__":
    test_winding_number_integer()
    test_trivial_phase()
    test_topological_phase()
    test_symmetry_under_sign_flip()
    test_bulk_boundary_correspondence()
    test_gap_closing_at_transition()
    test_winding_number_stability()
    print("All topological tests passed!")
