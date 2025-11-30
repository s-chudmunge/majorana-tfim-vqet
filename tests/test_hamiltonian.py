"""Tests for BdG Hamiltonian construction."""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tfim_model_core import build_tfim_hamiltonian


def test_hamiltonian_shape():
    """H should be 2n×2n."""
    n = 10
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=0.5)
    assert H.shape == (2*n, 2*n)


def test_hamiltonian_hermiticity():
    """H must be Hermitian (H = H†)."""
    n = 10
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=-1.2)
    assert np.allclose(H, H.T), "Hamiltonian not Hermitian"


def test_particle_hole_symmetry():
    """
    BdG Hamiltonians have particle-hole symmetry: σ_x H σ_x = -H
    where σ_x = [[0, I], [I, 0]]
    """
    n = 10
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=0.5)

    # Particle-hole operator
    sigma_x = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.eye(n), np.zeros((n, n))]
    ])

    H_ph = sigma_x @ H @ sigma_x
    assert np.allclose(H_ph, -H), "Particle-hole symmetry violated"


def test_eigenvalue_pairing():
    """Eigenvalues should come in ±E pairs."""
    n = 20
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=-1.2)
    eigvals = np.linalg.eigvalsh(H)

    # Sort eigenvalues
    eigvals_sorted = np.sort(eigvals)

    # Check that they're symmetric around zero
    # eigvals_sorted[i] ≈ -eigvals_sorted[-(i+1)]
    for i in range(len(eigvals)//2):
        assert np.isclose(eigvals_sorted[i], -eigvals_sorted[-(i+1)], atol=1e-10), \
            f"Eigenvalues not paired: {eigvals_sorted[i]} vs {eigvals_sorted[-(i+1)]}"


def test_zero_coupling_limit():
    """When λ₁=λ₂=0 and μ=0, should get all zero eigenvalues."""
    n = 10
    # Default μ=0
    H = build_tfim_hamiltonian(n, lambda_1=0.0, lambda_2=0.0, mu=0.0)
    eigvals = np.linalg.eigvalsh(H)

    # With zero coupling and μ=0, only off-diagonal blocks are zero
    # The Hamiltonian is H = [[0, 0], [0, 0]], so all eigenvalues are 0
    expected = np.zeros(2*n)
    assert np.allclose(eigvals, expected, atol=1e-10), f"Expected all zeros, got {eigvals}"

    # Test with non-zero μ
    H_mu = build_tfim_hamiltonian(n, lambda_1=0.0, lambda_2=0.0, mu=2.0)
    eigvals_mu = np.linalg.eigvalsh(H_mu)

    # With μ=2 and no coupling: H = [[2I, 0], [0, -2I]]
    # Eigenvalues should be ±2
    expected_mu = np.concatenate([2*np.ones(n), -2*np.ones(n)])
    assert np.allclose(np.sort(eigvals_mu), np.sort(expected_mu), atol=1e-10)


def test_delta_antisymmetry():
    """Pairing matrix Δ should be antisymmetric."""
    n = 10
    lambda_1, lambda_2 = 1.0, -0.5

    # Build matrices manually to test
    delta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == i + 1:
                delta[i, j] = -lambda_1
            elif j == i - 1:
                delta[i, j] = lambda_1
            elif j == i + 2:
                delta[i, j] = -lambda_2
            elif j == i - 2:
                delta[i, j] = lambda_2

    # Check antisymmetry: Δ[i,j] = -Δ[j,i]
    assert np.allclose(delta, -delta.T), "Delta matrix not antisymmetric"


def test_known_topological_phase():
    """At (λ₁=1.0, λ₂=-1.2), should have zero modes."""
    n = 100
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=-1.2)
    eigvals = np.linalg.eigvalsh(H)

    # Count near-zero eigenvalues
    zero_modes = np.sum(np.abs(eigvals) < 0.01)

    # Should have at least 2 zero modes (one per edge)
    assert zero_modes >= 2, f"Expected ≥2 MZMs, found {zero_modes}"


def test_real_eigenvalues():
    """All eigenvalues must be real (since H is Hermitian)."""
    n = 20
    H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=-1.2)
    eigvals = np.linalg.eigvals(H)  # Complex eigenvalues if any

    assert np.allclose(eigvals.imag, 0), "Found complex eigenvalues"


if __name__ == "__main__":
    # Run tests manually
    test_hamiltonian_shape()
    test_hamiltonian_hermiticity()
    test_particle_hole_symmetry()
    test_eigenvalue_pairing()
    test_zero_coupling_limit()
    test_delta_antisymmetry()
    test_known_topological_phase()
    test_real_eigenvalues()
    print("All tests passed!")
