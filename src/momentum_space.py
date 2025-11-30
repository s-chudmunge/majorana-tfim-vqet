"""Momentum-space analysis and band structure calculations."""
import numpy as np


def build_bdg_hamiltonian_k(k, lambda_1, lambda_2, mu=0.0):
    """
    BdG Hamiltonian in momentum space (2×2 for single-band model).

    H(k) = [[ε(k), Δ(k)], [-Δ(k), -ε(k)]]

    Eigenvalues: E_±(k) = ±√[ε(k)² + Δ(k)²]

    Parameters
    ----------
    k : float
        Momentum
    lambda_1 : float
        Nearest-neighbor coupling
    lambda_2 : float
        Next-nearest-neighbor coupling
    mu : float, optional
        Chemical potential (default: 0.0)
    """
    epsilon_k = mu + 2 * lambda_1 * np.cos(k) + 2 * lambda_2 * np.cos(2*k)
    delta_k = -2 * lambda_1 * np.sin(k) - 2 * lambda_2 * np.sin(2*k)

    H_k = np.array([
        [epsilon_k, delta_k],
        [-delta_k, -epsilon_k]
    ])

    return H_k


def compute_band_structure(lambda_1, lambda_2, n_k=200, mu=0.0):
    """
    Compute BdG band structure E(k).

    Returns k values and upper/lower band energies.
    """
    k_vals = np.linspace(-np.pi, np.pi, n_k)
    E_plus = np.zeros(n_k)
    E_minus = np.zeros(n_k)

    for i, k in enumerate(k_vals):
        H_k = build_bdg_hamiltonian_k(k, lambda_1, lambda_2, mu)
        eigvals = np.linalg.eigvalsh(H_k)
        E_minus[i] = eigvals[0]
        E_plus[i] = eigvals[1]

    return k_vals, E_plus, E_minus


def find_gap_closing_points(lambda_1, lambda_2, n_k=500):
    """
    Find momentum values where gap closes (|E(k)| < tol).

    These are the topological phase transition points.
    """
    k_vals, E_plus, E_minus = compute_band_structure(lambda_1, lambda_2, n_k)

    # Gap closes when E_plus crosses zero (since E_minus = -E_plus by PH symmetry)
    gap_closing = []
    tol = 0.01

    for i, k in enumerate(k_vals):
        if abs(E_plus[i]) < tol:
            gap_closing.append(k)

    return np.array(gap_closing)


def compute_dispersion_components(lambda_1, lambda_2, n_k=200, mu=0.0):
    """
    Compute ε(k) and Δ(k) separately for visualization.

    Useful for understanding which term dominates in different regions.
    """
    k_vals = np.linspace(-np.pi, np.pi, n_k)

    epsilon = mu + 2 * lambda_1 * np.cos(k_vals) + 2 * lambda_2 * np.cos(2*k_vals)
    delta = -2 * lambda_1 * np.sin(k_vals) - 2 * lambda_2 * np.sin(2*k_vals)

    return k_vals, epsilon, delta


def trace_phase_transition(lambda_2_range, lambda_1=1.0, n_k=200, mu=0.0):
    """
    Track band structure evolution through phase transition.

    Fixes λ₁ and varies λ₂ to see how gap opens/closes.
    Returns gap values at k=0 and k=π for each λ₂.
    """
    gaps_at_zero = []
    gaps_at_pi = []
    min_gaps = []

    for l2 in lambda_2_range:
        k_vals, E_plus, _ = compute_band_structure(lambda_1, l2, n_k)

        # Gap at specific k points
        idx_zero = np.argmin(np.abs(k_vals))
        idx_pi = np.argmin(np.abs(k_vals - np.pi))

        gaps_at_zero.append(E_plus[idx_zero])
        gaps_at_pi.append(E_plus[idx_pi])

        # Minimum gap (bulk gap)
        min_gaps.append(np.min(np.abs(E_plus)))

    return {
        'lambda_2': np.array(lambda_2_range),
        'gap_at_k0': np.array(gaps_at_zero),
        'gap_at_kpi': np.array(gaps_at_pi),
        'bulk_gap': np.array(min_gaps)
    }


def compute_berry_phase(lambda_1, lambda_2, n_k=100):
    """
    Compute Berry phase around Brillouin zone.

    For 1D systems: γ = i ∮ dk ⟨u(k)|∂_k|u(k)⟩
    Related to winding number.
    """
    k_vals = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    dk = k_vals[1] - k_vals[0]

    # Get eigenvectors
    eigvecs = []
    for k in k_vals:
        H_k = build_bdg_hamiltonian_k(k, lambda_1, lambda_2)
        _, evecs = np.linalg.eigh(H_k)
        eigvecs.append(evecs[:, 0])  # Lower band

    eigvecs = np.array(eigvecs)

    # Compute Berry connection: A(k) = i⟨u(k)|∂_k|u(k)⟩
    berry_connection = []
    for i in range(n_k):
        u_k = eigvecs[i]
        u_k_next = eigvecs[(i+1) % n_k]

        # Numerical derivative
        overlap = np.vdot(u_k, u_k_next)
        A_k = np.angle(overlap) / dk
        berry_connection.append(A_k)

    # Integrate to get Berry phase
    berry_phase = np.sum(berry_connection) * dk

    return berry_phase % (2*np.pi)
