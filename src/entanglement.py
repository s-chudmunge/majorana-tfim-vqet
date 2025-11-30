"""
Entanglement entropy calculations for TFIM.

For free fermions, entanglement can be computed efficiently
using the correlation matrix method.

IMPORTANT LIMITATIONS FOR TOPOLOGICAL DIAGNOSTICS:
===================================================

1. ENTANGLEMENT ENTROPY (TEE) IS NOT A RELIABLE TOPOLOGICAL DIAGNOSTIC IN 1D

   - Topological entanglement entropy formulas (Kitaev-Preskill, Levin-Wen)
     are designed for 2D systems where area-law terms can be cleanly subtracted

   - In 1D: S ~ (c/3)log(L) + γ where γ includes:
     * Boundary effects
     * Finite-size corrections
     * Non-universal contributions
     * NOT a clean topological constant

   - Observed γ ~ -20 to -25 (non-physical for TEE, which should be O(1))

2. ENTANGLEMENT SPECTRUM (ES) DEGENERACY FAILS IN 1D FREE FERMIONS

   - ES structure determined by single-particle modes, not topology
   - Even with PBC: all phases show same degeneracy (#degen = 19 everywhere)
   - Min ES gap varies by < 1000× across topological and trivial phases
   - Pattern does NOT correlate with winding number phase boundaries

3. EDGE MODE HYBRIDIZATION

   - With OBC: edge MZMs hybridize with amplitude ~ exp(-L/ξ)
   - Lifts ground state degeneracy
   - Entanglement signatures contaminated by finite-size effects

VALID USES OF THIS MODULE:
===========================

✓ Central charge extraction at critical points (S ~ (c/3)log(L))
✓ Correlation length estimates
✓ General quantum information properties (mutual information, etc.)
✓ CFT analysis at phase transitions

✗ DO NOT USE for identifying topological phases
  → Use winding number (topological_invariants.py) instead

See RESULTS_SUMMARY.md section "What Doesn't Work" for detailed analysis.
"""
import numpy as np
from scipy.linalg import logm


def compute_correlation_matrix(eigvecs, n, n_occupied):
    """
    Compute fermion correlation matrix from ground state.

    For BdG Hamiltonian, correlation matrix C_ij = <c_i† c_j>
    """
    # Get occupied states (negative energy eigenstates)
    # For particle-hole symmetric BdG, ground state has all E<0 occupied

    # Extract particle sector of occupied states
    occupied_states = eigvecs[:n, :n_occupied]

    # Correlation matrix
    C = occupied_states @ occupied_states.conj().T

    return C


def entanglement_entropy_from_correlation(C_A):
    """
    Compute von Neumann entropy from correlation matrix.

    For free fermions:
    S = -Tr[C log C + (1-C)log(1-C)]

    where C is the correlation matrix restricted to subsystem A.
    """
    # Eigenvalues of correlation matrix are occupation numbers
    eigvals = np.linalg.eigvalsh(C_A)

    # Clip to avoid log(0)
    eigvals = np.clip(eigvals, 1e-15, 1 - 1e-15)

    # von Neumann entropy
    S = -np.sum(eigvals * np.log(eigvals) + (1 - eigvals) * np.log(1 - eigvals))

    return S


def compute_entanglement_entropy(lambda_1, lambda_2, n, l_A=None):
    """
    Compute entanglement entropy for bipartition.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    n : int
        System size
    l_A : int, optional
        Size of subsystem A. Default: n//2

    Returns
    -------
    S : float
        von Neumann entropy
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    if l_A is None:
        l_A = n // 2

    # Build and diagonalize Hamiltonian
    H = build_tfim_hamiltonian(n, lambda_1, lambda_2)
    eigvals, eigvecs = np.linalg.eigh(H)

    # Number of occupied states (negative energy)
    n_occupied = np.sum(eigvals < 0)

    # Full correlation matrix
    C = compute_correlation_matrix(eigvecs, n, n_occupied)

    # Restrict to subsystem A (first l_A sites)
    C_A = C[:l_A, :l_A]

    # Compute entropy
    S = entanglement_entropy_from_correlation(C_A)

    return S


def extract_topological_entropy(lambda_1, lambda_2, sizes, l_A_fraction=0.5):
    """
    Extract topological entanglement entropy γ from scaling.

    For 1D systems: S = (c/3)log(L) + γ + o(1/L)

    Fit to extract γ.
    """
    from scipy.optimize import curve_fit

    entropies = []

    for n in sizes:
        l_A = int(n * l_A_fraction)
        S = compute_entanglement_entropy(lambda_1, lambda_2, n, l_A)
        entropies.append(S)

    entropies = np.array(entropies)
    sizes = np.array(sizes)

    # Fit: S = a*log(L) + gamma
    def scaling_form(L, a, gamma):
        return a * np.log(L) + gamma

    try:
        popt, pcov = curve_fit(scaling_form, sizes, entropies)
        a, gamma = popt
        gamma_err = np.sqrt(pcov[1, 1])

        # c = 3a for 1D CFT
        central_charge = 3 * a

        return {
            'gamma': gamma,
            'gamma_err': gamma_err,
            'central_charge': central_charge,
            'scaling_coeff': a,
            'entropies': entropies,
            'sizes': sizes
        }
    except:
        return {
            'gamma': np.nan,
            'gamma_err': np.nan,
            'central_charge': np.nan,
            'scaling_coeff': np.nan,
            'entropies': entropies,
            'sizes': sizes
        }


def scan_topological_entropy_map(lambda1_range, lambda2_range, n=100, l_A=None):
    """
    Compute γ across parameter space.

    This reveals topological phase boundaries.
    """
    gamma_map = np.zeros((len(lambda1_range), len(lambda2_range)))

    if l_A is None:
        l_A = n // 2

    for i, l1 in enumerate(lambda1_range):
        for j, l2 in enumerate(lambda2_range):
            S = compute_entanglement_entropy(l1, l2, n, l_A)
            gamma_map[i, j] = S

        if i % max(1, len(lambda1_range)//10) == 0:
            print(f"Progress: {i}/{len(lambda1_range)}")

    return gamma_map


def compare_entanglement_phases(test_points, sizes=[20, 30, 40, 60, 80, 100]):
    """
    Compare entanglement scaling across different phases.

    Parameters
    ----------
    test_points : list of tuples
        List of (λ₁, λ₂, label) to compare
    """
    results = {}

    for lambda_1, lambda_2, label in test_points:
        print(f"\n{label}: (λ₁={lambda_1}, λ₂={lambda_2})")
        result = extract_topological_entropy(lambda_1, lambda_2, sizes)

        if not np.isnan(result['gamma']):
            print(f"  γ = {result['gamma']:.4f} ± {result['gamma_err']:.4f}")
            print(f"  c = {result['central_charge']:.4f}")
        else:
            print(f"  Fit failed")

        results[label] = result

    return results


def area_law_check(lambda_1, lambda_2, n=100):
    """
    Check if entanglement follows area law.

    Scan different subsystem sizes and check S(l_A).
    """
    l_A_values = np.arange(5, n//2, 5)
    entropies = []

    for l_A in l_A_values:
        S = compute_entanglement_entropy(lambda_1, lambda_2, n, l_A)
        entropies.append(S)

    return l_A_values, np.array(entropies)


def mutual_information(lambda_1, lambda_2, n, l_A, l_B):
    """
    Compute mutual information I(A:B) = S_A + S_B - S_AB.

    Measures quantum correlations between regions.
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    # Build Hamiltonian
    H = build_tfim_hamiltonian(n, lambda_1, lambda_2)
    eigvals, eigvecs = np.linalg.eigh(H)
    n_occupied = np.sum(eigvals < 0)
    C = compute_correlation_matrix(eigvecs, n, n_occupied)

    # S_A
    C_A = C[:l_A, :l_A]
    S_A = entanglement_entropy_from_correlation(C_A)

    # S_B (region from l_A to l_A+l_B)
    C_B = C[l_A:l_A+l_B, l_A:l_A+l_B]
    S_B = entanglement_entropy_from_correlation(C_B)

    # S_AB (combined region)
    C_AB = C[:l_A+l_B, :l_A+l_B]
    S_AB = entanglement_entropy_from_correlation(C_AB)

    # Mutual information
    I_AB = S_A + S_B - S_AB

    return I_AB, S_A, S_B, S_AB


def compute_entanglement_spectrum(lambda_1, lambda_2, L=200, mu=0.0, pbc=False, l_A=None):
    """
    Compute entanglement spectrum (ES) for 1D Majorana chain.

    The ES is defined as ξᵢ = -log(λᵢ) where λᵢ are eigenvalues
    of the correlation matrix for a bipartition.

    In topological phases, the low-lying ES shows characteristic degeneracy.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    L : int
        System size
    mu : float, optional
        Chemical potential
    pbc : bool, optional
        Use periodic boundary conditions (removes edge hybridization)
    l_A : int, optional
        Subsystem size (default: L//2)

    Returns
    -------
    entanglement_spectrum : array
        ES eigenvalues ξᵢ, sorted ascending
    degeneracy_gap : float
        Smallest gap between adjacent ES levels (measures degeneracy)
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    # Build and diagonalize Hamiltonian
    H = build_tfim_hamiltonian(L, lambda_1, lambda_2, mu=mu, pbc=pbc)
    eigvals, eigvecs = np.linalg.eigh(H)
    n_occupied = np.sum(eigvals < 0)
    C = compute_correlation_matrix(eigvecs, L, n_occupied)

    # Bipartition: cut at specified size or middle
    if l_A is None:
        l_A = L // 2
    C_A = C[:l_A, :l_A]

    # Correlation matrix eigenvalues
    lambda_corr = np.linalg.eigvalsh(C_A)

    # Clip to valid range [0, 1]
    lambda_corr = np.clip(lambda_corr, 1e-15, 1 - 1e-15)

    # Entanglement spectrum: ξ = -log(λ)
    # For fermions, use both λ and 1-λ contributions
    xi_lower = -np.log(lambda_corr)
    xi_upper = -np.log(1 - lambda_corr)

    # Combine and sort
    xi_full = np.concatenate([xi_lower, xi_upper])
    xi_full = np.sort(xi_full)

    # Measure degeneracy: smallest gap in low-lying spectrum
    n_check = min(10, len(xi_full) - 1)
    if n_check > 0:
        gaps = np.diff(xi_full[:n_check])
        degeneracy_gap = np.min(gaps)
    else:
        degeneracy_gap = np.inf

    return xi_full, degeneracy_gap


def compute_es_degeneracy_measure(lambda_1, lambda_2, L=200, mu=0.0, pbc=False):
    """
    Compute degeneracy measure from entanglement spectrum.

    In topological phase: low-lying ES is nearly degenerate → small gaps
    In trivial phase: ES has large gaps → no degeneracy

    Returns
    -------
    degeneracy : float
        Inverse of smallest ES gap (large → degenerate)
    n_degenerate_pairs : int
        Number of nearly degenerate pairs (gap < threshold)
    min_gap : float
        Smallest gap in low-lying ES
    """
    xi_full, min_gap = compute_entanglement_spectrum(lambda_1, lambda_2, L, mu, pbc=pbc)

    # Count degenerate pairs (gap < 0.1)
    n_check = min(20, len(xi_full) - 1)
    gaps = np.diff(xi_full[:n_check])
    n_degenerate = np.sum(gaps < 0.1)

    # Degeneracy measure: 1/min_gap (saturate at large value)
    if min_gap > 1e-6:
        degeneracy = min(100.0, 1.0 / min_gap)
    else:
        degeneracy = 100.0

    return degeneracy, n_degenerate, min_gap
