"""Disorder and robustness analysis for TFIM."""
import numpy as np


def add_onsite_disorder(H, n, W):
    """
    Add random on-site energy disorder to BdG Hamiltonian.

    Disorder strength W: energies drawn from Uniform[-W, W].
    Maintains particle-hole symmetry.
    """
    H_disorder = H.copy()

    # Random on-site energies
    disorder = np.random.uniform(-W, W, n)

    # Add to particle sector, subtract from hole sector (maintains PH symmetry)
    H_disorder[:n, :n] += np.diag(disorder)
    H_disorder[n:, n:] -= np.diag(disorder)

    return H_disorder


def add_hopping_disorder(H, n, W):
    """Add disorder to hopping parameters (off-diagonal)."""
    H_disorder = H.copy()

    # Add random fluctuations to off-diagonal elements
    for i in range(n):
        for j in range(i+1, min(i+3, n)):  # NN and NNN only
            delta_hop = np.random.uniform(-W, W)
            H_disorder[i, j] += delta_hop
            H_disorder[j, i] += delta_hop  # Keep symmetric

    return H_disorder


def disorder_average_gap(lambda_1, lambda_2, W, n_realizations=100, n=100, disorder_type='onsite'):
    """
    Average energy gap over disorder realizations.

    Returns mean and standard deviation of gap.
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    H_clean = build_tfim_hamiltonian(n, lambda_1, lambda_2)
    gaps = []

    for _ in range(n_realizations):
        if disorder_type == 'onsite':
            H_dis = add_onsite_disorder(H_clean, n, W)
        elif disorder_type == 'hopping':
            H_dis = add_hopping_disorder(H_clean, n, W)
        else:
            raise ValueError(f"Unknown disorder_type: {disorder_type}")

        eigvals = np.linalg.eigvalsh(H_dis)
        eigvals.sort()
        gap = eigvals[1] - eigvals[0]
        gaps.append(gap)

    return np.mean(gaps), np.std(gaps)


def disorder_average_spectrum(lambda_1, lambda_2, W, n_realizations=50, n=100):
    """Get full distribution of eigenvalues under disorder."""
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    H_clean = build_tfim_hamiltonian(n, lambda_1, lambda_2)
    all_eigvals = []

    for _ in range(n_realizations):
        H_dis = add_onsite_disorder(H_clean, n, W)
        eigvals = np.linalg.eigvalsh(H_dis)
        all_eigvals.extend(eigvals)

    return np.array(all_eigvals)


def measure_zero_mode_survival(lambda_1, lambda_2, W_range, n_realizations=100, n=100, threshold=None):
    """
    Track how zero modes survive increasing disorder.

    Returns fraction of realizations with zero modes for each W.
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    # Auto-set threshold based on clean system gap
    if threshold is None:
        H_clean = build_tfim_hamiltonian(n, lambda_1, lambda_2)
        eigvals_clean = np.linalg.eigvalsh(H_clean)
        clean_gap = eigvals_clean[1] - eigvals_clean[0]
        threshold = max(0.02, 2.0 * clean_gap)  # Adaptive threshold

    results = {
        'W_values': W_range,
        'survival_prob': [],
        'mean_gap': [],
        'std_gap': [],
        'n_zero_modes': []
    }

    for W in W_range:
        n_with_zero = 0
        gaps = []
        zero_counts = []

        H_clean = build_tfim_hamiltonian(n, lambda_1, lambda_2)

        for _ in range(n_realizations):
            H_dis = add_onsite_disorder(H_clean, n, W)
            eigvals = np.linalg.eigvalsh(H_dis)
            eigvals.sort()

            # Count zero modes
            n_zero = np.sum(np.abs(eigvals) < threshold)
            zero_counts.append(n_zero)

            if n_zero > 0:
                n_with_zero += 1

            gaps.append(eigvals[1] - eigvals[0])

        survival = n_with_zero / n_realizations
        results['survival_prob'].append(survival)
        results['mean_gap'].append(np.mean(gaps))
        results['std_gap'].append(np.std(gaps))
        results['n_zero_modes'].append(np.mean(zero_counts))

        print(f"W={W:.3f}: survival={survival:.2%}, gap={np.mean(gaps):.4f}±{np.std(gaps):.4f}, avg_zeros={np.mean(zero_counts):.1f}")

    for key in ['survival_prob', 'mean_gap', 'std_gap', 'n_zero_modes']:
        results[key] = np.array(results[key])

    return results


def critical_disorder_strength(lambda_1, lambda_2, n=100, n_realizations=100):
    """
    Find critical disorder W_c where topological protection breaks.

    Uses bisection to find W where survival probability drops below 50%.
    """
    # Binary search for critical W
    W_low, W_high = 0.0, 2.0
    tolerance = 0.05

    while W_high - W_low > tolerance:
        W_mid = 0.5 * (W_low + W_high)

        result = measure_zero_mode_survival(lambda_1, lambda_2, [W_mid], n_realizations, n)
        survival = result['survival_prob'][0]

        if survival > 0.5:
            W_low = W_mid
        else:
            W_high = W_mid

    W_c = 0.5 * (W_low + W_high)
    print(f"\nCritical disorder strength: W_c ≈ {W_c:.3f}")
    return W_c
