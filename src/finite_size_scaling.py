"""
Finite-size scaling analysis for TFIM.

Real systems are finite, but we want to understand infinite-system behavior.
Finite-size scaling lets us extrapolate from small systems (which we can
simulate) to the thermodynamic limit (what we care about physically).
"""
import numpy as np
from scipy.optimize import curve_fit


def compute_gap_vs_size(lambda_1, lambda_2, sizes, verbose=True, mu=0.0):
    """
    Compute energy gap for different system sizes.

    The BULK gap is the minimum energy to create a bulk excitation.
    In BdG systems with particle-hole symmetry, energies come in ±E pairs.
    The bulk gap is the minimum |E| for states that are NOT zero modes.

    The gap typically scales as: Δ(L) = Δ_∞ + a/L^α
    where Δ_∞ is the thermodynamic limit gap.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    sizes : list of int
        System sizes to compute
    verbose : bool
        Print results as we compute
    mu : float
        Chemical potential (default: 0.0)

    Returns
    -------
    gaps : ndarray
        BULK energy gaps for each size (minimum non-zero |E|)
    zero_energies : ndarray
        Energies of zero modes (for tracking hybridization)
    n_zero_modes : ndarray
        Number of zero modes for each size
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    gaps = []
    zero_energies_list = []
    n_zero_modes_list = []

    for n in sizes:
        H = build_tfim_hamiltonian(n, lambda_1, lambda_2, mu=mu)
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.sort(eigvals)

        # Identify zero modes (energy threshold for zero)
        zero_threshold = 0.01
        zero_mask = np.abs(eigvals) < zero_threshold
        n_zero = np.sum(zero_mask)

        # Get zero mode energies for tracking
        if n_zero > 0:
            zero_energies_list.append(eigvals[zero_mask])
        else:
            zero_energies_list.append(np.array([]))

        # BULK GAP: minimum |E| excluding zero modes
        # This is what extrapolates to thermodynamic limit
        bulk_states = eigvals[~zero_mask]
        if len(bulk_states) > 0:
            bulk_gap = np.min(np.abs(bulk_states))
        else:
            # Pathological case: all states are zero modes
            bulk_gap = np.nan

        gaps.append(bulk_gap)
        n_zero_modes_list.append(n_zero)

        if verbose:
            zero_str = f"{n_zero} modes at E≈" + ", ".join([f"{e:.6f}" for e in eigvals[zero_mask]]) if n_zero > 0 else "0 modes"
            print(f"L={n:4d}: bulk_gap={bulk_gap:.6f}, zero modes: {zero_str}")

    return np.array(gaps), zero_energies_list, np.array(n_zero_modes_list)


def fit_gap_scaling(sizes, gaps, fit_type='power'):
    """
    Fit gap vs system size to extrapolate to infinite size.

    Two common scaling forms:
    - Power law: Δ(L) = Δ_∞ + a/L^α
    - Exponential: Δ(L) = Δ_∞ + a·exp(-L/ξ)

    Parameters
    ----------
    sizes : array-like
        System sizes
    gaps : array-like
        Measured gaps
    fit_type : str
        'power' or 'exponential'

    Returns
    -------
    params : dict
        Fitted parameters
    extrapolated_gap : float
        Gap in thermodynamic limit
    """
    sizes = np.array(sizes)
    gaps = np.array(gaps)

    if fit_type == 'power':
        # Fit: gap(L) = gap_inf + a / L^alpha
        def power_law(L, gap_inf, a, alpha):
            return gap_inf + a / L**alpha

        try:
            # Initial guess: gap_inf ≈ last value, a ≈ gap[0]-gap[-1], alpha ≈ 1
            p0 = [gaps[-1], (gaps[0] - gaps[-1]) * sizes[0], 1.0]
            popt, pcov = curve_fit(power_law, sizes, gaps, p0=p0, maxfev=5000)
            gap_inf, a, alpha = popt
            perr = np.sqrt(np.diag(pcov))

            params = {
                'gap_infinity': gap_inf,
                'amplitude': a,
                'exponent': alpha,
                'gap_infinity_error': perr[0],
                'fit_type': 'power'
            }

            return params, gap_inf

        except Exception as e:
            print(f"Power law fit failed: {e}")
            # Fallback: just use largest system
            return {'gap_infinity': gaps[-1], 'fit_type': 'failed'}, gaps[-1]

    elif fit_type == 'exponential':
        # Fit: gap(L) = gap_inf + a * exp(-L/xi)
        def exp_decay(L, gap_inf, a, xi):
            return gap_inf + a * np.exp(-L / xi)

        try:
            p0 = [gaps[-1], gaps[0] - gaps[-1], sizes[-1]/3]
            popt, pcov = curve_fit(exp_decay, sizes, gaps, p0=p0, maxfev=5000)
            gap_inf, a, xi = popt
            perr = np.sqrt(np.diag(pcov))

            params = {
                'gap_infinity': gap_inf,
                'amplitude': a,
                'correlation_length': xi,
                'gap_infinity_error': perr[0],
                'fit_type': 'exponential'
            }

            return params, gap_inf

        except Exception as e:
            print(f"Exponential fit failed: {e}")
            return {'gap_infinity': gaps[-1], 'fit_type': 'failed'}, gaps[-1]

    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")


def extract_localization_length(eigvec, n, edge='left'):
    """
    Extract localization length from edge mode spatial profile.

    Edge modes decay exponentially: |ψ(x)| ∝ exp(-x/ξ)
    The localization length ξ tells us how far the mode penetrates
    into the bulk.

    Parameters
    ----------
    eigvec : ndarray
        Eigenvector (should be a zero mode)
    n : int
        System size
    edge : str
        'left' or 'right' edge to analyze

    Returns
    -------
    xi : float
        Localization length (in units of lattice spacing)
    fit_quality : float
        R² value of exponential fit (1 = perfect)
    """
    # Extract the particle sector (first n components)
    psi = eigvec[:n]
    psi_abs = np.abs(psi)

    if edge == 'left':
        # Fit left edge (first quarter of system)
        fit_range = slice(0, n//4)
        x_data = np.arange(n//4)
    elif edge == 'right':
        # Fit right edge (last quarter)
        fit_range = slice(3*n//4, n)
        x_data = np.arange(n//4)
    else:
        raise ValueError("edge must be 'left' or 'right'")

    y_data = psi_abs[fit_range]

    # Fit exponential decay: y = A * exp(-x/xi)
    # Take log: log(y) = log(A) - x/xi
    # This becomes linear fit, but watch out for zeros

    # Remove any zeros/very small values
    valid = y_data > 1e-12
    x_fit = x_data[valid]
    y_fit = y_data[valid]

    if len(x_fit) < 3:
        # Not enough points for fit
        return np.nan, 0.0

    try:
        # Fit in log space
        log_y = np.log(y_fit)
        coeffs = np.polyfit(x_fit, log_y, 1)
        slope, intercept = coeffs

        # Localization length is -1/slope
        xi = -1.0 / slope

        # Compute R² to assess fit quality
        log_y_pred = slope * x_fit + intercept
        ss_res = np.sum((log_y - log_y_pred)**2)
        ss_tot = np.sum((log_y - np.mean(log_y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Localization length should be positive
        if xi < 0:
            xi = np.nan
            r_squared = 0

        return xi, r_squared

    except:
        return np.nan, 0.0


def analyze_edge_modes(lambda_1, lambda_2, sizes, mu=0.0):
    """
    Full edge mode analysis across different system sizes.

    Computes:
    - Localization lengths
    - Edge mode energies
    - Spatial profiles

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    sizes : list of int
        System sizes
    mu : float
        Chemical potential (default: 0.0)

    Returns
    -------
    results : dict
        Dictionary containing localization lengths, energies, etc.
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    results = {
        'sizes': [],
        'localization_lengths': [],
        'edge_mode_energies': [],
        'zero_mode_counts': []
    }

    for n in sizes:
        H = build_tfim_hamiltonian(n, lambda_1, lambda_2, mu=mu)
        eigvals, eigvecs = np.linalg.eigh(H)

        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Find zero modes (|E| < 0.01)
        zero_idx = np.abs(eigvals) < 0.01
        n_zero = np.sum(zero_idx)

        results['sizes'].append(n)
        results['zero_mode_counts'].append(n_zero)

        if n_zero > 0:
            # Analyze the first zero mode
            zero_mode = eigvecs[:, zero_idx][:, 0]
            xi, r2 = extract_localization_length(zero_mode, n, edge='left')
            results['localization_lengths'].append(xi)
            results['edge_mode_energies'].append(eigvals[zero_idx][0])
        else:
            results['localization_lengths'].append(np.nan)
            results['edge_mode_energies'].append(np.nan)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def run_full_scaling_analysis(lambda_1, lambda_2, sizes=None, mu=0.0):
    """
    Run complete finite-size scaling analysis.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    sizes : list, optional
        System sizes to analyze. Default: [20, 50, 100, 200, 400]
    mu : float
        Chemical potential (default: 0.0)

    Returns
    -------
    results : dict
        Complete analysis results including fits and extrapolations
    """
    if sizes is None:
        sizes = [20, 50, 100, 200, 400]

    print(f"=" * 60)
    print(f"Finite-size scaling analysis for (λ₁, λ₂) = ({lambda_1}, {lambda_2})")
    print(f"μ = {mu}")
    print(f"=" * 60)

    # Compute BULK gaps (corrected calculation)
    print("\n1. Computing BULK energy gaps...")
    gaps, zero_energies_list, n_zero_modes = compute_gap_vs_size(lambda_1, lambda_2, sizes, mu=mu)

    # Fit scaling
    print("\n2. Fitting gap scaling (power law)...")
    fit_params, gap_inf = fit_gap_scaling(sizes, gaps, fit_type='power')
    print(f"   Extrapolated bulk gap: Δ_∞ = {gap_inf:.6f}")
    if 'exponent' in fit_params:
        print(f"   Scaling exponent: α = {fit_params['exponent']:.3f}")
        if np.abs(gap_inf) < 0.1:
            print(f"   → Gap closes in thermodynamic limit (topological phase)")
        else:
            print(f"   → Finite gap in thermodynamic limit (trivial or gapped phase)")

    # Edge mode analysis
    print("\n3. Analyzing edge modes...")
    edge_results = analyze_edge_modes(lambda_1, lambda_2, sizes, mu=mu)

    print(f"\n   Localization lengths:")
    for L, xi in zip(edge_results['sizes'], edge_results['localization_lengths']):
        if not np.isnan(xi):
            print(f"     L={L:4d}: ξ = {xi:.2f} sites")
        else:
            print(f"     L={L:4d}: ξ = N/A (no zero mode)")

    # Combine results
    results = {
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'mu': mu,
        'sizes': sizes,
        'gaps': gaps,
        'zero_energies': zero_energies_list,
        'n_zero_modes': n_zero_modes,
        'fit_params': fit_params,
        'extrapolated_gap': gap_inf,
        'edge_analysis': edge_results
    }

    print(f"\n" + "=" * 60)
    return results
