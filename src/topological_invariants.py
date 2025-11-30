"""
Topological invariant calculations for 1D systems.

The winding number ν characterizes the topological phase of the system.
For a 1D BdG Hamiltonian, the winding number counts how many times
the Hamiltonian wraps around the origin in parameter space as we
vary momentum k from -π to π.
"""
import numpy as np


def epsilon_k(k, lambda_1, lambda_2, mu=0.0):
    """
    Single-particle energy dispersion in momentum space.

    After Fourier transforming the hopping terms:
    ε(k) = μ + 2λ₁cos(k) + 2λ₂cos(2k)

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
    return mu + 2 * lambda_1 * np.cos(k) + 2 * lambda_2 * np.cos(2*k)


def delta_k(k, lambda_1, lambda_2):
    """
    Pairing amplitude in momentum space.

    After Fourier transforming the antisymmetric Δ matrix:
    Δ(k) = -2λ₁sin(k) - 2λ₂sin(2k)

    The factor of 2 comes from the hopping terms ±i in the Majorana basis.
    """
    return -2 * lambda_1 * np.sin(k) - 2 * lambda_2 * np.sin(2*k)


def compute_winding_number(lambda_1, lambda_2, n_k=1000, mu=0.0):
    """
    Compute topological winding number ν.

    The winding number is defined as:
        ν = (1/2π) ∮ dk ∂_k θ(k)

    where θ(k) = arg[q(k)] and q(k) = Δ(k) + iε(k) is the off-diagonal
    block of the BdG Hamiltonian in momentum space.

    Physically, ν counts the number of Majorana zero modes at each edge
    of an open chain (bulk-boundary correspondence).

    Parameters
    ----------
    lambda_1 : float
        Nearest-neighbor coupling
    lambda_2 : float
        Next-nearest-neighbor coupling
    n_k : int, optional
        Number of k-points for numerical integration (default: 1000)
    mu : float, optional
        Chemical potential (default: 0.0)

    Returns
    -------
    nu : int
        Winding number. Typical values: 0 (trivial), ±1, ±2 (topological)

    Notes
    -----
    - The winding number is a topological invariant: it only changes
      when the bulk gap closes (phase transition)
    - |ν| = number of Majorana zero modes per edge
    - Sign of ν indicates chirality of edge modes
    """
    # Create momentum grid covering the Brillouin zone
    # We use endpoint=False to avoid double-counting k=-π and k=π (they're the same point)
    k_vals = np.linspace(-np.pi, np.pi, n_k, endpoint=False)

    # Compute ε(k) and Δ(k) for all k
    epsilon_vals = np.array([epsilon_k(k, lambda_1, lambda_2, mu) for k in k_vals])
    delta_vals = np.array([delta_k(k, lambda_1, lambda_2) for k in k_vals])

    # Construct the complex number q(k) = Δ(k) + i·ε(k)
    # This represents the off-diagonal block of H_BdG(k)
    q_k = delta_vals + 1j * epsilon_vals

    # Calculate the phase θ(k) = arg[q(k)]
    theta = np.angle(q_k)

    # Unwrap the phase to handle 2π discontinuities
    # This is crucial: without unwrapping, we'd miss full windings
    theta_unwrapped = np.unwrap(theta)

    # The winding number is the total phase change divided by 2π
    # We integrate from -π to π, so the endpoints give us the total winding
    nu = (theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi)

    # Round to nearest integer (should be exact integer theoretically,
    # but numerical integration introduces small errors)
    return int(np.round(nu))


def compute_winding_map(lambda1_range, lambda2_range, verbose=True):
    """
    Compute winding number across a 2D parameter space.

    This generates a phase diagram showing topological regions.

    Parameters
    ----------
    lambda1_range : array-like
        Range of λ₁ values to scan
    lambda2_range : array-like
        Range of λ₂ values to scan
    verbose : bool, optional
        Print progress updates (default: True)

    Returns
    -------
    winding_map : ndarray, shape (len(lambda1_range), len(lambda2_range))
        2D array of winding numbers. Each entry is an integer.

    Examples
    --------
    >>> lambda1 = np.linspace(-3, 3, 50)
    >>> lambda2 = np.linspace(-3, 3, 50)
    >>> nu_map = compute_winding_map(lambda1, lambda2)
    >>> plt.contourf(lambda2, lambda1, nu_map, levels=[-2.5,-1.5,-0.5,0.5,1.5,2.5])
    """
    n1, n2 = len(lambda1_range), len(lambda2_range)
    winding_map = np.zeros((n1, n2), dtype=int)

    for i, l1 in enumerate(lambda1_range):
        for j, l2 in enumerate(lambda2_range):
            winding_map[i, j] = compute_winding_number(l1, l2)

        # Progress indicator
        if verbose and (i % max(1, n1//10) == 0):
            print(f"Progress: {i}/{n1} rows completed ({100*i//n1}%)")

    if verbose:
        print(f"Progress: {n1}/{n1} rows completed (100%)")
        print("\nWinding number statistics:")
        unique, counts = np.unique(winding_map, return_counts=True)
        for val, count in zip(unique, counts):
            pct = 100 * count / winding_map.size
            print(f"  ν = {val:+2d}: {count:5d} points ({pct:5.1f}%)")

    return winding_map


def find_phase_boundaries(lambda1_range, lambda2_range, winding_map):
    """
    Find phase transition boundaries where winding number changes.

    Parameters
    ----------
    lambda1_range : array-like
        λ₁ values (y-axis)
    lambda2_range : array-like
        λ₂ values (x-axis)
    winding_map : ndarray
        Winding number at each point

    Returns
    -------
    boundaries : list of tuples
        List of (λ₁, λ₂) points where phase transitions occur
    """
    boundaries = []

    # Check for changes in horizontal direction
    for i in range(len(lambda1_range)):
        for j in range(len(lambda2_range) - 1):
            if winding_map[i, j] != winding_map[i, j+1]:
                # Interpolate to find approximate boundary
                l1 = lambda1_range[i]
                l2 = 0.5 * (lambda2_range[j] + lambda2_range[j+1])
                boundaries.append((l1, l2))

    # Check for changes in vertical direction
    for i in range(len(lambda1_range) - 1):
        for j in range(len(lambda2_range)):
            if winding_map[i, j] != winding_map[i+1, j]:
                l1 = 0.5 * (lambda1_range[i] + lambda1_range[i+1])
                l2 = lambda2_range[j]
                boundaries.append((l1, l2))

    return boundaries


def verify_bulk_boundary_correspondence(lambda_1, lambda_2, n=100, tol=0.01):
    """
    Test bulk-boundary correspondence: winding number = number of edge modes.

    This computes both the winding number (bulk property) and counts
    zero-energy eigenstates (boundary property), then checks if they match.

    Parameters
    ----------
    lambda_1, lambda_2 : float
        Coupling parameters
    n : int
        System size for open boundary calculation
    tol : float
        Energy threshold for identifying zero modes

    Returns
    -------
    nu_bulk : int
        Winding number from bulk calculation
    n_edge : int
        Number of near-zero modes from boundary calculation
    match : bool
        True if bulk-boundary correspondence holds
    """
    try:
        from .tfim_model_core import build_tfim_hamiltonian
    except ImportError:
        from tfim_model_core import build_tfim_hamiltonian

    # Bulk: compute winding number
    nu_bulk = compute_winding_number(lambda_1, lambda_2)

    # Boundary: count zero modes in finite system
    H = build_tfim_hamiltonian(n, lambda_1, lambda_2)
    eigvals = np.linalg.eigvalsh(H)
    n_edge = np.sum(np.abs(eigvals) < tol)

    # The correspondence predicts |ν| edge modes per edge
    # For a chain with 2 edges, we expect 2|ν| total zero modes
    expected = 2 * abs(nu_bulk)
    match = (n_edge == expected)

    return nu_bulk, n_edge, match
