"""
Core Extended Kitaev Chain Hamiltonian in Bogoliubov-de Gennes form.

PHYSICAL MODEL:
===============
This implements the extended Kitaev chain, which is the fermionic representation
of the transverse-field Ising model (TFIM) after Jordan-Wigner transformation.

FERMIONIC HAMILTONIAN (what we implement):
H = ∑ᵢ [(μ - 2) cᵢ†cᵢ] - ∑ᵢ [λ₁(cᵢ†cᵢ₊₁ + H.c.) + λ₂(cᵢ†cᵢ₊₂ + H.c.)
         + λ₁(cᵢ†cᵢ₊₁† + H.c.) + λ₂(cᵢ†cᵢ₊₂† + H.c.)]

where:
  - μ: chemical potential shift (default 0 gives on-site energy -2)
  - λ₁: nearest-neighbor hopping and p-wave pairing amplitude
  - λ₂: next-nearest-neighbor hopping and p-wave pairing amplitude
  - cᵢ†, cᵢ: fermionic creation/annihilation operators

Note: The -2 on-site term comes from the Jordan-Wigner transformation
      of the transverse-field Ising model with g=1.

CONNECTION TO SPIN MODEL:
=========================
After Jordan-Wigner transformation σᵢˣ → ∏ⱼ₍ⱼ<ᵢ₎ σⱼᶻ (cᵢ† + cᵢ), the TFIM maps to
this fermionic model. The extended version includes longer-range interactions.

BDG FORMULATION:
================
We use the Bogoliubov-de Gennes (BdG) formalism to handle the pairing terms:

Ψ = (c₁, c₂, ..., cₙ, c₁†, c₂†, ..., cₙ†)ᵀ

H_BdG Ψ = E Ψ, where H_BdG = [[h, Δ], [-Δ, -h]]

  - h: single-particle Hamiltonian (hopping + μ)
  - Δ: antisymmetric pairing matrix (p-wave superconducting pairing)

REFERENCES:
===========
- Kitaev (2001): Unpaired Majorana fermions in quantum wires
- Long-range Kitaev models: arXiv:1405.5440, arXiv:1508.00820
- Extended Kitaev chains: arXiv:2509.04420
"""
import numpy as np

def build_tfim_hamiltonian(n, lambda_1, lambda_2, mu=0.0, pbc=False):
    """
    Build Bogoliubov-de Gennes Hamiltonian for extended TFIM.

    The BdG Hamiltonian has the structure:
        H_BdG = [[h, Δ], [-Δ, -h]]

    where h is the single-particle Hamiltonian and Δ is the pairing matrix.
    Note: Δ is antisymmetric (Δ[i,j] = -Δ[j,i]), which is why we use
    the convention H_BdG = [[h, Δ], [-Δ, -h]] to maintain particle-hole symmetry.

    Parameters
    ----------
    n : int
        Number of lattice sites
    lambda_1 : float
        Nearest-neighbor coupling strength
    lambda_2 : float
        Next-nearest-neighbor coupling strength
    mu : float, optional
        Chemical potential shift (default: 0.0). The actual on-site energy
        is (mu - 2), where -2 comes from the Jordan-Wigner transformation.
        Setting mu=0 gives the standard extended Kitaev model from Niu et al.
    pbc : bool, optional
        Use periodic boundary conditions (default: False for OBC)

    Returns
    -------
    H : ndarray, shape (2n, 2n)
        BdG Hamiltonian.
        Eigenvalues come in ±E pairs due to particle-hole symmetry.

    Notes
    -----
    - OBC (pbc=False): Allows edge-localized Majorana zero modes
    - PBC (pbc=True): No physical edges, cleaner entanglement spectrum
    - μ=0 gives on-site energy -2 (standard model with g=1 in TFIM)
    - Particle-hole symmetry: σ_x H σ_x = -H where σ_x = [[0,I],[I,0]]
    """
    h = np.zeros((n, n))
    delta = np.zeros((n, n))
    H = np.zeros((2*n, 2*n))

    # Build single-particle Hamiltonian h
    # Diagonal: -2 (from Jordan-Wigner transformation) + chemical potential
    # Off-diagonal: hopping terms
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i, j] = mu - 2  # -2 comes from J-W transformation, mu is chemical potential shift
            elif abs(j - i) == 1:
                h[i, j] = lambda_1
            elif abs(j - i) == 2:
                h[i, j] = lambda_2
            # PBC: wraparound terms
            elif pbc and abs(j - i) == n - 1:  # nearest-neighbor wrap
                h[i, j] = lambda_1
            elif pbc and abs(j - i) == n - 2:  # next-nearest wrap
                h[i, j] = lambda_2

    # Build pairing matrix Δ (antisymmetric)
    # This comes from the p-wave superconducting pairing after JW transform
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
            # PBC: wraparound pairing
            elif pbc and j == i + (n - 1):  # i=n-1, j=0
                delta[i, j] = -lambda_1
            elif pbc and j == i - (n - 1):  # i=0, j=n-1
                delta[i, j] = lambda_1
            elif pbc and j == i + (n - 2):  # wrap +2
                delta[i, j] = -lambda_2
            elif pbc and j == i - (n - 2):  # wrap -2
                delta[i, j] = lambda_2

    # Assemble BdG Hamiltonian
    # Top-left: h (particle sector)
    # Top-right: Δ (particle-hole coupling)
    # Bottom-left: -Δ (maintains antisymmetry)
    # Bottom-right: -h (hole sector)
    H[:n, :n] = h
    H[:n, n:] = delta
    H[n:, :n] = -delta
    H[n:, n:] = -h

    return H
