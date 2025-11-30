# majorana-tfim-vqet

A minimal, clean implementation of the extended Kitaev chain (transverse-field Ising model after Jordan-Wigner transformation) using Bogoliubov-de Gennes formalism. Hunt for Majorana zero modes, compute topological invariants, and explore the quantum phase diagram.

This repo is all about understanding **topological superconductivity** through the lens of the 1D Kitaev chain. The code is ~200 lines for the core model (`src/tfim_model_core.py`) with additional modules for topological analysis, entanglement measures, and disorder effects. Perfect for researchers, students, or anyone curious about Majorana fermions.

## install

Dependencies are minimal. You need numpy, scipy, matplotlib, and jupyter for notebooks:

```bash
pip install -r requirements.txt
```

That's it. All numpy <3

## quick start

The simplest way to get started is to build the Hamiltonian and find eigenvalues:

```python
from src.tfim_model_core import build_tfim_hamiltonian
import numpy as np

# Build extended Kitaev chain: n sites, λ₁ (nearest), λ₂ (next-nearest)
n = 100
H = build_tfim_hamiltonian(n, lambda_1=1.0, lambda_2=0.5, mu=0.0, pbc=False)

# Diagonalize (BdG structure gives ±E pairs)
eigvals, eigvecs = np.linalg.eigh(H)

# Check for zero modes (topological phase indicator)
zero_modes = eigvals[np.abs(eigvals) < 1e-6]
print(f"Found {len(zero_modes)} zero mode(s)")
```

The magic parameter regime is `λ₁ = 1.0, λ₂ < -1.0` for topological phase with Majorana edge modes.

## what's in here

The repo contains:

- **`src/tfim_model_core.py`**: Core BdG Hamiltonian construction (the heart of it all)
- **`src/topological_invariants.py`**: Pfaffian invariant, winding number calculations
- **`src/entanglement.py`**: Entanglement entropy and spectrum analysis
- **`src/momentum_space.py`**: k-space transformations and band structure
- **`src/disorder.py`**: Random disorder effects on topological protection
- **`src/finite_size_scaling.py`**: Scaling analysis for phase transitions
- **`src/vqe_solver.py`**: Variational quantum eigensolver routines
- **`notebooks/`**: Jupyter notebooks with examples and visualizations
- **`tests/`**: Unit tests for core functionality
- **`scripts/`**: Production runs and parameter sweeps

## the physics

After Jordan-Wigner transformation, the 1D transverse-field Ising model becomes a fermionic model with p-wave pairing:

```
H = -∑ᵢ [μ cᵢ†cᵢ + λ₁(cᵢ†cᵢ₊₁ + h.c.) + λ₂(cᵢ†cᵢ₊₂ + h.c.)
         + λ₁(cᵢ†cᵢ₊₁† + h.c.) + λ₂(cᵢ†cᵢ₊₂† + h.c.)]
```

The pairing terms (cᵢ†cⱼ†) are the smoking gun for topological superconductivity. We use the BdG formalism to handle this elegantly:

```
H_BdG = [[h, Δ], [-Δ, -h]]
```

where `h` is single-particle hopping and `Δ` is the antisymmetric pairing matrix. Particle-hole symmetry guarantees eigenvalues come in ±E pairs.

**Topological phase**: When `|λ₂| > 1`, you get Majorana zero modes localized at the chain edges. These are robust against local perturbations (until you hit the phase boundary).

## example: finding the topological phase

```python
from src.tfim_model_core import build_tfim_hamiltonian
import numpy as np
import matplotlib.pyplot as plt

lambda_1 = 1.0
lambda_2_range = np.linspace(-2, 0, 50)
n = 100
gap = []

for lambda_2 in lambda_2_range:
    H = build_tfim_hamiltonian(n, lambda_1, lambda_2, pbc=False)
    eigvals = np.linalg.eigvalsh(H)
    # Gap is smallest positive eigenvalue (BdG gives ±E)
    positive_eigvals = eigvals[eigvals > 1e-10]
    gap.append(positive_eigvals[0] if len(positive_eigvals) > 0 else 0)

plt.plot(lambda_2_range, gap)
plt.xlabel('λ₂')
plt.ylabel('Energy Gap')
plt.title('Topological Phase Transition (gap closes at λ₂ = -1)')
plt.show()
```

You'll see the gap close and reopen at λ₂ = -1, marking the phase transition.

## computational notes

- **Eigendecomposition**: Using `np.linalg.eigh()` for Hermitian matrices (though BdG is particle-hole symmetric, not strictly Hermitian)
- **Scaling**: n=200 sites runs in <1s on laptop. For large-scale sweeps, consider parallelization in `scripts/`
- **Boundary conditions**: OBC (open) for edge modes, PBC (periodic) for bulk properties
- **Numerical stability**: Watch out for near-zero eigenvalues when checking topological phase (use threshold ~1e-6)

## references

- Kitaev (2001): "Unpaired Majorana fermions in quantum wires" [Physics-Uspekhi 44:131]
- Long-range Kitaev: [arXiv:1405.5440](https://arxiv.org/abs/1405.5440), [arXiv:1508.00820](https://arxiv.org/abs/1508.00820)
- Extended Kitaev chains: [arXiv:2509.04420](https://arxiv.org/abs/2509.04420)

## faq

**Q: Why BdG formalism instead of direct diagonalization?**
A: Pairing terms mix particle and hole sectors. BdG handles this naturally and respects particle-hole symmetry.

**Q: What's the deal with Majorana zero modes?**
A: They're eigenstates at exactly E=0, localized at edges, and are their own antiparticles. Non-abelian braiding statistics make them useful for topological quantum computing.

**Q: Can I use this for higher dimensions?**
A: Code is 1D-specific, but the BdG framework extends to 2D/3D. You'd need to generalize the lattice construction.

## license

MIT

---

Built with numpy. Inspired by the beauty of emergent topology in condensed matter systems.
