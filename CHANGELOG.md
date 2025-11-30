# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2025-11-28

### Added
- **Topological invariants module** (`src/topological_invariants.py`)
  - Winding number calculation for phase classification
  - Bulk-boundary correspondence verification
  - Phase boundary detection

- **Finite-size scaling module** (`src/finite_size_scaling.py`)
  - Gap extrapolation to thermodynamic limit
  - Localization length extraction
  - Scaling exponent fitting

- **Disorder analysis module** (`src/disorder.py`)
  - On-site and hopping disorder
  - Disorder averaging over many realizations
  - Zero-mode survival probability
  - Critical disorder strength finder

- **Momentum-space module** (`src/momentum_space.py`)
  - Band structure E(k) calculations
  - Gap closing detection
  - Berry phase computation

- **Comprehensive test suite** (`tests/`)
  - Hamiltonian property tests (Hermiticity, particle-hole symmetry)
  - Topological invariant tests
  - Bulk-boundary correspondence verification

- **Configuration system** (`config/default_params.yaml`)
  - YAML-based parameter management
  - Config loader in `src/config.py`

- **Analysis scripts** (`scripts/`)
  - Winding number phase diagram generator
  - Finite-size scaling analysis

- **Documentation**
  - Detailed docstrings in all modules
  - CHANGELOG.md (this file)
  - Updated README.md with usage instructions

### Fixed
- Windows-specific hardcoded path in `notebooks/eigenvectors.ipynb`
- Unified sign convention for Δ matrix in BdG Hamiltonian
- Import errors with relative imports

### Changed
- Updated `requirements.txt` with flexible version constraints
- Enhanced `tfim_model_core.py` with comprehensive documentation
- All modules now support both relative and absolute imports

## [0.1.0] - 2024-04-28 (Initial state)

### Added
- Initial BdG Hamiltonian construction for extended TFIM
- Jordan-Wigner transformation implementation
- Basic eigenvector plotting
- Phase diagram generation via energy gap
- Jupyter notebooks for exploration
