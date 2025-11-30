#!/usr/bin/env python3
"""Generate momentum-space band structure plots."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from momentum_space import compute_band_structure, trace_phase_transition, compute_dispersion_components

print("=" * 60)
print("Band Structure Analysis")
print("=" * 60)

# Plot 1: Band structure at topological point
print("\n1. Computing band structure at (λ₁=1.0, λ₂=-1.2)...")
k_vals, E_plus, E_minus = compute_band_structure(lambda_1=1.0, lambda_2=-1.2, n_k=300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(k_vals/np.pi, E_plus, 'b-', linewidth=2, label='E₊(k)')
ax.plot(k_vals/np.pi, E_minus, 'r-', linewidth=2, label='E₋(k)')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('k/π', fontsize=12)
ax.set_ylabel('Energy E(k)', fontsize=12)
ax.set_title('Band Structure (λ₁=1.0, λ₂=-1.2)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 1)

# Plot 2: Dispersion components
print("2. Computing ε(k) and Δ(k)...")
k_vals, epsilon, delta = compute_dispersion_components(lambda_1=1.0, lambda_2=-1.2, n_k=300)

ax = axes[1]
ax.plot(k_vals/np.pi, epsilon, 'g-', linewidth=2, label='ε(k)')
ax.plot(k_vals/np.pi, delta, 'm-', linewidth=2, label='Δ(k)')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('k/π', fontsize=12)
ax.set_ylabel('ε(k), Δ(k)', fontsize=12)
ax.set_title('Dispersion Components', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 1)

plt.tight_layout()
plt.savefig('notebooks/plots/band_structure.png', dpi=300, bbox_inches='tight')
print("   Saved to notebooks/plots/band_structure.png")

# Plot 3: Phase transition (gap closing)
print("\n3. Tracing phase transition (λ₁=1.0, varying λ₂)...")
lambda2_range = np.linspace(-2.0, 0.0, 100)
transition = trace_phase_transition(lambda2_range, lambda_1=1.0, n_k=300)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(transition['lambda_2'], transition['bulk_gap'], 'b-', linewidth=2, label='Bulk gap')
ax.plot(transition['lambda_2'], transition['gap_at_k0'], 'r--', linewidth=1.5, label='Gap at k=0')
ax.plot(transition['lambda_2'], transition['gap_at_kpi'], 'g--', linewidth=1.5, label='Gap at k=π')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axvline(-1.2, color='orange', linestyle=':', linewidth=2, label='Test point')
ax.set_xlabel('λ₂', fontsize=12)
ax.set_ylabel('Energy Gap', fontsize=12)
ax.set_title('Gap Closing at Phase Transition (λ₁=1.0)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/plots/phase_transition.png', dpi=300, bbox_inches='tight')
print("   Saved to notebooks/plots/phase_transition.png")

# Plot 4: Band structure evolution through transition
print("\n4. Creating band evolution through transition...")
lambda2_values = [-2.0, -1.5, -1.2, -0.5, 0.0]
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for i, l2 in enumerate(lambda2_values):
    k_vals, E_plus, E_minus = compute_band_structure(lambda_1=1.0, lambda_2=l2, n_k=200)
    ax = axes[i]
    ax.plot(k_vals/np.pi, E_plus, 'b-', linewidth=2)
    ax.plot(k_vals/np.pi, E_minus, 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('k/π', fontsize=10)
    if i == 0:
        ax.set_ylabel('E(k)', fontsize=11)
    ax.set_title(f'λ₂={l2}', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-1, 1)

    # Find minimum gap
    min_gap = np.min(np.abs(E_plus))
    ax.text(0.5, 0.9, f'gap={min_gap:.3f}', transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('notebooks/plots/band_evolution.png', dpi=300, bbox_inches='tight')
print("   Saved to notebooks/plots/band_evolution.png")

print("\n" + "=" * 60)
print("Key observations:")
print(f"  At (1.0, -1.2): Small but non-zero bulk gap")
print(f"  Gap closes around λ₂ ≈ -0.5 to 0.0")
print(f"  ε(k) and Δ(k) have different momentum dependence")
print("=" * 60)
