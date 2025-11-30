#!/usr/bin/env python3
"""
[DEPRECATED] Generate entanglement entropy phase diagram.

WARNING: This script produces non-physical results for 1D topological phases.
The extracted γ values (~-20 to -25) are NOT topological entanglement entropy.

For each (λ₁, λ₂) point:
  - Compute S at multiple system sizes
  - Fit S = a·log(L) + γ
  - Extract constant term γ (but this is NOT TEE!)

ISSUES:
1. γ is non-universal, contaminated by finite-size/boundary effects
2. Does not match topological phase boundaries from winding number
3. TEE formulas (Kitaev-Preskill) are designed for 2D, not 1D

USE INSTEAD:
- scripts/generate_winding_diagram.py (rigorous topological diagnostic)
- Entanglement code kept for central charge extraction at critical points

This script is preserved for reference only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from entanglement import compute_entanglement_entropy
import time

print("=" * 70)
print("RIGOROUS ENTANGLEMENT ENTROPY PHASE DIAGRAM")
print("=" * 70)

# Grid parameters
lambda1_range = np.linspace(-3, 3, 40)  # 40x40 for reasonable time
lambda2_range = np.linspace(-3, 3, 40)
sizes = [20, 30, 40, 60, 80]  # System sizes for scaling

n_points = len(lambda1_range) * len(lambda2_range)
print(f"\nGrid: {len(lambda1_range)} × {len(lambda2_range)} = {n_points} points")
print(f"System sizes: {sizes}")
print(f"Total calculations: {n_points * len(sizes)}")
print(f"Estimated time: ~2-3 hours")
print(f"\nStarting computation...\n")

# Storage arrays
gamma_map = np.zeros((len(lambda1_range), len(lambda2_range)))
gamma_error_map = np.zeros((len(lambda1_range), len(lambda2_range)))
central_charge_map = np.zeros((len(lambda1_range), len(lambda2_range)))

# Scaling fit function
def scaling_form(L, a, gamma):
    return a * np.log(L) + gamma

# Track progress
start_time = time.time()
completed = 0

for i, l1 in enumerate(lambda1_range):
    row_start = time.time()

    for j, l2 in enumerate(lambda2_range):
        try:
            # Compute entanglement at each size
            entropies = []
            for n in sizes:
                l_A = n // 2
                S = compute_entanglement_entropy(l1, l2, n, l_A)
                entropies.append(S)

            entropies = np.array(entropies)

            # Fit to extract γ
            popt, pcov = curve_fit(scaling_form, sizes, entropies,
                                   p0=[1.0, 0.0], maxfev=5000)
            a, gamma = popt
            gamma_err = np.sqrt(pcov[1, 1])

            gamma_map[i, j] = gamma
            gamma_error_map[i, j] = gamma_err
            central_charge_map[i, j] = 3 * a

        except Exception as e:
            # If fit fails, use NaN
            gamma_map[i, j] = np.nan
            gamma_error_map[i, j] = np.nan
            central_charge_map[i, j] = np.nan

        completed += 1

    # Progress report
    row_time = time.time() - row_start
    elapsed = time.time() - start_time
    percent = 100 * (i + 1) / len(lambda1_range)
    eta = (elapsed / (i + 1)) * (len(lambda1_range) - i - 1)

    print(f"Row {i+1}/{len(lambda1_range)} ({percent:.1f}%) | "
          f"Row time: {row_time:.1f}s | "
          f"Elapsed: {elapsed/60:.1f}min | "
          f"ETA: {eta/60:.1f}min")

    # Save intermediate results every 5 rows
    if (i + 1) % 5 == 0:
        np.savez('data/entanglement_diagram_partial.npz',
                 gamma_map=gamma_map,
                 gamma_error_map=gamma_error_map,
                 central_charge_map=central_charge_map,
                 lambda1_range=lambda1_range,
                 lambda2_range=lambda2_range,
                 completed_rows=i+1)
        print(f"  → Checkpoint saved ({i+1} rows)")

total_time = time.time() - start_time
print(f"\nCompleted in {total_time/60:.1f} minutes")

# Save final results
np.savez('data/entanglement_diagram.npz',
         gamma_map=gamma_map,
         gamma_error_map=gamma_error_map,
         central_charge_map=central_charge_map,
         lambda1_range=lambda1_range,
         lambda2_range=lambda2_range)
print("Data saved to data/entanglement_diagram.npz")

# Generate plots
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Topological entropy γ
ax = axes[0, 0]
im1 = ax.contourf(lambda2_range, lambda1_range, gamma_map,
                  levels=20, cmap='viridis')
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('Topological Entropy γ', fontsize=12)
ax.set_xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
ax.set_ylabel('λ₁ (nearest neighbor)', fontsize=12)
ax.set_title('Entanglement Entropy Phase Diagram', fontsize=14, weight='bold')
ax.plot([-1.2], [1.0], 'r*', markersize=15, label='Test point')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Central charge
ax = axes[0, 1]
im2 = ax.contourf(lambda2_range, lambda1_range, central_charge_map,
                  levels=20, cmap='plasma')
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('Central Charge c', fontsize=12)
ax.set_xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
ax.set_ylabel('λ₁ (nearest neighbor)', fontsize=12)
ax.set_title('Effective Central Charge from Scaling', fontsize=14)
ax.grid(True, alpha=0.3)

# Plot 3: Uncertainty in γ
ax = axes[1, 0]
im3 = ax.contourf(lambda2_range, lambda1_range, gamma_error_map,
                  levels=20, cmap='hot_r')
cbar3 = plt.colorbar(im3, ax=ax)
cbar3.set_label('γ Uncertainty', fontsize=12)
ax.set_xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
ax.set_ylabel('λ₁ (nearest neighbor)', fontsize=12)
ax.set_title('Fitting Uncertainty', fontsize=14)
ax.grid(True, alpha=0.3)

# Plot 4: γ histogram
ax = axes[1, 1]
gamma_valid = gamma_map[~np.isnan(gamma_map)]
ax.hist(gamma_valid, bins=30, alpha=0.7, edgecolor='black')
ax.set_xlabel('Topological Entropy γ', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of γ Values', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(gamma_valid.mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {gamma_valid.mean():.2f}')
ax.legend()

plt.tight_layout()
plt.savefig('notebooks/plots/entanglement_phase_diagram.png',
            dpi=300, bbox_inches='tight')
print("Main plot saved to notebooks/plots/entanglement_phase_diagram.png")

# Additional plot: Comparison with winding number
print("\nLoading winding number data for comparison...")
try:
    winding_data = np.load('data/winding_map.npy')
    w_lambda1 = np.load('data/lambda1_range.npy')
    w_lambda2 = np.load('data/lambda2_range.npy')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Winding number
    ax = axes[0]
    levels_w = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    contour_w = ax.contourf(w_lambda2, w_lambda1, winding_data,
                            levels=levels_w, cmap='RdBu_r', extend='both')
    cbar_w = plt.colorbar(contour_w, ax=ax, ticks=[-2, -1, 0, 1, 2])
    cbar_w.set_label('Winding Number ν', fontsize=12)
    ax.set_xlabel('λ₂', fontsize=12)
    ax.set_ylabel('λ₁', fontsize=12)
    ax.set_title('Topological Phase (Winding Number)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

    # Entanglement entropy
    ax = axes[1]
    contour_e = ax.contourf(lambda2_range, lambda1_range, gamma_map,
                            levels=20, cmap='viridis')
    cbar_e = plt.colorbar(contour_e, ax=ax)
    cbar_e.set_label('Topological Entropy γ', fontsize=12)
    ax.set_xlabel('λ₂', fontsize=12)
    ax.set_ylabel('λ₁', fontsize=12)
    ax.set_title('Topological Phase (Entanglement)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/plots/winding_vs_entanglement.png',
                dpi=300, bbox_inches='tight')
    print("Comparison plot saved to notebooks/plots/winding_vs_entanglement.png")

except FileNotFoundError:
    print("Winding number data not found, skipping comparison plot")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total points computed: {n_points}")
print(f"Successful fits: {np.sum(~np.isnan(gamma_map))}")
print(f"Failed fits: {np.sum(np.isnan(gamma_map))}")
print(f"\nγ statistics:")
print(f"  Mean: {gamma_valid.mean():.3f}")
print(f"  Std:  {gamma_valid.std():.3f}")
print(f"  Min:  {gamma_valid.min():.3f}")
print(f"  Max:  {gamma_valid.max():.3f}")
print(f"\nCentral charge statistics:")
c_valid = central_charge_map[~np.isnan(central_charge_map)]
print(f"  Mean: {c_valid.mean():.3f}")
print(f"  Std:  {c_valid.std():.3f}")
print("=" * 70)
