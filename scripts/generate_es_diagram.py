#!/usr/bin/env python3
"""
[DEPRECATED] Generate entanglement spectrum degeneracy phase diagram.

WARNING: This script does NOT reliably discriminate topological phases in 1D.

Uses periodic boundary conditions to avoid edge-mode hybridization.
Computes ES gap as attempted diagnostic for topological phases.

ISSUES:
1. All 1600 points show identical degeneracy (#degen = 19 everywhere)
2. Min ES gap varies by less than 1000× across all phases
3. Pattern does not match winding number phase boundaries
4. ES degeneracy is NOT a universal topological diagnostic in 1D free fermions

REASON:
For 1D Majorana chains, entanglement spectrum structure is determined by
single-particle modes, not many-body topological properties. Unlike 2D
systems, there's no clean ES degeneracy signature in 1D.

USE INSTEAD:
- scripts/generate_winding_diagram.py (winding number - rigorous)
- scripts/run_scaling_analysis.py (gap scaling - validates topology)

This script is preserved for reference only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from entanglement import compute_es_degeneracy_measure
import time

print("=" * 70)
print("ENTANGLEMENT SPECTRUM DEGENERACY PHASE DIAGRAM")
print("=" * 70)

# Grid parameters
lambda1_range = np.linspace(-3, 3, 40)
lambda2_range = np.linspace(-3, 3, 40)
L = 150  # System size (PBC)

n_points = len(lambda1_range) * len(lambda2_range)
print(f"\nGrid: {len(lambda1_range)} × {len(lambda2_range)} = {n_points} points")
print(f"System size: L={L} (PBC)")
print(f"Estimated time: ~5-10 minutes\n")
print("Starting computation...\n")

# Storage arrays
min_gap_map = np.zeros((len(lambda1_range), len(lambda2_range)))
n_degen_map = np.zeros((len(lambda1_range), len(lambda2_range)))

# Track progress
start_time = time.time()

for i, l1 in enumerate(lambda1_range):
    row_start = time.time()

    for j, l2 in enumerate(lambda2_range):
        try:
            # Compute ES degeneracy with PBC
            degeneracy, n_degen, min_gap = compute_es_degeneracy_measure(
                l1, l2, L=L, mu=0.0, pbc=True
            )

            min_gap_map[i, j] = min_gap
            n_degen_map[i, j] = n_degen

        except Exception as e:
            # If computation fails, mark as NaN
            min_gap_map[i, j] = np.nan
            n_degen_map[i, j] = 0

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
        np.savez('data/es_diagram_partial.npz',
                 min_gap_map=min_gap_map,
                 n_degen_map=n_degen_map,
                 lambda1_range=lambda1_range,
                 lambda2_range=lambda2_range,
                 completed_rows=i+1)
        print(f"  → Checkpoint saved ({i+1} rows)")

total_time = time.time() - start_time
print(f"\nCompleted in {total_time/60:.1f} minutes")

# Save final results
np.savez('data/es_diagram.npz',
         min_gap_map=min_gap_map,
         n_degen_map=n_degen_map,
         lambda1_range=lambda1_range,
         lambda2_range=lambda2_range)
print("Data saved to data/es_diagram.npz")

# Generate plots
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Minimum ES gap (small = degenerate = topological)
ax = axes[0]
# Use log scale for better visualization
min_gap_plot = np.log10(min_gap_map + 1e-12)
im1 = ax.contourf(lambda2_range, lambda1_range, min_gap_plot,
                  levels=20, cmap='viridis_r')
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('log₁₀(Min ES Gap)', fontsize=12)
ax.set_xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
ax.set_ylabel('λ₁ (nearest neighbor)', fontsize=12)
ax.set_title('ES Degeneracy: Min Gap (PBC)', fontsize=14, weight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: Number of degenerate pairs
ax = axes[1]
im2 = ax.contourf(lambda2_range, lambda1_range, n_degen_map,
                  levels=20, cmap='plasma')
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('# Degenerate Pairs', fontsize=12)
ax.set_xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
ax.set_ylabel('λ₁ (nearest neighbor)', fontsize=12)
ax.set_title('ES Degeneracy: Pair Count (PBC)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/plots/es_degeneracy_diagram.png',
            dpi=300, bbox_inches='tight')
print("Plot saved to notebooks/plots/es_degeneracy_diagram.png")

# Comparison with winding number
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

    # ES degeneracy
    ax = axes[1]
    im_es = ax.contourf(lambda2_range, lambda1_range, min_gap_plot,
                        levels=20, cmap='viridis_r')
    cbar_es = plt.colorbar(im_es, ax=ax)
    cbar_es.set_label('log₁₀(Min ES Gap)', fontsize=12)
    ax.set_xlabel('λ₂', fontsize=12)
    ax.set_ylabel('λ₁', fontsize=12)
    ax.set_title('Topological Phase (ES Degeneracy, PBC)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/plots/winding_vs_es.png',
                dpi=300, bbox_inches='tight')
    print("Comparison plot saved to notebooks/plots/winding_vs_es.png")

except FileNotFoundError:
    print("Winding number data not found, skipping comparison plot")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total points computed: {n_points}")
valid = ~np.isnan(min_gap_map)
print(f"Successful: {np.sum(valid)}")
print(f"Failed: {np.sum(~valid)}")

if np.any(valid):
    print(f"\nMin ES gap statistics:")
    print(f"  Mean: {np.nanmean(min_gap_map):.3e}")
    print(f"  Std:  {np.nanstd(min_gap_map):.3e}")
    print(f"  Min:  {np.nanmin(min_gap_map):.3e}")
    print(f"  Max:  {np.nanmax(min_gap_map):.3e}")

    print(f"\n# Degenerate pairs statistics:")
    print(f"  Mean: {np.nanmean(n_degen_map):.1f}")
    print(f"  Std:  {np.nanstd(n_degen_map):.1f}")
    print(f"  Min:  {np.nanmin(n_degen_map):.0f}")
    print(f"  Max:  {np.nanmax(n_degen_map):.0f}")

print("=" * 70)
