#!/usr/bin/env python3
"""Run finite-size scaling analysis."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from finite_size_scaling import run_full_scaling_analysis

# Test point: known topological phase
lambda_1 = 1.0
lambda_2 = -1.2

# System sizes to test
sizes = [20, 50, 100, 200, 400]

print("Running finite-size scaling analysis...")
results = run_full_scaling_analysis(lambda_1, lambda_2, sizes=sizes)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BULK Gap vs 1/L
ax = axes[0]
ax.plot(1/np.array(results['sizes']), results['gaps'], 'o-', markersize=8, label='Bulk gap', linewidth=2)

# Add fit line if available
if 'fit_params' in results and 'exponent' in results['fit_params']:
    from scipy.optimize import curve_fit
    def power_law(L, gap_inf, a, alpha):
        return gap_inf + a / L**alpha

    params = results['fit_params']
    L_fine = np.linspace(min(results['sizes']), max(results['sizes']), 100)
    fit_curve = power_law(L_fine, params['gap_infinity'], params['amplitude'], params['exponent'])
    ax.plot(1/L_fine, fit_curve, '--', color='red', linewidth=2,
            label=f'Fit: Δ∞={params["gap_infinity"]:.4f}, α={params["exponent"]:.2f}')

ax.set_xlabel('1/L', fontsize=12)
ax.set_ylabel('Bulk Energy Gap Δ', fontsize=12)
ax.set_title(f'Bulk Gap Scaling (λ₁={lambda_1}, λ₂={lambda_2})', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Localization lengths
ax = axes[1]
edge_data = results['edge_analysis']
valid = ~np.isnan(edge_data['localization_lengths'])
if np.any(valid):
    ax.plot(np.array(edge_data['sizes'])[valid], np.array(edge_data['localization_lengths'])[valid], 's-', markersize=8, linewidth=2)
    ax.axhline(y=np.mean(np.array(edge_data['localization_lengths'])[valid]),
               color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(np.array(edge_data["localization_lengths"])[valid]):.1f} sites')
ax.set_xlabel('System Size L', fontsize=12)
ax.set_ylabel('Localization Length ξ (sites)', fontsize=12)
ax.set_title('Edge Mode Localization', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/plots/finite_size_scaling.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to notebooks/plots/finite_size_scaling.png")

# Save data
np.savez('data/scaling_results.npz',
         lambda_1=lambda_1,
         lambda_2=lambda_2,
         mu=results['mu'],
         sizes=results['sizes'],
         gaps=results['gaps'],  # Now contains BULK gaps
         n_zero_modes=results['n_zero_modes'],
         extrapolated_gap=results['extrapolated_gap'],
         fit_params=results['fit_params'])
print(f"Data saved to data/scaling_results.npz")
print(f"\nKey result: Bulk gap extrapolates to Δ∞ = {results['extrapolated_gap']:.6f}")
