#!/usr/bin/env python3
"""Analyze disorder robustness of Majorana zero modes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from disorder import measure_zero_mode_survival, disorder_average_spectrum

# Test point: topological phase
lambda_1 = 1.0
lambda_2 = -1.2

print("=" * 60)
print("Disorder Robustness Analysis")
print("=" * 60)
print(f"Testing point: (λ₁, λ₂) = ({lambda_1}, {lambda_2})")
print(f"System size: n = 100")
print(f"Realizations per disorder strength: 50\n")

# Disorder strengths to test
W_range = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])

print("Computing disorder averaging...")
results = measure_zero_mode_survival(lambda_1, lambda_2, W_range,
                                     n_realizations=50, n=100)

# Plot survival probability and gap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Survival probability
ax = axes[0]
ax.plot(results['W_values'], results['survival_prob'], 'o-',
        markersize=8, linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', label='50% threshold')
ax.set_xlabel('Disorder Strength W', fontsize=12)
ax.set_ylabel('Zero-Mode Survival Probability', fontsize=12)
ax.set_title('Topological Protection vs Disorder', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Average gap with error bars
ax = axes[1]
ax.errorbar(results['W_values'], results['mean_gap'],
            yerr=results['std_gap'], fmt='s-', markersize=8,
            capsize=5, linewidth=2)
ax.set_xlabel('Disorder Strength W', fontsize=12)
ax.set_ylabel('Mean Energy Gap', fontsize=12)
ax.set_title('Gap Evolution Under Disorder', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/plots/disorder_robustness.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to notebooks/plots/disorder_robustness.png")

# Spectrum for weak vs strong disorder
print("\nComputing eigenvalue distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, W in enumerate([0.0, 0.1, 0.5]):
    ax = axes[i]
    eigvals = disorder_average_spectrum(lambda_1, lambda_2, W,
                                       n_realizations=30, n=100)
    ax.hist(eigvals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Energy', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'W = {W}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-8, 8)

plt.tight_layout()
plt.savefig('notebooks/plots/disorder_spectrum.png', dpi=300, bbox_inches='tight')
print(f"Spectrum plot saved to notebooks/plots/disorder_spectrum.png")

# Save results
np.savez('data/disorder_results.npz',
         W_values=results['W_values'],
         survival_prob=results['survival_prob'],
         mean_gap=results['mean_gap'],
         std_gap=results['std_gap'])
print(f"Data saved to data/disorder_results.npz")

print("\n" + "=" * 60)
print("Key findings:")
critical_idx = np.argmin(np.abs(results['survival_prob'] - 0.5))
W_critical = results['W_values'][critical_idx]
print(f"  Critical disorder: W_c ≈ {W_critical:.2f}")
print(f"  Zero modes survive up to W ≈ {W_critical:.2f}")
print(f"  At W=0.5: survival = {results['survival_prob'][W_range==0.5][0]:.1%}")
print("=" * 60)
