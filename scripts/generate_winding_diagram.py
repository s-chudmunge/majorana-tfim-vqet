#!/usr/bin/env python3
"""Generate winding number phase diagram."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from topological_invariants import compute_winding_map

# Parameters
lambda1_range = np.linspace(-3, 3, 60)
lambda2_range = np.linspace(-3, 3, 60)

print("Computing winding number phase diagram...")
print(f"Grid: {len(lambda1_range)} x {len(lambda2_range)} = {len(lambda1_range)*len(lambda2_range)} points")
print("This may take a few minutes...\n")

winding_map = compute_winding_map(lambda1_range, lambda2_range)

# Save data
np.save('data/winding_map.npy', winding_map)
np.save('data/lambda1_range.npy', lambda1_range)
np.save('data/lambda2_range.npy', lambda2_range)
print(f"\nData saved to data/winding_map.npy")

# Plot
plt.figure(figsize=(10, 8))
levels = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
contour = plt.contourf(lambda2_range, lambda1_range, winding_map,
                       levels=levels, cmap='RdBu_r', extend='both')
cbar = plt.colorbar(contour, ticks=[-2, -1, 0, 1, 2])
cbar.set_label('Winding Number ν', fontsize=12)

plt.xlabel('λ₂ (next-nearest neighbor)', fontsize=12)
plt.ylabel('λ₁ (nearest neighbor)', fontsize=12)
plt.title('Topological Phase Diagram (Winding Number)', fontsize=14)
plt.grid(True, alpha=0.3)

# Mark the known topological point
plt.plot([-1.2], [1.0], 'k*', markersize=15, label='(λ₁=1.0, λ₂=-1.2)')
plt.legend()

plt.tight_layout()
plt.savefig('notebooks/plots/winding_number_phase_diagram.png', dpi=300, bbox_inches='tight')
print(f"Phase diagram saved to notebooks/plots/winding_number_phase_diagram.png")

plt.show()
