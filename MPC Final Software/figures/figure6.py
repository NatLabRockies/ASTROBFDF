"""
Generate Figure 6
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x_i values
x0 = np.linspace(-5, 10, 50)
x1 = np.linspace(-5, 10, 50)
X0, X1 = np.meshgrid(x0, x1)

# Define the correlation coefficient
cor = 0  # You can change this value as needed

# Calculate deterministic_hfn for each combination of x0 and x1
deterministic_hfn = (
    X1 - (5.1 * X0**2 / (4 * np.pi**2)) + (5 * X0 / np.pi) - 6 +
    10 * (1 - 1 / (8 * np.pi)) * np.cos(X0) + 10
)

# Calculate hfn_eval_at_x and lfn_eval_at_x for each combination of x0 and x1
hfn_eval_at_x = deterministic_hfn
lfn_eval_at_x = (
    deterministic_hfn - (0.5 * cor**2 -2 * cor + 1.7) *
    (X1 - 5.1 * X0**2 / (4 * np.pi**2) + 5 * X0 / np.pi - 6)**2
)

# Create subplots for hfn_eval_at_x and lfn_eval_at_x
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X0, X1, hfn_eval_at_x, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x[1]',  fontsize=18)
plt.ylabel('x[2]',  fontsize=18)
plt.title('High-Fidelity Objective Function',  fontsize=18)

plt.subplot(1, 2, 2)
plt.contourf(X0, X1, lfn_eval_at_x, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x[1]',  fontsize=18)
plt.ylabel('x[2]',  fontsize=18)
plt.title('Low-Fidelity Objective Function',  fontsize=18)

plt.tight_layout()
plt.savefig("experiments/plots/figure6.pdf", format='pdf')