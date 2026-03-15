import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters (Eq. 2) ---
alpha = 10.0   # Stiffness parameter [cite: 79]
c = 0.5        # Cutoff threshold [cite: 79]
A = 1.0        # Simplified 1x1 convolution matrix [cite: 76]
y = 1.0        # Target output [cite: 83]

# --- Function Definitions (Eq. 1 & 3) ---
def sigmoid(z, alpha, c):
    """Sigmoid activation function h(z) [cite: 78]"""
    return 1 / (1 + np.exp(-alpha * (z - c)))

def cost_function(x, A, y, alpha, c):
    """Cost function J(x) = ||y - f_theta(x)||^2 [cite: 82]"""
    f_theta_x = sigmoid(A * x, alpha, c)
    return (y - f_theta_x)**2

# --- Generating Data for Plotting ---
# Feasible domain for x is 0 to 1 [cite: 81]
x_values = np.linspace(0, 1, 500)
j_values = cost_function(x_values, A, y, alpha, c)

# --- Plotting the Landscape ---
plt.figure(figsize=(10, 6))
plt.plot(x_values, j_values, label=f'$J(x)$ with $\\alpha={alpha}, c={c}$', color='blue', linewidth=2)

# Highlighting the minimum
min_idx = np.argmin(j_values)
plt.scatter(x_values[min_idx], j_values[min_idx], color='red', zorder=5, label='Minimum $x^*$')

# Labels and styling
plt.title('Optimization Landscape of $J(x)$ for $N=1$', fontsize=14)
plt.xlabel('Input $x$', fontsize=12)
plt.ylabel('Cost $J(x)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.ylim(-0.1, max(j_values) + 0.1)
plt.show()