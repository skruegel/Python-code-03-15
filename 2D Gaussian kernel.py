import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Project Parameters (from Eq. 2 & 3) ---
alpha = 15.0      # Stiffness parameter [cite: 20]
c = 0.5           # Cutoff threshold [cite: 20]
y_target = np.array([1, 0])  # Binary target vector y [cite: 24]

# --- 1. Define Gaussian Kernel Matrix A ---
# For N=2, we simulate a Gaussian blur where the input influences itself 
# more than its neighbor.
# sigma determines the "spread".
sigma = 1.0
weights = np.array([np.exp(-0**2/(2*sigma**2)), np.exp(-1**2/(2*sigma**2))])
weights /= np.sum(weights) # Normalize so rows sum to 1

# A is an NxN convolution matrix [cite: 17]
A = np.array([[weights[0], weights[1]],
              [weights[1], weights[0]]])

# --- Function Definitions ---
def h(z, alpha, c):
    """Sigmoid activation function h(z) [cite: 19]"""
    return 1 / (1 + np.exp(-alpha * (z - c)))

def cost_function(x1, x2, A, y, alpha, c):
    """J(x) = ||y - h(Ax)||^2 [cite: 23]"""
    x = np.array([x1, x2])
    Ax = np.dot(A, x)
    f_theta_x = h(Ax, alpha, c)
    return np.sum((y - f_theta_x)**2)

# --- 2. Create Grid for x1, x2 between 0 and 1 ---
# The input domain is constrained: 0 <= x <= 1 [cite: 22]
N_grid = 100
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# --- 3. Calculate J(x) for the Grid ---
Z = np.zeros_like(X1)
for i in range(N_grid):
    for j in range(N_grid):
        Z[i, j] = cost_function(X1[i, j], X2[i, j], A, y_target, alpha, c)

# --- 4. 3D Plotting ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='magma', edgecolor='none', alpha=0.9)

ax.set_title(f'Optimization Landscape with Gaussian Kernel ($\sigma={sigma}$)', fontsize=14)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$J(x)$')
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()