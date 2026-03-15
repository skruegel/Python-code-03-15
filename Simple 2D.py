import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Project Parameters ---
alpha = 15.0  # Stiffness parameter (from Eq. 2)
c = 0.5       # Cutoff threshold (from Eq. 2)
y_target = np.array([1, 0])  # Target vector y

# Define a simple 2x2 averaging kernel matrix A
# A convolution matrix A such that Ax represents local averages
A = np.array([[0.5, 0.5],
              [0.5, 0.5]])

# --- Function Definitions ---
def h(z, alpha, c):
    """Sigmoid activation function (Eq. 2)"""
    return 1 / (1 + np.exp(-alpha * (z - c)))

def J(x1, x2, A, y_target, alpha, c):
    """Cost function J(x) = ||y - h(Ax)||^2 (Eq. 3)"""
    x = np.array([x1, x2])
    # Calculate Ax
    Ax = np.dot(A, x)
    # Calculate f_theta(x) = h(Ax)
    f_theta_x = h(Ax, alpha, c)
    # Calculate squared Euclidean distance (L2 norm squared)
    return np.sum((y_target - f_theta_x)**2)

# 1) Define a grid of points for x1 and x2 between 0 and 1
N_grid = 100
x1_range = np.linspace(0, 1, N_grid)
x2_range = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_range, x2_range)

# 2 & 3) Calculate J(x) for every pair (x1, x2)
# Vectorizing the cost function for the grid
Z = np.zeros_like(X1)
for i in range(N_grid):
    for j in range(N_grid):
        Z[i, j] = J(X1[i, j], X2[i, j], A, y_target, alpha, c)

# 4) Produce a 3D plot of J(x) versus (x1, x2)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Labels and formatting
ax.set_title('Optimization Landscape $J(x)$ for $N=2$', fontsize=14)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_zlabel('$J(x)$ (Cost)', fontsize=12)
fig.colorbar(surface, shrink=0.5, aspect=10, label='Cost Value')

plt.show()