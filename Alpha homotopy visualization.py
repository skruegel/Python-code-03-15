import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
c = 0.5
y_target = np.array([1.0, 0.0])
# Matrix A: Moderate Coupling
k = 0.5
A_raw = np.array([[1.0, k], [k, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z, alpha):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, alpha):
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y_target - y_pred)**2)

# Alpha Schedule: From "Convex-like" to "Very Stiff"
alpha_steps = [1.0, 5.0, 15.0, 40.0]
titles = [f'Alpha = {a} (Smooth)' if a < 5 else f'Alpha = {a} (Stiff)' for a in alpha_steps]

# --- Visualization ---
fig = plt.figure(figsize=(16, 12))
N = 50
x1_vals = np.linspace(0, 1, N)
x2_vals = np.linspace(0, 1, N)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

for i, alpha_curr in enumerate(alpha_steps):
    Z = np.zeros_like(X1)
    for r in range(N):
        for col in range(N):
            Z[r, col] = get_cost(np.array([X1[r, col], X2[r, col]]), alpha_curr)
            
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='magma', alpha=0.8, edgecolor='none')
    
    # Labeling
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Cost $J(x)$')
    ax.set_zlim(0, 2)
    ax.view_init(elev=30, azim=-60)

plt.suptitle('Alpha Homotopy: Deforming the Non-Convex Landscape', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()