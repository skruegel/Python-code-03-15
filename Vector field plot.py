import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Parameters ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_gradient(x, A, y):
    y_pred = h(np.dot(A, x))
    error = y_pred - y
    dsigmoid = alpha * y_pred * (1 - y_pred)
    return 2 * np.dot(A.T, (error * dsigmoid))

# --- Define Matrices ---
A1 = np.eye(2) # Identity

k2 = 0.5
A2_raw = np.array([[1.0, k2], [k2, 1.0]])
A2 = A2_raw / A2_raw.sum(axis=1, keepdims=True)

k3 = 0.95 # Higher coupling to show the "narrow valley"
A3_raw = np.array([[1.0, k3], [k3, 1.0]])
A3 = A3_raw / A3_raw.sum(axis=1, keepdims=True)

matrices = [A1, A2, A3]
titles = ['Identity (No Coupling)', 'Moderate Coupling (k=0.5)', 'High Coupling (k=0.95)']

# --- Plotting ---
N = 15
x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# FIXED: Use plt.subplots instead of plt.figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, A_curr in enumerate(matrices):
    U = np.zeros_like(x1)
    V = np.zeros_like(x2)
    Mag = np.zeros_like(x1)

    for r in range(N):
        for col in range(N):
            g = get_gradient(np.array([x1[r,col], x2[r,col]]), A_curr, y_target)
            U[r,col], V[r,col] = -g[0], -g[1] # Negative gradient for descent direction
            Mag[r,col] = np.sqrt(g[0]**2 + g[1]**2)

    # Normalize vectors to show direction clearly in flat regions
    U_n = U / (Mag + 1e-10)
    V_n = V / (Mag + 1e-10)

    # Plot Quiver: Color determined by Magnitude (log scale often looks better)
    q = axes[i].quiver(x1, x2, U_n, V_n, Mag, cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=Mag.max()))
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')
    axes[i].set_aspect('equal')

fig.colorbar(q, ax=axes, label='Gradient Magnitude (Log Scale)')
plt.show()