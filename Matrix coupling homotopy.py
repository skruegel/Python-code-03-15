import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
alpha_fixed = 10.0 # Keep alpha moderate to focus on matrix effects
c = 0.5
y_target = np.array([1.0, 0.0])

def h(z):
    return 1 / (1 + np.exp(-alpha_fixed * (z - c)))

def get_cost(x, A):
    y_pred = h(np.dot(A, x))
    return np.sum((y_target - y_pred)**2)

# K Schedule: From Independent (Identity) to Heavy Blur (Singular)
k_steps = [0.0, 0.4, 0.7, 0.95]
titles = [f'k = {k} (Identity)' if k == 0 else f'k = {k} (High Coupling)' for k in k_steps]

# --- Visualization ---
fig = plt.figure(figsize=(16, 12))
N = 50
x1_vals = np.linspace(0, 1, N)
x2_vals = np.linspace(0, 1, N)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

for i, k_curr in enumerate(k_steps):
    # Construct and Normalize Matrix A
    A_raw = np.array([[1.0, k_curr], [k_curr, 1.0]])
    A_curr = A_raw / A_raw.sum(axis=1, keepdims=True)
    
    Z = np.zeros_like(X1)
    for r in range(N):
        for col in range(N):
            Z[r, col] = get_cost(np.array([X1[r, col], X2[r, col]]), A_curr)
            
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Draw a contour on the floor to show the changing "footprint"
    ax.contourf(X1, X2, Z, zdir='z', offset=-0.2, cmap='viridis', alpha=0.3)
    
    # Labeling
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Cost $J(x)$')
    ax.set_zlim(-0.2, 2.0)
    ax.view_init(elev=30, azim=45)

plt.suptitle('Matrix Homotopy: Stretching the Optimization Valley', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()