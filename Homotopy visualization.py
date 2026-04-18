import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# --- Parameters ---
c_val = 0.5
y_target = np.array([1.0, 0.0]) # Target vector y
N_grid = 60 # Resolution of the cost surface

def h(z, alpha):
    """Sigmoid activation function (Eq. 2)"""
    return 1 / (1 + np.exp(-alpha * (z - c_val)))

def get_cost(x, A, alpha):
    """L2 Cost J(x, alpha, k) (Eq. 3)"""
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y_target - y_pred)**2)

# Define the Homotopy "Snapshots" (alpha, k)
phases = [
    (1.0, 0.0),   # Phase 1: Perfect Bowl (Start)
    (10.0, 0.0),  # Phase 2: Stiffer Bowl
    (1.0, 0.8),   # Phase 3: Smooth Ellipse
    (10.0, 0.8),  # Phase 4: Stiff & Coupled (Target)
]

titles = [
    'Start (alpha=1, k=0): Smooth Bowl',
    'Alpha Homotopy (alpha=10, k=0): Stiff mesa',
    'Matrix Homotopy (alpha=1, k=0.8): Ellipsoid',
    'Target (alpha=10, k=0.8): Narrow Canyon'
]

# --- Visualization ---
fig = plt.figure(figsize=(18, 12))
norm = Normalize(vmin=0, vmax=2) # Consistent cost color scale

# Generate Grid
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

for i, (alpha_curr, k_curr) in enumerate(phases):
    Z = np.zeros_like(X1)
    
    # Construct Normalized Matrix A for this k
    A_raw = np.array([[1.0, k_curr], [k_curr, 1.0]])
    A_curr = A_raw / A_raw.sum(axis=1, keepdims=True)
    
    # Calculate Cost at each grid point
    for r in range(N_grid):
        for col in range(N_grid):
            xi = np.array([X1[r, col], X2[r, col]])
            Z[r, col] = get_cost(xi, A_curr, alpha_curr)
            
    # Subplot placement (2x2 grid)
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    
    # Plot Surface (Color determined by magnitude, slight transparency)
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', norm=norm, alpha=0.6, edgecolor='none')
    
    # Add Contours to clarify the "canyon" shape
    ax.contourf(X1, X2, Z, zdir='z', offset=-0.1, cmap='viridis', norm=norm, alpha=0.3)
    
    # Target Marker
    ax.scatter(0.95, 0.05, 0, color='red', marker='*', s=200, zorder=10)

    # Plot Settings
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost $J(x)$')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(-0.1, 2.1)
    ax.view_init(elev=30, azim=45)

plt.suptitle('Dual Homotopy Strategy: Dynamic Cost Surface Evolution', fontsize=18, y=0.95)
plt.tight_layout(pad=3.0)
plt.show()