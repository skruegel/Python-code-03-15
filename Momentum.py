import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Updated Project Parameters ---
alpha = 25.0      # High stiffness
c = 0.5           # Threshold
y_target = np.array([1.0, 0.0])
learning_rate = 0.05 # Lowered slightly for stability with momentum
beta = 0.9           # Momentum parameter
iterations = 250
num_runs = 100    

# --- Define and Normalize Matrix A ---
A_raw = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, A, y):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    return np.sum((y - y_pred)**2)

def get_gradient(x, A, y, alpha, c):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    error = y_pred - y
    dsigmoid = alpha * y_pred * (1 - y_pred)
    grad = 2 * np.dot(A.T, (error * dsigmoid))
    return grad

# --- Run Optimization with Momentum ---
all_paths = []
all_costs = []

for _ in range(num_runs):
    x = np.random.rand(2) 
    v = np.zeros(2) # Initial velocity
    path = [x.copy()]
    costs = [get_cost(x, A, y_target)]
    
    for i in range(iterations):
        grad = get_gradient(x, A, y_target, alpha, c)
        
        # --- MOMENTUM UPDATE ---
        v = beta * v + grad
        new_x = np.clip(x - learning_rate * v, 0, 1)
        
        if np.linalg.norm(new_x - x) < 1e-6:
            break
        x = new_x
        path.append(x.copy())
        costs.append(get_cost(x, A, y_target))
    
    all_paths.append(np.array(path))
    all_costs.append(np.array(costs))

# --- 3D Visualization ---
N_grid = 60
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(N_grid):
    for j in range(N_grid):
        xi = np.array([X1[i,j], X2[i,j]])
        Z[i,j] = get_cost(xi, A, y_target)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.4, edgecolor='none')
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Cost J(x)')

# Plot Paths and Yellow Triangle Final Points
for i in range(num_runs):
    path = all_paths[i]
    costs = all_costs[i]
    # Path lines
    ax.plot(path[:, 0], path[:, 1], costs, color='red', alpha=0.15, linewidth=0.5)
    # Final yellow triangles
    ax.scatter(path[-1, 0], path[-1, 1], costs[-1], 
               color='yellow', marker='^', s=80, edgecolors='black', linewidths=1, zorder=10)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Cost J(x)')
ax.set_title('3D Landscape with Momentum (beta=0.9, alpha=25)')

# Orientation: (x1=1, x2=0) in front
ax.view_init(elev=30, azim=-60)

plt.show()