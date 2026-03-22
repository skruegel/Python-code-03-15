import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Problem Setup (Problem Formulation Section) ---
alpha = 25.0       # Stiffness parameter [cite: 20]
c = 0.5            # Threshold [cite: 20]
y_target = np.array([1.0, 0.0]) # Target vector y [cite: 24]
learning_rate = 0.1
iterations = 250
num_runs = 100     # Multiple trials [cite: 79]

# Matrix A: Normalized [1, 0.5; 0.5, 1] as requested
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    """Sigmoid activation function (Eq. 2) [cite: 19]"""
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, A, y):
    """Cost function J(x) (Eq. 3) [cite: 23]"""
    y_pred = h(np.dot(A, x))
    return np.sum((y - y_pred)**2)

def get_gradient(x, A, y):
    """Gradient of J(x) with respect to input x [cite: 29]"""
    y_pred = h(np.dot(A, x))
    error = y_pred - y
    # Derivative of sigmoid: alpha * h(z) * (1 - h(z))
    dsigmoid = alpha * y_pred * (1 - y_pred)
    # Chain rule application
    return 2 * np.dot(A.T, (error * dsigmoid))

# --- 2. Optimization Loop (Task Section)  ---
all_paths = []
all_costs = []

for _ in range(num_runs):
    x = np.random.rand(2) # Random start in domain [0, 1] 
    path = [x.copy()]
    costs = [get_cost(x, A, y_target)]
    
    for _ in range(iterations):
        grad = get_gradient(x, A, y_target)
        
        # Gradient Update
        new_x = x - learning_rate * grad
        
        # PROJECTION STEP: Constrain x to [0, 1] 
        new_x = np.clip(new_x, 0, 1)
        
        if np.linalg.norm(new_x - x) < 1e-6:
            break
        x = new_x
        path.append(x.copy())
        costs.append(get_cost(x, A, y_target))
        
    all_paths.append(np.array(path))
    all_costs.append(np.array(costs))

# --- 3. Visualization (Landscape Analysis Section) [cite: 38, 60] ---
N_grid = 60
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(N_grid):
    for j in range(N_grid):
        Z[i, j] = get_cost(np.array([X1[i,j], X2[i,j]]), A, y_target)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.4, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Cost J(x)')

for i in range(num_runs):
    ax.plot(all_paths[i][:, 0], all_paths[i][:, 1], all_costs[i], color='red', alpha=0.15)
    # Final results: Yellow Triangles
    ax.scatter(all_paths[i][-1, 0], all_paths[i][-1, 1], all_costs[i][-1], 
               color='yellow', marker='^', s=80, edgecolors='black', zorder=10)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Cost $J(x)$')
ax.set_title(f'Projected Gradient Descent (alpha={alpha})')
ax.view_init(elev=30, azim=-60) # Oriented with x1=1, x2=0 in front

plt.show()