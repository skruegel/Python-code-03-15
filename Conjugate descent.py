import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Project Parameters ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])
num_runs = 100
iterations = 100

# --- Define and Normalize Matrix A ---
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, A, y):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    return np.sum((y - y_pred)**2)

def get_gradient(x, A, y):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    error = y_pred - y
    dsigmoid = alpha * y_pred * (1 - y_pred)
    return 2 * np.dot(A.T, (error * dsigmoid))

# --- Conjugate Gradient (Polak-Ribiere) for Non-Convex Optimization ---
all_paths = []
all_costs = []

for _ in range(num_runs):
    x = np.random.rand(2)
    path = [x.copy()]
    costs = [get_cost(x, A, y_target)]
    
    # Initial setup
    g = get_gradient(x, A, y_target)
    d = -g  # Initial search direction
    
    for i in range(iterations):
        if np.linalg.norm(g) < 1e-7:
            break
            
        # Line search: a simple constant or small step for this complex surface
        # In formal CG, this would be an exact line search
        step_size = 0.05
        
        new_x = np.clip(x + step_size * d, 0, 1)
        new_g = get_gradient(new_x, A, y_target)
        
        # Polak-Ribiere update for beta
        # beta = (g_new^T * (g_new - g_old)) / (g_old^T * g_old)
        num = np.dot(new_g, new_g - g)
        den = np.dot(g, g)
        beta = max(0, num / (den + 1e-9))
        
        # Update direction
        d = -new_g + beta * d
        
        # Update variables
        x = new_x
        g = new_g
        
        path.append(x.copy())
        costs.append(get_cost(x, A, y_target))
        
        if np.linalg.norm(path[-1] - path[-2]) < 1e-8:
            break

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
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Cost J(x)')

for i in range(num_runs):
    path = all_paths[i]
    costs = all_costs[i]
    ax.plot(path[:, 0], path[:, 1], costs, color='red', alpha=0.15, linewidth=0.5)
    ax.scatter(path[-1, 0], path[-1, 1], costs[-1], 
               color='yellow', marker='^', s=80, edgecolors='black', zorder=10)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Cost J(x)')
ax.set_title(r"3D Landscape: Conjugate Gradient (alpha=25)")
ax.view_init(elev=30, azim=-60)
plt.show()