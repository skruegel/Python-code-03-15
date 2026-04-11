import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Problem Setup ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])
iterations = 100
num_runs = 15

# Matrix A: Normalized
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x):
    y_pred = h(np.dot(A, x))
    return np.sum((y_target - y_pred)**2)

def get_gradient_and_hessian(x):
    z = np.dot(A, x)
    y_p = h(z)
    error = y_p - y_target
    hp = alpha * y_p * (1 - y_p)
    # Second derivative of sigmoid for the Hessian
    hpp = alpha**2 * y_p * (1 - y_p) * (1 - 2 * y_p)
    
    g = 2 * np.dot(A.T, error * hp)
    # Gauss-Newton approximation is often used, but here we use the full Hessian
    diag_elements = hp**2 + error * hpp
    H = 2 * np.dot(A.T * diag_elements, A)
    return g, H

# --- 2. Optimization Loop ---
all_paths = []
all_costs = []

for _ in range(num_runs):
    x = np.random.rand(2)
    lam = 0.1  # Initial damping
    path = [x.copy()]
    costs = [get_cost(x)]
    
    for i in range(iterations):
        curr_cost = get_cost(x)
        g, H = get_gradient_and_hessian(x)
        
        # Levenberg-Marquardt Update: (H + lambda*I)dx = -g
        try:
            H_damped = H + lam * np.eye(2)
            dx = np.linalg.solve(H_damped, -g)
        except np.linalg.LinAlgError:
            break
            
        new_x = np.clip(x + dx, 0, 1)
        new_cost = get_cost(new_x)
        
        if new_cost < curr_cost:
            # Improvement: Accept step and decrease damping
            x = new_x
            lam /= 10
            path.append(x.copy())
            costs.append(new_cost)
        else:
            # No improvement: Increase damping (behave more like Gradient Descent)
            lam *= 10
            
        if np.linalg.norm(g) < 1e-6:
            break
            
    all_paths.append(np.array(path))
    all_costs.append(np.array(costs))

# --- 3. 3D Visualization ---
N_grid = 60
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)
for i in range(N_grid):
    for j in range(N_grid):
        Z[i, j] = get_cost(np.array([X1[i,j], X2[i,j]]))

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.4, edgecolor='none')

for i in range(len(all_paths)):
    p = all_paths[i]
    c_list = all_costs[i]
    ax.plot(p[:, 0], p[:, 1], c_list, color='red', alpha=0.3, linewidth=1.5)
    ax.scatter(p[-1, 0], p[-1, 1], c_list[-1], color='yellow', marker='^', s=60, edgecolors='black')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Cost J(x)')
ax.set_title('Levenberg-Marquardt Optimization Paths')
ax.view_init(elev=30, azim=-60)

plt.show()