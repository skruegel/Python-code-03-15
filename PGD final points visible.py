import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 20.0      
c = 0.5           
y_target = np.array([1.0, 0.0])
learning_rate = 0.1
iterations = 250
num_runs = 100    

# Gaussian-like Kernel A (2x2)
sigma = 1.0
w0 = np.exp(-0**2/(2*sigma**2))
w1 = np.exp(-1**2/(2*sigma**2))
row_sum = w0 + w1
A = np.array([[w0/row_sum, w1/row_sum],
              [w1/row_sum, w0/row_sum]])

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_gradient(x, A, y, alpha, c):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    error = y_pred - y
    dsigmoid = alpha * y_pred * (1 - y_pred)
    grad = 2 * np.dot(A.T, (error * dsigmoid))
    return grad

# --- Run Optimization ---
all_paths = []
final_points = []

for _ in range(num_runs):
    x = np.random.rand(2) 
    path = [x.copy()]
    
    for i in range(iterations):
        grad = get_gradient(x, A, y_target, alpha, c)
        new_x = np.clip(x - learning_rate * grad, 0, 1)
        if np.linalg.norm(new_x - x) < 1e-5:
            break
        x = new_x
        path.append(x.copy())
    
    all_paths.append(np.array(path))
    final_points.append(x)

final_points = np.array(final_points)

# --- Visualization ---
N_grid = 100
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(N_grid):
    for j in range(N_grid):
        xi = np.array([X1[i,j], X2[i,j]])
        pred = h(np.dot(A, xi))
        Z[i,j] = np.sum((y_target - pred)**2)

plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.6)
plt.colorbar(label='Cost $J(x)$')

# 1. Plot the paths with very low alpha to highlight the "flow"
for path in all_paths:
    plt.plot(path[:, 0], path[:, 1], color='white', alpha=0.15, linewidth=1)

# 2. Plot the final points with large, visible markers
# Using 'x' markers with an edge for visibility
plt.scatter(final_points[:, 0], final_points[:, 1], 
            color='red', 
            marker='x', 
            s=60, 
            linewidths=2, 
            label='Final Converged Points',
            zorder=5)

# 3. Add a star for the "Ideal" solution if known (e.g., center of the best valley)
# For this cost function, let's just highlight the best one found
best_idx = np.argmin([np.sum((y_target - h(np.dot(A, p)))**2) for p in final_points])
plt.scatter(final_points[best_idx, 0], final_points[best_idx, 1], 
            color='yellow', marker='*', s=200, edgecolors='black', label='Best Found Solution', zorder=6)

plt.title(f'Optimization Convergence: 100 Runs ($\\alpha={alpha}$)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='upper right')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()