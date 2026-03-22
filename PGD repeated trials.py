import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 15.0      # Stiffness
c = 0.5           # Threshold
y_target = np.array([1.0, 0.0])
learning_rate = 0.1
iterations = 200
num_runs = 100    # Number of different starting points

# Gaussian-like Kernel A (2x2)
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_gradient(x, A, y, alpha, c):
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    error = y_pred - y
    dsigmoid = alpha * y_pred * (1 - y_pred)
    grad = 2 * np.dot(A.T, (error * dsigmoid))
    return grad

# --- Run Optimization 100 Times ---
all_paths = []

for _ in range(num_runs):
    x = np.random.rand(2) # Random start in [0, 1]
    path = [x.copy()]
    
    for i in range(iterations):
        grad = get_gradient(x, A, y_target, alpha, c)
        new_x = np.clip(x - learning_rate * grad, 0, 1)
        
        if np.linalg.norm(new_x - x) < 1e-5:
            break
        x = new_x
        path.append(x.copy())
    
    all_paths.append(np.array(path))

# --- Visualization ---
N_grid = 100
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

# Calculate Cost Landscape for the background
for i in range(N_grid):
    for j in range(N_grid):
        xi = np.array([X1[i,j], X2[i,j]])
        pred = h(np.dot(A, xi))
        Z[i,j] = np.sum((y_target - pred)**2)

plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
plt.colorbar(label='Cost $J(x)$')

# Plot all 100 paths
for i, path in enumerate(all_paths):
    # Plotting with low alpha (transparency) so we can see density
    plt.plot(path[:, 0], path[:, 1], color='white', alpha=0.3, linewidth=1)
    # Mark the final point of each path
    plt.scatter(path[-1, 0], path[-1, 1], color='red', s=5, alpha=0.5)

plt.title(f'100 Gradient Descent Paths ($\\alpha={alpha}, c={c}$)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()