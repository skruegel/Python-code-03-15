import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# --- 1. System Setup (Problem Formulation) ---
c = 0.5
y_target = np.array([1.0, 0.0]) # Target vector y
target_alpha = 30.0            # Final "hard" target stiffness
num_runs = 50                  # Run multiple trials to see variation
learning_rate = 0.05
steps_per_alpha = 50

# Matrix A: Normalized [[1, 0.5], [0.5, 1]]
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

# Homotopy Schedule: Alpha grows from smooth to sharp
alpha_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, target_alpha]
total_steps = len(alpha_schedule) * steps_per_alpha

def h(z, alpha_val):
    """Sigmoid activation function (Eq. 2)"""
    return 1 / (1 + np.exp(-alpha_val * (z - c)))

def get_cost(x, alpha_val):
    """Cost function J(x) (Eq. 3)"""
    y_pred = h(np.dot(A, x), alpha_val)
    return np.sum((y_target - y_pred)**2)

def run_adam_step(x, alpha_val, m, v, t, lr):
    """Single update step of the Adam Optimizer"""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    # Gradient Calculation
    y_p = h(np.dot(A, x), alpha_val)
    grad = 2 * np.dot(A.T, ((y_p - y_target) * (alpha_val * y_p * (1 - y_p))))
    
    # Moment Updates and Bias Correction
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # Update and Projection (Clipping to [0, 1])
    x_new = np.clip(x - lr * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
    return x_new, m, v

# --- 2. Generate 3D Landscape (at Final Hardness) ---
# We compute the 3D surface grid once using target_alpha
print("Generating 3D surface at target alpha...")
N_grid = 60
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z_final = np.zeros_like(X1)

for r in range(N_grid):
    for col in range(N_grid):
        xi = np.array([X1[r, col], X2[r, col]])
        Z_final[r, col] = get_cost(xi, target_alpha)

# --- 3. Run Optimization Trials with Homotopy ---
all_paths_x = []
all_paths_cost = []

print(f"Starting {num_runs} Homotopy Adam trials...")
for run_id in range(num_runs):
    x = np.random.rand(2) # New random starting point in feasible domain
    m, v = np.zeros(2), np.zeros(2)
    t_global = 1
    
    path_x = [x.copy()]
    # CRITICAL: Track cost against the FINAL high alpha, not the current one
    path_cost = [get_cost(x, target_alpha)] 
    
    for current_alpha in alpha_schedule:
        for _ in range(steps_per_alpha):
            x, m, v = run_adam_step(x, current_alpha, m, v, t_global, learning_rate)
            t_global += 1
            
            path_x.append(x.copy())
            path_cost.append(get_cost(x, target_alpha))
            
    all_paths_x.append(np.array(path_x))
    all_paths_cost.append(np.array(path_cost))

print("Optimization complete.")

# --- 4. 3D Visualization of All Runs ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the FINAL cost surface (alpha = 30)
# We use a slight opacity (alpha=0.4) to make the paths visible inside the landscape.
surf = ax.plot_surface(X1, X2, Z_final, cmap='viridis', alpha=0.4, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Cost $J(x)$ (Final Target Alpha)')

# Plot all 50 optimization trajectories in 3D
# These paths show how x converges in 2D space while the associated cost drops in 3D.
for i in range(num_runs):
    px = all_paths_x[i]
    pc = all_paths_cost[i]
    ax.plot(px[:, 0], px[:, 1], pc, color='red', alpha=0.2, linewidth=0.5)
    # Mark the final point with a yellow triangle
    ax.scatter(px[-1, 0], px[-1, 1], pc[-1], color='yellow', marker='^', s=80, edgecolors='black', zorder=10)

# Identify the approximate global minimum (based on target y=[1,0])
# If A is near identity, this is x ~ [1,0]
min_x1, min_x2 = 0.95, 0.05 # Approximate min based on A = [[0.67, 0.33], [0.33, 0.67]]
ax.scatter(min_x1, min_x2, get_cost(np.array([min_x1, min_x2]), target_alpha), 
           color='green', marker='*', s=300, label='Approx. Global Min', zorder=11)

# Labels and View
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Cost $J(x)$')
ax.set_title(f'Homotopy Adam optimization runs (alpha schedule) on final surface )')
ax.view_init(elev=30, azim=-60)
plt.legend()

plt.show()