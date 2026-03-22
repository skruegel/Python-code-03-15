import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Project Parameters ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])
num_runs = 100
iterations = 250

# Adam Hyperparameters
learning_rate = 0.05
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# --- Define Matrix A ---
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
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

# --- 1. Xavier/He Style Input Initialization ---
def xavier_init_input(A, c_val):
    """
    Initializes x such that Ax starts near the threshold c.
    This prevents the sigmoid from being 'dead' at the start.
    """
    n_in = A.shape[1]
    # Small variance centered around the threshold c
    # We use a normal distribution scaled by the fan-in (number of inputs)
    std = np.sqrt(2.0 / n_in) 
    x = np.random.normal(loc=c_val, scale=std, size=n_in)
    return np.clip(x, 0, 1)

# --- Run Optimization with Adam ---
all_paths = []
all_costs = []

for _ in range(num_runs):
    # USE XAVIER INITIALIZATION INSTEAD OF UNIFORM RAND
    x = xavier_init_input(A, c)
    
    m, v = np.zeros(2), np.zeros(2)
    path = [x.copy()]
    costs = [get_cost(x, A, y_target)]
    
    for t in range(1, iterations + 1):
        y_p = h(np.dot(A, x))
        grad = 2 * np.dot(A.T, ((y_p - y_target) * (alpha * y_p * (1 - y_p))))
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon), 0, 1)
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
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Cost $J(x)$')

for i in range(num_runs):
    path = all_paths[i]
    costs = all_costs[i]
    ax.plot(path[:, 0], path[:, 1], costs, color='red', alpha=0.15, linewidth=0.5)
    ax.scatter(path[-1, 0], path[-1, 1], costs[-1], 
               color='yellow', marker='^', s=80, edgecolors='black', linewidths=1, zorder=10)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Cost $J(x)$')
ax.set_title(r'3D Landscape: Adam Optimizer ($\alpha=25$)')
ax.view_init(elev=30, azim=-60)

plt.show()