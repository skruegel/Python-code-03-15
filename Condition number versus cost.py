import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha_val = 20.0
c_val = 0.5
y_target = np.array([1.0, 0.0])
learning_rate = 0.05
iterations = 250
num_trials = 5 # Average over trials to smooth results

def h(z, alpha):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-alpha * (z - c_val)))

def get_cost(x, A, y, alpha):
    """L2 Cost Function"""
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y - y_pred)**2)

def run_adam(A, alpha):
    """Adam optimizer for stable convergence"""
    x = np.random.rand(2)
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for t in range(1, iterations + 1):
        y_p = h(np.dot(A, x), alpha)
        grad = 2 * np.dot(A.T, ((y_p - y_target) * (alpha * y_p * (1 - y_p))))
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
    return get_cost(x, A, y_target, alpha)

# --- Simulation: Varying Coupling (k) ---
k_vals = np.linspace(0, 0.99, 50)
cond_nums = []
final_costs = []

for k in k_vals:
    # Construct normalized symmetric matrix A
    A_raw = np.array([[1.0, k], [k, 1.0]])
    A = A_raw / A_raw.sum(axis=1, keepdims=True)
    
    # Compute Condition Number
    cond = np.linalg.cond(A)
    cond_nums.append(cond)
    
    # Run trials and record average cost
    trial_costs = [run_adam(A, alpha_val) for _ in range(num_trials)]
    final_costs.append(np.mean(trial_costs))

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(cond_nums, final_costs, 'o-', color='darkred', markersize=4)
plt.xscale('log') # Log scale for the wide range of condition numbers
plt.title('Impact of Matrix Condition Number on Reconstruction Error')
plt.xlabel('Condition Number $\kappa(A)$ (Log Scale)')
plt.ylabel('Average Final Cost $J(x)$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()