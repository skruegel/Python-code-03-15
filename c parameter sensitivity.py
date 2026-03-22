import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 25.0
y_target = np.array([1.0, 0.0])
num_trials_per_c = 10
iterations = 250
learning_rate = 0.05

# Adam Parameters
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Matrix A (Normalized [[1, 0.5], [0.5, 1]])
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z, alpha, c_val):
    """Sigmoid activation function with variable threshold c"""
    return 1 / (1 + np.exp(-alpha * (z - c_val)))

def get_cost(x, A, y, alpha, c_val):
    y_pred = h(np.dot(A, x), alpha, c_val)
    return np.sum((y - y_pred)**2)

def run_adam(c_val, x_start):
    x = x_start.copy()
    m, v = np.zeros(2), np.zeros(2)
    for t in range(1, iterations + 1):
        # Calculate current prediction and gradient
        Ax = np.dot(A, x)
        y_p = h(Ax, alpha, c_val)
        grad = 2 * np.dot(A.T, ((y_p - y_target) * (alpha * y_p * (1 - y_p))))
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
    return get_cost(x, A, y_target, alpha, c_val)

# --- Simulation: Varying Threshold c ---
c_values = np.linspace(0.0, 1.0, 50)
final_errors_c = []

for c_val in c_values:
    trial_errors = [run_adam(c_val, np.random.rand(2)) for _ in range(num_trials_per_c)]
    final_errors_c.append(np.mean(trial_errors))

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(c_values, final_errors_c, 'o-', color='darkorange', linewidth=2, markersize=4)
plt.title('Parametric Sensitivity: Final Error vs. Threshold $c$')
plt.xlabel('Threshold $c$')
plt.ylabel('Average Final Cost $J(x)$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('parametric_sensitivity_c.png')
#print("Plot saved as parametric_sensitivity_c.png")
plt.show()