import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])
num_trials_per_k = 10
iterations = 250
learning_rate = 0.05

# Adam Parameters
beta1, beta2, eps = 0.9, 0.999, 1e-8

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, A, y):
    y_pred = h(np.dot(A, x))
    return np.sum((y - y_pred)**2)

def run_adam(A, x_start):
    x = x_start.copy()
    m, v = np.zeros(2), np.zeros(2)
    for t in range(1, iterations + 1):
        y_p = h(np.dot(A, x))
        # Gradient
        grad = 2 * np.dot(A.T, ((y_p - y_target) * (alpha * y_p * (1 - y_p))))
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
    return get_cost(x, A, y_target)

# --- Varying Coupling (k) in Matrix A ---
# A_raw = [[1, k], [k, 1]]
k_values = np.linspace(0, 0.99, 50)
final_errors_A = []

for k in k_values:
    A_raw = np.array([[1.0, k], [k, 1.0]])
    A = A_raw / A_raw.sum(axis=1, keepdims=True)
    
    trial_errors = [run_adam(A, np.random.rand(2)) for _ in range(num_trials_per_k)]
    final_errors_A.append(np.mean(trial_errors))

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(k_values, final_errors_A, 'o-', color='crimson', linewidth=2, markersize=4)
plt.title('Parametric Sensitivity: Final Error vs. Matrix Coupling ($k$)', fontsize=14)
plt.xlabel('Coupling Coefficient $k$ (Blurring Intensity)', fontsize=12)
plt.ylabel('Average Final Cost $J(x)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('parametric_sensitivity_matrix_A.png')
print("Plot saved as parametric_sensitivity_matrix_A.png")
plt.show()