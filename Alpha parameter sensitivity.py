import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
c = 0.5
y_target = np.array([1.0, 0.0])
learning_rate = 0.05
iterations = 200
num_trials = 10  # Number of random starts per alpha to average results
alphas = np.arange(1, 51)

# Define and Normalize Matrix A
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z, alpha):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, alpha):
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y_target - y_pred)**2)

def get_gradient(x, alpha):
    y_p = h(np.dot(A, x), alpha)
    error = y_p - y_target
    dsigmoid = alpha * y_p * (1 - y_p)
    return 2 * np.dot(A.T, (error * dsigmoid))

def run_adam(alpha):
    trial_errors = []
    for _ in range(num_trials):
        x = np.random.rand(2)
        m, v = np.zeros(2), np.zeros(2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for t in range(1, iterations + 1):
            grad = get_gradient(x, alpha)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
        
        trial_errors.append(get_cost(x, alpha))
    return np.mean(trial_errors)

# --- Execution ---
final_errors = []
for alpha_val in alphas:
    avg_error = run_adam(alpha_val)
    final_errors.append(avg_error)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(alphas, final_errors, marker='o', linestyle='-', color='teal', markersize=4)
plt.title(r'Parametric Sensitivity: Impact of Stiffness ($\alpha$) on Final Reconstruction Error', fontsize=14)
plt.xlabel(r'Stiffness Parameter ($\alpha$)', fontsize=12)
plt.ylabel('Average Final Cost $J(x^*)$', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()
plt.savefig('parametric_sensitivity_alpha.png')
print("Plot saved as parametric_sensitivity_alpha.png")