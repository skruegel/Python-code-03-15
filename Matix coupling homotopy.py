import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
alpha_val = 30.0
c_val = 0.5
y_target = np.array([1.0, 0.0])
total_iterations = 400
learning_rate = 0.05

def h(z):
    return 1 / (1 + np.exp(-alpha_val * (z - c_val)))

def get_cost(x, A):
    y_pred = h(np.dot(A, x))
    return np.sum((y_target - y_pred)**2)

def run_adam_with_matrix_schedule(x_init, A_final, use_homotopy=False):
    x = x_init.copy()
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    costs = []
    
    # Define coupling schedule (k is the off-axis weight)
    if use_homotopy:
        k_schedule = [0.0, 0.3, 0.6, 0.95] # From Identity to Heavy Blur
        steps_per_k = total_iterations // len(k_schedule)
    else:
        k_schedule = [0.95]
        steps_per_k = total_iterations

    t_step = 1
    for k in k_schedule:
        # Construct Matrix A for this specific homotopy phase
        A_raw = np.array([[1.0, k], [k, 1.0]])
        A_curr = A_raw / A_raw.sum(axis=1, keepdims=True)
        
        for _ in range(steps_per_k):
            y_p = h(np.dot(A_curr, x))
            grad = 2 * np.dot(A_curr.T, ((y_p - y_target) * (alpha_val * y_p * (1 - y_p))))
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t_step)
            v_hat = v / (1 - beta2**t_step)
            
            x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
            # CRITICAL: Always evaluate error against the FINAL target matrix
            costs.append(get_cost(x, A_final))
            t_step += 1
            
    return costs

# --- Setup Target Problem (k=0.95 is extremely ill-conditioned) ---
k_final = 0.95
A_raw_f = np.array([[1.0, k_final], [k_final, 1.0]])
A_final = A_raw_f / A_raw_f.sum(axis=1, keepdims=True)

# Run Comparison from a "difficult" starting point
x_start = np.array([0.1, 0.1]) 
std_costs = run_adam_with_matrix_schedule(x_start, A_final, use_homotopy=False)
hom_costs = run_adam_with_matrix_schedule(x_start, A_final, use_homotopy=True)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(std_costs, label='Standard Adam (Fixed k=0.95)', color='red', lw=2)
plt.plot(hom_costs, label='Matrix Homotopy (k: 0 -> 0.95)', color='blue', lw=2)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Cost J(x) (Evaluated at Final A)')
plt.title('Innovative Approach: Matrix Coupling Homotopy')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()