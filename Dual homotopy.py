import numpy as np
import matplotlib.pyplot as plt

# --- 1. Problem Configuration ---
c_val = 0.5
y_target = np.array([1.0, 0.0])
target_alpha = 35.0  # High stiffness
target_k = 0.98     # Extreme ill-conditioning
total_iterations = 600
learning_rate = 0.05

def h(z, alpha):
    return 1 / (1 + np.exp(-alpha * (z - c_val)))

def get_cost(x, A, alpha):
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y_target - y_pred)**2)

def run_adam_dual_comparison(x_init, use_dual_homotopy=False):
    x = x_init.copy()
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    costs = []
    
    # Define Schedule
    if use_dual_homotopy:
        # Gradually increase both alpha and k in sync
        alphas = [1.0, 5.0, 15.0, target_alpha]
        ks = [0.0, 0.5, 0.8, target_k]
        steps_per_phase = total_iterations // len(alphas)
    else:
        # Fixed at the hardest settings
        alphas = [target_alpha]
        ks = [target_k]
        steps_per_phase = total_iterations

    # Define the final target Matrix A for cost evaluation
    A_raw_final = np.array([[1.0, target_k], [target_k, 1.0]])
    A_final = A_raw_final / A_raw_final.sum(axis=1, keepdims=True)

    t_step = 1
    for i in range(len(alphas)):
        curr_alpha = alphas[i]
        curr_k = ks[i]
        
        # Build current Matrix A
        A_raw = np.array([[1.0, curr_k], [curr_k, 1.0]])
        A_curr = A_raw / A_raw.sum(axis=1, keepdims=True)
        
        for _ in range(steps_per_phase):
            y_p = h(np.dot(A_curr, x), curr_alpha)
            grad = 2 * np.dot(A_curr.T, ((y_p - y_target) * (curr_alpha * y_p * (1 - y_p))))
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t_step)
            v_hat = v / (1 - beta2**t_step)
            
            x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
            # Monitor progress against the FINAL hard objective
            costs.append(get_cost(x, A_final, target_alpha))
            t_step += 1
            
    return costs

# --- 2. Execute Comparison ---
# Use a difficult initialization where standard Adam is likely to stall
x_start = np.array([0.05, 0.05]) 

standard_adam_costs = run_adam_dual_comparison(x_start, use_dual_homotopy=False)
dual_homotopy_costs = run_adam_dual_comparison(x_start, use_dual_homotopy=True)

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(standard_adam_costs, label=f'Standard Adam (Fixed alpha={target_alpha}, k={target_k})', color='red', lw=2)
plt.plot(dual_homotopy_costs, label='Dual Homotopy (Scheduled alpha & k)', color='blue', lw=2)
plt.yscale('log')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost $J(x)$ (Evaluated at Final Objective)', fontsize=12)
plt.title('Performance Comparison: Dual Homotopy Strategy', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()