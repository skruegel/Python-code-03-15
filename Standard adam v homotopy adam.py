import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
c = 0.5
y_target = np.array([1.0, 0.0])
target_alpha = 30.0  # High stiffness for the "hard" test
num_runs = 50
learning_rate = 0.05
total_steps = 300

# Matrix A (Normalized)
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z, alpha):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, alpha):
    y_pred = h(np.dot(A, x), alpha)
    return np.sum((y_target - y_pred)**2)

def get_gradient(x, alpha):
    y_pred = h(np.dot(A, x), alpha)
    error = y_pred - y_target
    dsigmoid = alpha * y_pred * (1 - y_pred)
    return 2 * np.dot(A.T, (error * dsigmoid))

def adam_optimizer(x_init, alpha_val, steps, m_start=None, v_start=None, t_start=1):
    x = x_init.copy()
    m = m_start if m_start is not None else np.zeros(2)
    v = v_start if v_start is not None else np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    path_costs = []
    for t_step in range(t_start, t_start + steps):
        grad = get_gradient(x, alpha_val)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t_step)
        v_hat = v / (1 - beta2**t_step)
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
        path_costs.append(get_cost(x, target_alpha)) # Always track against target hardness
    return x, path_costs, m, v

# --- Simulation ---
standard_results = []
homotopy_results = []

for _ in range(num_runs):
    x_start = np.random.rand(2)
    
    # 1. Standard Adam: Constant Alpha
    _, costs_std, _, _ = adam_optimizer(x_start, target_alpha, total_steps)
    standard_results.append(costs_std)
    
    # 2. Homotopy Adam: Alpha Scheduling (Curriculum Learning)
    # Start smooth (alpha=2), then sharpen (alpha=10), then target (alpha=30)
    x_curr = x_start.copy()
    costs_hom = []
    m_h, v_h = np.zeros(2), np.zeros(2)
    
    alphas = [2.0, 10.0, target_alpha]
    steps_per_alpha = total_steps // len(alphas)
    
    t_curr = 1
    for a in alphas:
        x_curr, c_h, m_h, v_h = adam_optimizer(x_curr, a, steps_per_alpha, m_h, v_h, t_curr)
        costs_hom.extend(c_h)
        t_curr += steps_per_alpha
        
    homotopy_results.append(costs_hom)

# --- Process Results for Plotting ---
std_mean = np.mean(standard_results, axis=0)
hom_mean = np.mean(homotopy_results, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(std_mean, label=f'Standard Adam (Fixed alpha={target_alpha})', linewidth=2, color='red')
plt.plot(hom_mean, label='Homotopy Adam (Alpha: 2 -> 10 -> 30)', linewidth=2, color='blue')
plt.yscale('log')
plt.xlabel('Total Iterations')
plt.ylabel('Cost J(x) (Evaluated at Alpha=30)')
plt.title('Homotopy vs. Standard Optimization')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()
#plt.savefig('homotopy_vs_standard.png')
print("Comparison saved as homotopy_vs_standard.png")