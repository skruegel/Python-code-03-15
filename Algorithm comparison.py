import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Setup ---
alpha_final = 30.0
c = 0.5
y_target = np.array([1.0, 0.0])
total_steps = 300
lr = 0.05

# Matrix A: Normalized [[1, 0.5], [0.5, 1]]
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z, a):
    return 1 / (1 + np.exp(-a * (z - c)))

def get_cost(x, a):
    y_p = h(np.dot(A, x), a)
    return np.sum((y_target - y_p)**2)

def get_grad_hess(x, a):
    z = np.dot(A, x)
    y_p = h(z, a)
    error = y_p - y_target
    hp = a * y_p * (1 - y_p)
    hpp = a**2 * y_p * (1 - y_p) * (1 - 2 * y_p)
    
    grad = 2 * np.dot(A.T, error * hp)
    diag_elements = hp**2 + error * hpp
    hess = 2 * np.dot(A.T * diag_elements, A)
    return grad, hess

# --- 2. Optimizer Implementations ---

def run_comparison(x_start):
    results = {}

    # A. Projected Gradient Descent (PGD)
    x, costs = x_start.copy(), []
    for _ in range(total_steps):
        g, _ = get_grad_hess(x, alpha_final)
        x = np.clip(x - lr * g, 0, 1)
        costs.append(get_cost(x, alpha_final))
    results['PGD'] = costs

    # B. PGD with Momentum
    x, costs, v_mom = x_start.copy(), [], np.zeros(2)
    beta = 0.9
    for _ in range(total_steps):
        g, _ = get_grad_hess(x, alpha_final)
        v_mom = beta * v_mom + (1 - beta) * g
        x = np.clip(x - lr * 5 * v_mom, 0, 1) # Scaled LR for momentum
        costs.append(get_cost(x, alpha_final))
    results['PGD + Momentum'] = costs

    # C. Pure Newton's Method
    x, costs = x_start.copy(), []
    for _ in range(total_steps):
        g, H = get_grad_hess(x, alpha_final)
        try:
            dx = np.linalg.solve(H, -g)
            x = np.clip(x + dx, 0, 1)
        except np.linalg.LinAlgError: pass
        costs.append(get_cost(x, alpha_final))
    results["Newton's"] = costs

    # D. Newton's with Levenberg-Marquardt
    x, costs, lam = x_start.copy(), [], 0.1
    for _ in range(total_steps):
        g, H = get_grad_hess(x, alpha_final)
        H_d = H + lam * np.eye(2)
        dx = np.linalg.solve(H_d, -g)
        new_x = np.clip(x + dx, 0, 1)
        if get_cost(new_x, alpha_final) < get_cost(x, alpha_final):
            x, lam = new_x, lam / 10
        else:
            lam *= 10
        costs.append(get_cost(x, alpha_final))
    results['Levenberg-Marquardt'] = costs

    # E. Adam
    x, costs, m, v = x_start.copy(), [], np.zeros(2), np.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, total_steps + 1):
        g, _ = get_grad_hess(x, alpha_final)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g**2)
        m_h, v_h = m / (1 - b1**t), v / (1 - b2**t)
        x = np.clip(x - lr * m_h / (np.sqrt(v_h) + eps), 0, 1)
        costs.append(get_cost(x, alpha_final))
    results['Adam'] = costs

    # F. Adam with Homotopy
    x, costs, m, v = x_start.copy(), [], np.zeros(2), np.zeros(2)
    schedule = [2.0, 10.0, alpha_final]
    steps_per = total_steps // len(schedule)
    t_global = 1
    for alpha_curr in schedule:
        for _ in range(steps_per):
            g, _ = get_grad_hess(x, alpha_curr)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g**2)
            m_h, v_h = m / (1 - b1**t_global), v / (1 - b2**t_global)
            x = np.clip(x - lr * m_h / (np.sqrt(v_h) + eps), 0, 1)
            costs.append(get_cost(x, alpha_final))
            t_global += 1
    results['Adam + Homotopy'] = costs

    return results

# --- 3. Run and Plot ---
x_init = np.array([0.1, 0.1]) # Starting in a "difficult" corner
final_results = run_comparison(x_init)

plt.figure(figsize=(12, 7))
for label, costs in final_results.items():
    plt.plot(costs, label=label, lw=2)

plt.yscale('log')
plt.title(f'Comparison of Optimizers (Alpha={alpha_final}, Start={x_init})', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Reconstruction Cost J(x)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()