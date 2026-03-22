import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 25.0
c = 0.5
y_target = np.array([1.0, 0.0])
iterations = 250
num_runs = 100
learning_rate = 0.05

# Adam Hyperparameters
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Matrix A: Normalized
A_raw = np.array([[1.0, 0.5], [0.5, 1.0]])
A = A_raw / A_raw.sum(axis=1, keepdims=True)

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_cost(x, A, y):
    y_pred = h(np.dot(A, x))
    return np.sum((y - y_pred)**2)

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

# --- 2. Adam Optimization ---
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
        
        x = np.clip(x - learning_rate * m_hat / (np.sqrt(v_hat) + eps), 0, 1)
        path.append(x.copy())
        costs.append(get_cost(x, A, y_target))
        
    all_paths.append(np.array(path))
    all_costs.append(np.array(costs))

print("Optimization complete with Xavier-style initialization.")