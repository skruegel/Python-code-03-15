import numpy as np
import matplotlib.pyplot as plt

# --- Project Parameters ---
alpha = 15.0
c = 0.5
y_target = np.array([1.0, 0.0])
learning_rate = 0.1
iterations = 500

# Gaussian-like Kernel A (2x2)
sigma = 1.0
w0 = np.exp(-0**2/(2*sigma**2))
w1 = np.exp(-1**2/(2*sigma**2))
row_sum = w0 + w1
A = np.array([[w0/row_sum, w1/row_sum],
              [w1/row_sum, w0/row_sum]])

def h(z):
    return 1 / (1 + np.exp(-alpha * (z - c)))

def get_gradient(x, A, y, alpha, c):
    # Forward pass
    Ax = np.dot(A, x)
    y_pred = h(Ax)
    
    # Error: (y_pred - y)
    error = y_pred - y
    
    # Gradient of sigmoid: alpha * h(Ax) * (1 - h(Ax))
    dsigmoid = alpha * y_pred * (1 - y_pred)
    
    # Chain rule: 2 * A.T @ (error * dsigmoid)
    # The 2 comes from the square in the L2 norm
    grad = 2 * np.dot(A.T, (error * dsigmoid))
    return grad

# --- Gradient Descent Loop ---

# 1) Start at a random (x1, x2) between 0 and 1
x = np.random.rand(2)
history = [x.copy()]

for i in range(iterations):
    # 2) Calculate the gradient
    grad = get_gradient(x, A, y_target, alpha, c)
    
    # 3) Step downhill
    new_x = x - learning_rate * grad
    
    # Projection: Keep x between 0 and 1
    new_x = np.clip(new_x, 0, 1)
    
    # Check for convergence (if x stops changing much)
    if np.linalg.norm(new_x - x) < 1e-6:
        break
        
    x = new_x
    history.append(x.copy())

history = np.array(history)

# --- Visualization ---
# Plotting the path on a contour map of J(x)
N_grid = 100
x1_vals = np.linspace(0, 1, N_grid)
x2_vals = np.linspace(0, 1, N_grid)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(N_grid):
    for j in range(N_grid):
        xi = np.array([X1[i,j], X2[i,j]])
        pred = h(np.dot(A, xi))
        Z[i,j] = np.sum((y_target - pred)**2)

plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(label='Cost J(x)')
plt.plot(history[:, 0], history[:, 1], 'r.-', label='Optimization Path')
plt.scatter(history[0, 0], history[0, 1], color='white', label='Start')
plt.scatter(x[0], x[1], color='red', marker='*', s=200, label='Final x*')
plt.title('Gradient Descent Path on Cost Landscape')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

print(f"Final x: {x}")
print(f"Final Cost: {np.sum((y_target - h(np.dot(A, x)))**2)}")