import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z, alpha=5.0, c=0.5):
    """
    Sigmoidal activation function h(z)
    alpha: stiffness parameter
    c: threshold/center point
    """
    return 1 / (1 + np.exp(-alpha * (z - c)))

def sigmoid_derivative(z, alpha=5.0, c=0.5):
    """
    Derivative of the sigmoidal function h'(z)
    Calculated using the identity: alpha * h(z) * (1 - h(z))
    """
    h = sigmoid(z, alpha, c)
    return alpha * h * (1 - h)

# Define range for z
z_vals = np.linspace(-2, 3, 500)

# Parameters for the sigmoid
alpha_val = 5.0
c_val = 0.5

# Compute values
h_z = sigmoid(z_vals, alpha_val, c_val)
h_prime_z = sigmoid_derivative(z_vals, alpha_val, c_val)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z_vals, h_z, label=rf'Sigmoid $h(z)$ ($\alpha={alpha_val}$)', color='blue', linewidth=2)
plt.plot(z_vals, h_prime_z, label=rf"Derivative $h'(z)$", color='red', linestyle='--', linewidth=2)

# Add styling and labels
plt.title('Sigmoidal Function and its Derivative', fontsize=14)
plt.xlabel('$z$ (Input to activation)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.axvline(c_val, color='gray', linestyle=':', label='Center $c$')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Save and show
plt.tight_layout()
plt.show()