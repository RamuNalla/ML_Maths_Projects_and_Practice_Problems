#  Weekend Project Introduction & Setup (Remaining Time):
#  Project: Linear Regression from Scratch.

#  Goal: Apply gradient descent (which you implemented) to minimize the Mean Squared Error (MSE) cost function for a simple linear regression problem.



#  Today's Task:

# Understand MSE: Review the formula for Mean Squared Error for linear regression: J(θ₀, θ₁) = (1/m) * Σ( (θ₀ + θ₁*xᵢ) - yᵢ )².

# Calculate Gradient of MSE: Derive (or look up and understand) the partial derivatives of the MSE cost function with respect to the parameters θ₀ (intercept) and θ₁ (slope). This links Day 4 (Gradients) to the project.

# ∂J/∂θ₀ = (2/m) * Σ( (θ₀ + θ₁*xᵢ) - yᵢ )

# ∂J/∂θ₁ = (2/m) * Σ( (θ₀ + θ₁*xᵢ) - yᵢ ) * xᵢ

# Data Generation: Write the Python code (using NumPy) to generate synthetic data: create an array x, create y based on a known linear relationship (e.g., y = 3*x + 4) and add some random noise (np.random.randn).

# Setup Project Structure: Create a new Jupyter notebook or Python file for the project. Put your data generation code there.




# Weekend Goal (To be completed over Saturday/Sunday):

# Implement the MSE cost function in Python.

# Implement the gradient calculation function (using the derivatives derived above).

# Use your Batch Gradient Descent implementation from Day 5 (or the more general one) to find the optimal θ₀ and θ₁ that minimize the MSE for your synthetic data.

# Plot the original data points and the final regression line you found using the learned parameters.

# (Optional) Plot the cost function value at each iteration of gradient descent to visualize convergence.

# (Optional) Calculate the R-squared value manually to evaluate how well your line fits the data.

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def MSE_cost_function(x, y, theta_0, theta_1, m):             # Measn Squared error cost function
    
    # sum = 0

    # for i in range(m):
    #     sum += ((theta_0 + theta_1 * x[i]) - y[i])**2
    
    # return sum/m

    # OR

    sum_error = np.sum(((theta_0 + theta_1 *x)-y)**2)
    return sum_error/m


def grad_cost_function(x, y, theta_0, theta_1, m):         # Gradient Calculation function

    # grad_MSE = np.array([0, 0])

    # for i in range(m):
    #     grad_MSE[0] += 2*((theta_0 + theta_1*x[i]) - y[i])
    #     grad_MSE[1] += 2*x[i]*((theta_0 + theta_1*x[i])-y[i]) 
    
    # grad_MSE = grad_MSE/m
    # return grad_MSE

#OR
    grad_0 = np.sum(2 * ((theta_0 + theta_1 * x) - y)) / m
    grad_1 = np.sum(2 * x * ((theta_0 + theta_1 * x) - y)) / m
    return np.array([grad_0, grad_1])


np.random.seed(42)

x = np.linspace(0, 10, 100)                 # Array
y = 2*x + 4 + np.random.normal(0, 1, 100)  # True function: y = 2x + 4 + noise   # Array

theta_0, theta_1 = 0, 0

l_rate = 0.005
max_iters = 2000
tol = 1e-6
m = len(x)

MSE_array = []
iterations_array = []

for i in range(max_iters):

    grad = grad_cost_function(x, y, theta_0, theta_1, m)    # Need to calculate gradient for every theta_0 and theta_1. remember, this look is to iterate over thetas

    step_0 = l_rate * grad[0]
    step_1 = l_rate * grad[1]

    if np.abs(step_0) < tol and np.abs(step_1) < tol:
         print(f"Converged at iteration {i}")
         break
    
    theta_0 -= step_0
    theta_1 -= step_1

    MSE = MSE_cost_function(x, y, theta_0, theta_1, m)
    MSE_array.append(MSE)
    iterations_array.append(i)


print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
print("True values: θ₀ = 4, θ₁ = 2")

# Calculation of R2:
y_pred = theta_0 + theta_1*x                        # Array
SS_res = np.sum((y-y_pred)**2)                      # Residual sum of squares
SS_tot = np.sum((y-np.mean(y))**2)                  # Total sum of squares

R_squared = 1-((SS_res)/SS_tot)

print(f"R-squared: {R_squared:.4f}")  # Display R-squared value

fig, ax = plt.subplots(1,2, figsize=(12,5))

ax[0].scatter(x, y, label="Data")
ax[0].plot(x, theta_0 + theta_1*x, color='red', label="Batch GD Fit Line")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].legend()
ax[0].set_title("Batch Gradient Descent for Linear Regression")

ax[1].plot(iterations_array, MSE_array, color='blue', label = "Cost Function with iterations")
ax[1].set_xlabel("# Iterations")
ax[1].set_ylabel("Cost function")
ax[1].set_yscale('log')
ax[1].legend()
ax[1].set_title("Cost function vs iterations")

plt.show()



