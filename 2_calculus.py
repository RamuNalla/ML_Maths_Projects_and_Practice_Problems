
###################################################

### 1A: FINDING DERIVATIVES USING SYMPY

# import sympy as sp

# x = sp.symbols('x')

# f1 = x**3 - 2*x + 1
# f2 = sp.sin(x)*sp.cos(x)
# f3 = sp.exp(-x**2)

# df1 = sp.diff(f1,x)
# df2 = sp.diff(f2,x)
# df3 = sp.diff(f3,x)

# print("Derivative of x^3 - 2x + 1:", df1)
# print("Derivative of sin(x) * cos(x):", df2)
# print("Derivative of exp(-x^2):", df3)




### 1B: FINDING GRADIENT VECTOR USING SYMPY

# import sympy as sp

# x, y = sp.symbols('x y')
# f = x**2 + 2*x*y + y**3

# grad_f = [sp.diff(f, var) for var in (x,y)]   ## This contains the symbolic partial derivatives

# gradient_at_point = [df.subs({x: 1, y: 2}) for df in grad_f]  #subs() in sympy replaces variables in an expression, remember the syntax

# print("Gradient of f(x,y): ", grad_f)
# print("Gradient ar (1,2): ", gradient_at_point)




### 1C: IMPLEMENTING CHAIN RULE MANUALLY AND VERIFYING IT WITH SYMPY in a function (sinx)^2 

# import sympy as sp

# x = sp.symbols('x')

# g_x = sp.sin(x)                      
# f_u = sp.symbols('u')**2

# dg_dx = sp.diff(g_x, x)
# df_du = sp.diff(f_u, sp.symbols('u'))

# df_dx_manual = df_du.subs(sp.symbols('u'), g_x)*dg_dx

# df_dx_sympy = sp.diff(g_x**2, x)                   ## Calculating chain rule with sp.diff() directly

# print(f"Manual Chain Rule Result: {df_dx_manual}")
# print(f"SymPy Computed Derivative: {df_dx_sympy}")

# assert df_dx_manual == df_dx_sympy, "Manual calculation and SymPy result do not match!"





### 1D: VERIFYING DIFFERENTIATION USING FINITE DIFFERENCE METHOD and SYMPY diff() METHOD

# import numpy as np
# import sympy as sp

# def numerical_derivative(h,x, epsilon = 1e-5):
#     return (h(x + epsilon) - h(x-epsilon))/(2*epsilon)

# def h(x):
#     return x**2

# x_value = 2

# numerical_der_value = numerical_derivative(h, x_value)

# x_sym = sp.symbols('x')
# h_1 = x_sym**2

# sympy_derivative = sp.diff(h_1, x_sym)
# sympy_derivative = sympy_derivative.subs(x_sym, x_value)

# print(f"Numerical Derivative at x = {x_value}: {numerical_der_value}")

# print("Sympy computed Deriavtive:", sympy_derivative)
# print(f"Exact Derivative at x = {x_value}: {sympy_derivative}")




#### 1E: VISUALIZING DERIVATIVE 

# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# def numerical_derivative(h,x, epsilon = 1e-5):
#     return (h(x + epsilon) - h(x-epsilon))/(2*epsilon)

# def h(x):
#     return x**2

# x_value = 2

# slope = numerical_derivative(h, x_value)

# x = np.linspace(0,2,100)
# y = h(x)                 # Vector

# c = h(x_value) - slope*x_value    # Scalar

# tangent_line = slope * x + c  # Calculating tangent line vector

# plt.figure(figsize = (8,6))
# plt.plot(x,y,label=r'$h(x) = x^2$', color='blue')
# plt.plot(x, tangent_line, '--' , label=f'Tangent at x={x_value}', color='red')

# plt.scatter(x_value, h(x_value), color='black', zorder=3, label=f'Point ({x_value}, {h(x_value)})')

# plt.xlabel('x')
# plt.ylabel('h(x)')
# plt.title('Function and Tangent Line (Derivative Visualization)')
# plt.legend()
# plt.grid()
# plt.show()






################ GRADIENT DESCENT ALGORITHM ####################################

### 2A: Gradient Descent for f(x)

# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# def f(x):
#     return x**2 - 4*x + 5

# x = sp.symbols('x')
# f_x = x**2 - 4*x + 5

# grad_f_x = sp.diff(f_x, x)
# grad_f_num = sp.lambdify(x, grad_f_x, "numpy")     #converts symbolic expression into numerical function  and allows to use numpy arrays for vectorized operations

# def gradient_descent(grad_f, init_x = 0.0, lr = 0.001, max_iters = 1000, tol = 0.0001):
#     x_values = [init_x]               # To store all x values for plotting, the first element being init_x
#     prev_x = init_x

#     for i in range(max_iters):
#         grad_value = grad_f(prev_x)
#         next_x = prev_x - lr * grad_value

#         x_values.append(next_x)

#         if abs(lr*grad_value) < tol:
#             print(f"Converged at iteration {i}, x = {next_x:.6f}")
#             break
        
#         prev_x = next_x

#         if i%50 == 0:
#             print(f"Iteration {i}: x = {prev_x:.6f}, f(x) = {f(prev_x):.6f}")
    
#     print(f"Optimum solution found at x = {prev_x:.6f}")

#     return prev_x, x_values


# optimal_x, x_vals = gradient_descent(grad_f_num, init_x=0.0, lr = 0.01, max_iters=1000)

# x_range = np.linspace(-1,4,100)
# y_range = f(x_range)

# plt.figure(figsize=(8,5))
# plt.plot(x_range, y_range, label="f(x) = x² - 4x + 5", color='blue')
# plt.scatter(x_vals, [f(x) for x in x_vals], color='red', s=10, label="Optimization Path")

# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("Gradient Descent Optimization")
# plt.legend()
# plt.grid()
# plt.show()




#### 2B: Gradient Descent for f(x, y)

# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# def f(x, y):
#     return x**2 + 2*y**2

# def gradient(x, y):
#     df_dx = 2*x
#     df_dy = 4*y
#     return np.array([df_dx, df_dy])

# learning_rate = 0.1
# num_iterations = 500
# initial_point = np.array([2.0, 3.0])

# point = initial_point

# gradient_path = [initial_point]

# for i in range(num_iterations):
#     grad = gradient(point[0], point[1])
#     point = point - learning_rate*grad
#     gradient_path.append(point)

# gradient_path = np.array(gradient_path)

# x_vals = np.linspace(-3,3,100)
# y_vals = np.linspace(-3,3,100)

# X, Y = np.meshgrid(x_vals, y_vals)

# Z = f(X, Y)

# plt.figure(figsize=(8, 6))
# plt.contour(X, Y, Z, levels=20, cmap='viridis')  # Contour plot of f(x, y)
# plt.plot(gradient_path[:, 0], gradient_path[:, 1], 'ro-', markersize=4, label='Gradient Descent Path')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Gradient Descent Path on f(x, y) = x^2 + 2*y^2")
# plt.legend()
# plt.show()




#### 2C: Gradient Descent for a generalized function


# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# def f_grad(x1, x2, x3):
#     return np.array([2*x1, 4*x2, 6*x3])

# def gradient_descent(f_grad, start_point, lr = 0.001, max_iters = 1000, tol = 0.0001):
   
#     point = np.array(start_point, dtype=float)
#     trajectory = [point]

#     for i in range(max_iters):
#         grad = np.array(f_grad(*point))
#         step = lr * grad

#         if np.linalg.norm(step) < tol:
#             print(f"Converged at iteration {i}")
#             break

#         point -= step
#         trajectory.append(point)
    
#     return trajectory, point

# start_point = np.array([3,2,1])

# trajectory, final_point = gradient_descent(f_grad, start_point, lr=0.1, max_iters = 1000, tol = 0.001)

# print("Final optimized point: ", final_point)





#### 2D: Stochastic Gradient Descent for a generalized function

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# x = np.linspace(0, 10, 100)
# y = 2*x + 1 + np.random.normal(0, 1, 100)  # True function: y = 2x + 1 + noise

# theta_0, theta_1 = 0, 0

# l_rate = 0.005
# max_iters = 2000

# for i in range(max_iters):
#     idx = np.random.randint(0, len(x))

#     x_i, y_i = x[idx], y[idx]

#     grad_0 = 2*((theta_0 + theta_1*x_i) - y_i)      ## Ideally here, we do another loop to find the sum across all datapoints, but, we are 
#     grad_1 = 2*x_i*((theta_0 + theta_1*x_i)-y_i)    ## finding the gradient at only one random datapoint, Note that the loss function is also summation across all rows

#     theta_0 -= l_rate * grad_0
#     theta_1 -= l_rate * grad_1

# print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
# print("True values: θ₀ = 1, θ₁ = 2")

# plt.scatter(x, y, label="Data")
# plt.plot(x, theta_0 + theta_1*x, color='red', label="SGD Fit Line")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Stochastic Gradient Descent for Linear Regression")
# plt.show()

   
   


#### 2E: MOMENTUM GRADIENT DESCENT ALONG WITH STOCHASTIC GD

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# x = np.linspace(0, 10, 100)
# y = 2*x + 1 + np.random.normal(0, 1, 100)  # True function: y = 2x + 1 + noise

# theta_0, theta_1 = 0, 0

# l_rate = 0.005
# max_iters = 2000
# tol = 1e-4

# beta = 0.99                                             ## Momentum coefficient
# v_prev_theta_0 = 0                   # Initial velocity for theta_0
# v_prev_theta_1 = 0                   # Initial velocity for theta_1

# for i in range(max_iters):
#     idx = np.random.randint(0, len(x))

#     x_i, y_i = x[idx], y[idx]

#     grad_0 = 2*((theta_0 + theta_1*x_i) - y_i)      ## Ideally here, we do another loop to find the sum across all datapoints, but, we are 
#     grad_1 = 2*x_i*((theta_0 + theta_1*x_i)-y_i)    ## finding the gradient at only one random datapoint, Note that the loss function is also summation across all rows

#     v_next_theta_0 = beta*v_prev_theta_0 + (1 - beta)*grad_0
#     v_next_theta_1 = beta*v_prev_theta_1 + (1- beta) * grad_1

#     step_0 = l_rate * v_next_theta_0
#     step_1 = l_rate * v_next_theta_1

#     if np.abs(step_0) < tol and np.abs(step_1) < tol:
#         print(f"Converged at iteration {i}")
#         break
    
#     theta_0 -= step_0
#     theta_1 -= step_1    

#     v_prev_theta_0 = v_next_theta_0
#     v_prev_theta_1 = v_next_theta_1

# print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
# print("True values: θ₀ = 1, θ₁ = 2")

# plt.scatter(x, y, label="Data")
# plt.plot(x, theta_0 + theta_1*x, color='red', label="SGD Fit Line")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Stochastic Gradient Descent for Linear Regression")
# plt.show()





#### 2F: MOMENTUM GRADIENT DESCENT ALONG WITH BATCH GD

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# x = np.linspace(0, 10, 100)
# y = 2*x + 1 + np.random.normal(0, 1, 100)  # True function: y = 2x + 1 + noise

# theta_0, theta_1 = 0, 0

# l_rate = 0.005
# max_iters = 2000
# tol = 1e-4

# beta = 0.99                                              ## Momentum coefficient
# v_prev_theta_0 = 0                   # Initial velocity for theta_0
# v_prev_theta_1 = 0                   # Initial velocity for theta_1

# for i in range(max_iters):
#     grad_0_sum = 0
#     grad_1_sum = 0

#     for j in range(len(x)):                                # Looping it over entire dataset to find the gradient instead of a single random datapoint
#         grad_0_sum += 2 * ((theta_0 + theta_1 * x[j]) - y[j])
#         grad_1_sum += 2 * x[j] * ((theta_0 + theta_1 * x[j]) - y[j])

#     grad_0 = grad_0_sum/len(x)
#     grad_1 = grad_1_sum/len(x)

#     v_next_theta_0 = beta*v_prev_theta_0 + (1 - beta)*grad_0
#     v_next_theta_1 = beta*v_prev_theta_1 + (1- beta) * grad_1

#     step_0 = l_rate * v_next_theta_0
#     step_1 = l_rate * v_next_theta_1

#     if np.abs(step_0) < tol and np.abs(step_1) < tol:
#         print(f"Converged at iteration {i}")
#         break
    
#     theta_0 -= step_0
#     theta_1 -= step_1    

#     v_prev_theta_0 = v_next_theta_0
#     v_prev_theta_1 = v_next_theta_1

# print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
# print("True values: θ₀ = 1, θ₁ = 2")

# plt.scatter(x, y, label="Data")
# plt.plot(x, theta_0 + theta_1*x, color='red', label="SGD Fit Line")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Stochastic Gradient Descent for Linear Regression")
# plt.show()



#### 2G: MOMENTUM GRADIENT DESCENT ALONG WITH STOCHASTIC GD and Decay learning rate

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# x = np.linspace(0, 10, 100)
# y = 2*x + 1 + np.random.normal(0, 1, 100)  # True function: y = 2x + 1 + noise

# theta_0, theta_1 = 0, 0

# l_rate = 0.005
# decay_rate = 0.001

# max_iters = 2000
# tol = 1e-4

# beta = 0.99                                             ## Momentum coefficient
# v_prev_theta_0 = 0                   # Initial velocity for theta_0
# v_prev_theta_1 = 0                   # Initial velocity for theta_1

# for i in range(max_iters):
#     idx = np.random.randint(0, len(x))

#     x_i, y_i = x[idx], y[idx]

#     grad_0 = 2*((theta_0 + theta_1*x_i) - y_i)      ## Ideally here, we do another loop to find the sum across all datapoints, but, we are 
#     grad_1 = 2*x_i*((theta_0 + theta_1*x_i)-y_i)    ## finding the gradient at only one random datapoint, Note that the loss function is also summation across all rows

#     v_next_theta_0 = beta*v_prev_theta_0 + (1 - beta)*grad_0
#     v_next_theta_1 = beta*v_prev_theta_1 + (1- beta) * grad_1

#     lr_t = l_rate / (1 + decay_rate*i)

#     step_0 = lr_t * v_next_theta_0
#     step_1 = lr_t* v_next_theta_1

#     if np.abs(step_0) < tol and np.abs(step_1) < tol:
#         print(f"Converged at iteration {i}")
#         break
    
#     theta_0 -= step_0
#     theta_1 -= step_1    

#     v_prev_theta_0 = v_next_theta_0
#     v_prev_theta_1 = v_next_theta_1

# print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
# print("True values: θ₀ = 1, θ₁ = 2")

# plt.scatter(x, y, label="Data")
# plt.plot(x, theta_0 + theta_1*x, color='red', label="SGD Fit Line")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Stochastic Gradient Descent for Linear Regression")
# plt.show()





#### 2H: ADAM OPTIMIZER


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2*x + 1 + np.random.normal(0, 1, 100)  # True function: y = 2x + 1 + noise

theta_0, theta_1 = 0, 0
m_t0, v_t0 = 0, 0                  # Coefficients for Adam optimizer
m_t1, v_t1 = 0, 0

l_rate = 0.01
beta1, beta2 = 0.9, 0.999
max_iters = 2000

epsilon = 1e-8
tol = 1e-6


for i in range(1, max_iters+1):     ## Note, we start from 1 as the multiplication with i is involved
    idx = np.random.randint(0, len(x))

    x_i, y_i = x[idx], y[idx]

    grad_0 = 2*((theta_0 + theta_1*x_i) - y_i)      ## Ideally here, we do another loop to find the sum across all datapoints, but, we are 
    grad_1 = 2*x_i*((theta_0 + theta_1*x_i)-y_i)    ## finding the gradient at only one random datapoint, Note that the loss function is also summation across all rows

    m_t0 = beta1 * m_t0 + (1 - beta1) * grad_0      ## Updating Moment1 and Moment2
    v_t0 = beta2 * v_t0 + (1 - beta2) * grad_0**2

    m_t1 = beta1 * m_t1 + (1 - beta1) * grad_1
    v_t1 = beta2 * v_t1 + (1 - beta2) * grad_1**2

    m_t0_hat = m_t0 / (1 - beta1**i)
    v_t0_hat = v_t0 / (1 - beta2**i)

    m_t1_hat = m_t1 / (1 - beta1**i)
    v_t1_hat = v_t1 / (1 - beta2**i)

    step_0 = (l_rate * m_t0_hat) / (np.sqrt(v_t0_hat) + epsilon)
    step_1 = (l_rate * m_t1_hat) / (np.sqrt(v_t1_hat) + epsilon)

    if np.abs(step_0) < tol and np.abs(step_1) < tol:
        print(f"Converged at iteration {i}")
        break
    
    theta_0 -= step_0
    theta_1 -= step_1    


print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
print("True values: θ₀ = 1, θ₁ = 2")

plt.scatter(x, y, label="Data")
plt.plot(x, theta_0 + theta_1*x, color='red', label="SGD Fit Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Stochastic Gradient Descent for Linear Regression")
plt.show()













################################# 3. INTEGRAL CALCULUS ###############################################


### 3A: SYMBOLIC INTEGRATION of 3 functions (indefinite integral)

# import numpy as np
# import sympy as sp

# x = sp.symbols('x')

# f1 = 2*x + 1
# f2 = x**3
# f3 = sp.cos(x)

# int_f1 = sp.integrate(f1,x)
# diff_f1 = sp.diff(f1, x)
# int_f2 = sp.integrate(f2,x)
# int_f3 = sp.integrate(f3,x)

# print("∫ (2x + 1) dx =", int_f1, "+ C")
# print("∫ (x³) dx =", int_f2, "+ C")
# print("∫ (cos(x)) dx =", int_f3, "+ C")



### 3B: SYMBOLIC INTEGRATION of 3 functions (definite integral)

# import numpy as np
# import sympy as sp

# x = sp.symbols('x')

# f1 = x**2
# f2 = sp.sin(x)

# def_int_f1 = sp.integrate(f1,(x, 0, 1))
# def_int_f2 = sp.integrate(f2,(x, 0, np.pi))

# print("∫ (x**2) dx with limits 0, 1 =", def_int_f1)
# print("∫ (x³) dx with limits 0, pi =", def_int_f2 )


### 3C: NUMERICAL INTEGRATION of 3 functions (definite integral) using Scipy

# import numpy as np
# import scipy.integrate as spi

# def f(x):
#     return np.exp(-x**2)

# result, error = spi.quad(f, -np.inf, np.inf)             # scipy.integrate.quad numerically computes definite integrals using adaptive quadrature techniques

# print("Numerical Integral of e^(-x^2) from -∞ to ∞:", result)

# print("Estimated Numerical Error:", error)
# quad should be used when a function has no simple antiderivative or when integrating over infinite limits




### 3D: Integrating PDF using quad netween -1 and 1

# import numpy as np
# import scipy.integrate as spi
# import scipy.stats as stats

# def normal_pdf(x):
#     return stats.norm.pdf(x, loc=0, scale=1)  # Mean - 0 and Std Dev - 1

# result, error = spi.quad(normal_pdf, -1, 1)  

# print("Area under the standard normal PDF from -1 to 1:", result)
# print("Estimated Numerical Error:", error)




### 3E: RIEMANN SUM IMPLEMENTATION  (function x^2 from 1 to 2)

# import numpy as np
# import scipy.integrate as spi
# import sympy as sp

# def f(x):
#     return x**2

# x = sp.symbols('x')
# f_x = x**2

# n = 1000   # number of partitions in the graph
# x_vals = np.linspace(1,2,n+1)

# area_riemann_sum = 0

# for i in range(n):
#     mid_x = (x_vals[i+1] + x_vals[i])/2
#     seg_area = (x_vals[i+1]-x_vals[i])*f(mid_x)
#     area_riemann_sum += seg_area


# area_scipy = sp.integrate(f_x, (x, 1, 2)).evalf()               ## adding evalf() at the end gives 2.3333, but without it gives 7/3

# print("Area under the x**2 from 1 to 2 - RIEMANN SUM CALCULATION:", area_riemann_sum)
# print("Area under the x**2 from 1 to 2 - sp.integrate:", area_scipy)







