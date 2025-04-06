
## 6A: GENERATE CORRELATED DATA

# Task: Create two NumPy arrays, x and y, that exhibit different levels of linear correlation.

# Positive Correlation: x = np.random.rand(100) * 10, y = 2 * x + np.random.normal(0, 5, 100) (Clear positive trend with noise).
# Negative Correlation: x_neg = np.random.rand(100) * 10, y_neg = -1.5 * x_neg + np.random.normal(0, 5, 100)
# Weak/No Correlation: x_no = np.random.rand(100) * 10, y_no = np.random.rand(100) * 10

# Visualize: Create scatter plots (plt.scatter()) for each pair (x, y), (x_neg, y_neg), (x_no, y_no) to visually inspect the relationship.

# import numpy as np
# import matplotlib.pyplot as plt

# #Positive correlation

# x = np.random.rand(100)*10
# y = 2*x + np.random.normal(0,5,100)

# #Negative Correlation

# x_neg = np.random.rand(100)*10
# y_neg = -1.5 * x_neg + np.random.normal(0, 5, 100)

# #Weak Correlation

# x_no = np.random.rand(100)*10
# y_no = np.random.rand(100) * 10

# fig, axes = plt.subplots(1,3, figsize = (12,5))

# # Positive Correlation Plot
# axes[0].scatter(x, y, color='red', alpha=0.7)
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("y")
# axes[0].set_title("Positive Correlation")

# # Negative Correlation Plot
# axes[1].scatter(x_neg, y_neg, color='blue', alpha=0.7)
# axes[1].set_xlabel("x_neg")
# axes[1].set_ylabel("y_neg")
# axes[1].set_title("Negative Correlation")

# # Weak Correlation Plot
# axes[2].scatter(x_no, y_no, color='green', alpha=0.7)
# axes[2].set_xlabel("x_no")
# axes[2].set_ylabel("y_no")
# axes[2].set_title("Weak/No Correlation")

# plt.tight_layout() # Improves spacing between plots
# plt.show()





### 6B: CALCULATE COVARIANCE AND CORRELATION USING NUMPY

# Library: np.cov, np.corrcoef

# Task: For each pair of generated data ((x, y), (x_neg, y_neg), (x_no, y_no)):

# Calculate the covariance matrix using np.cov(x_array, y_array). Note this returns a 2x2 matrix; the off-diagonal elements represent the covariance between x and y.
# Calculate the correlation matrix using np.corrcoef(x_array, y_array). The off-diagonal element is the Pearson correlation coefficient (r).
# Interpret: Print the correlation coefficient for each pair. Does the value match your visual inspection (close to 1 for positive, close to -1 for negative, close to 0 for none)? What does the sign of the covariance tell you?

# import numpy as np
# import matplotlib.pyplot as plt

# #Positive correlation
# x = np.random.rand(100)*10
# y = 2*x + np.random.normal(0,5,100)

# cov_pos = np.cov(x, y)
# corr_coef_pos = np.corrcoef(x, y)

# print("Covariance matrix of positive correlation data:\n", cov_pos)
# print("Correlation matrix of positive correlation data:\n", corr_coef_pos)

# #Negative Correlation
# x_neg = np.random.rand(100)*10
# y_neg = -1.5 * x_neg + np.random.normal(0, 5, 100)

# cov_neg = np.cov(x_neg, y_neg)
# corr_coef_neg = np.corrcoef(x_neg, y_neg)

# print("Covariance matrix of negative correlation data:\n", cov_neg)
# print("Correlation matrix of negative correlation data:\n", corr_coef_neg)

# #Weak Correlation
# x_no = np.random.rand(100)*10
# y_no = np.random.rand(100) * 10

# cov_no = np.cov(x_no, y_no)
# corr_coef_no = np.corrcoef(x_no, y_no)

# print("Covariance matrix of Weak correlation data:\n", cov_no)
# print("Correlation matrix of Weak correlation data:\n", corr_coef_no)






### 6C: CALCULATE COVARIANCE MANUALLY

# Task: Implement the formula for covariance manually: Cov(X, Y) = Σ[(xᵢ - μₓ)(yᵢ - μ<0xC3><0xAA>)] / (n - 1) (sample covariance).

# Calculate the means μₓ and μ<0xC3><0xAA>.
# Calculate the terms (xᵢ - μₓ) and (yᵢ - μ<0xC3><0xAA>).
# Multiply them element-wise.
# Sum the results.
# Divide by n-1.
# Compare: Calculate manually for your positively correlated (x, y) data and compare the result to the off-diagonal value obtained from np.cov.


# import numpy as np
# import matplotlib.pyplot as plt

# #Positive correlation
# x = np.random.rand(100)*10
# y = 2*x + np.random.normal(0,5,100)

# cov_pos = np.cov(x, y)

# print("Covariance matrix of positive correlation data:\n", cov_pos)

# mean_x = np.mean(x)
# mean_y = np.mean(y)

# n = len(x)

# cov_manual = np.sum((x-mean_x)*(y-mean_y)/(n-1))

# print("Covariance with manual calculation:\n", cov_manual)






### 6D: LINEAR REGRESSION WITH SCIKIT-LEARN

# Library: sklearn.linear_model.LinearRegression, sklearn.model_selection.train_test_split (optional but good practice), sklearn.metrics.r2_score

# Task: Use your positively correlated data (x, y).
# Reshape: Scikit-learn expects features (X) as a 2D array. Reshape x: X = x.reshape(-1, 1). y can remain 1D.
# (Optional Split): X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create Model: model = LinearRegression()
# Train Model: model.fit(X_train, y_train) (or model.fit(X, y) if not splitting).
# Get Coefficients: Print the learned intercept (model.intercept_) and slope (model.coef_). Compare the slope to the 2 you used when generating the data.
# Make Predictions: y_pred = model.predict(X_test) (or model.predict(X)).
# Visualize Fit: Create a scatter plot of the original data (X_test, y_test or X, y). Plot the regression line using the predictions (plt.plot(X_test, y_pred, color='red')).


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score


# def R2_manual(y_test, y_pred):
#     y_mean = np.mean(y_test)

#     SSR = np.sum((y_test - y_pred)**2)
#     SST = np.sum((y_test - y_mean)**2)

#     return 1-(SSR/SST) 


# np.random.seed(42)  # Ensure reproducibility

# #Positive correlation
# x = np.random.rand(1000)*10              # This is a 1D vector (1x100)
# y = 2*x + np.random.normal(0,1,1000)

# X = x.reshape(-1,1)                    # Scikit-learn requires data in 2D array, 100x1 - column vector. This operation is converting into column vector

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)   #80% training data and 20% test data. random_state helps generate the same data making the model reproducible

# model = LinearRegression()      # creating the model
# model.fit(X_train, y_train)     # training the model

# print(f"Intercept: {model.intercept_:.4f}")  # b (y-intercept)
# print(f"Slope (Coefficient): {model.coef_[0]:.4f}")  # m (slope)

# y_pred = model.predict(X_test)

# r2 = r2_score(y_test, y_pred)                  # R2 relation is only with y data
# print(f"R² Score on test data: {r2:.4f}")
# print(f"R² manual calculation on test data: {R2_manual(y_test, y_pred):.4f}")
# print(f"R² Score on Trained data: {r2_score(y_train, model.predict(X_train)):.4f}")
# print(f"R² Score on Complete data: {r2_score(y, model.predict(X)):.4f}")


# plt.scatter(X_test, y_test, color='blue', label='Actual Data')  # Scatter plot of actual data
# plt.plot(X_test, y_pred, color='red', label='Regression Line')  # Regression Line
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Linear Regression Fit")
# plt.legend()
# plt.show()
