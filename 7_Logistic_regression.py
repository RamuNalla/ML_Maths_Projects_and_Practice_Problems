
## PROBLEM 7A #############

# Sigmoid Function:

# Task: Implement the sigmoid function sigmoid(z) in Python using NumPy.

# Visualize: Generate a range of input values z (e.g., np.linspace(-10, 10, 100)). 
# Calculate the sigmoid output for each z. Plot z vs sigmoid(z) using matplotlib. Observe the S-shape and that the output is always between 0 and 1.

# import numpy as np
# import matplotlib.pyplot as plt

# def sigmoid (x):
#     return 1/(1+np.exp(-x))

# x = np.linspace(-10,10,100)
# y = sigmoid(x)

# plt.plot(x, y, color='red', label='Sigmoid Function')  # Regression Line
# plt.axhline(y=0.5, color='blue', linestyle='--', label="Threshold (0.5)")
# plt.axvline(x=0, color='green', linestyle='--', label="z = 0")


# plt.xlabel("x")
# plt.ylabel("sigmoid(x)")
# plt.title("Sigmoid Function")
# plt.legend()
# plt.show()





## PROBLEM 7B ######

# Generate Binary Classification Data:

# Library: sklearn.datasets.make_classification or make_blobs
# Task: Create a simple, linearly separable (or nearly separable) dataset for binary classification.
# Example: from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
# Visualize: Create a scatter plot (plt.scatter) of the two features (X[:, 0] vs X[:, 1]), coloring the points based on their class label (y). Use c=y argument in plt.scatter.

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=100, n_features=2, n_informative=2,            ## 100 data points with 2 features (independent variables), each data point will have one cluster, either 0 or 1
#                            n_redundant=0, n_clusters_per_class=1, random_state=42)  ## No redundant features   X is (100,2) and y = (100)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.8)      ## blue for one class and red for another

# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Binary Classification Dataset")
# plt.show()






## PROBLEM 7C #####

# Train Logistic Regression Model:

# Library: sklearn.linear_model.LogisticRegression, sklearn.model_selection.train_test_split

# Task:
# Split the data: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create Model: log_reg = LogisticRegression()
# Train Model: log_reg.fit(X_train, y_train)

# Library: sklearn.metrics (accuracy_score, precision_score, recall_score, confusion_matrix)

# Task:
# Predict Classes: y_pred = log_reg.predict(X_test)
# Predict Probabilities: y_pred_proba = log_reg.predict_proba(X_test) (Look at the output - it gives probabilities for both classes, usually class 0 and class 1).

# Calculate Metrics:
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred) (What proportion of positive identifications was actually correct?)
# recall = recall_score(y_test, y_pred) (What proportion of actual positives was identified correctly?)
# conf_matrix = confusion_matrix(y_test, y_pred)
# Print & Interpret: Print the metrics and the confusion matrix. Understand what they tell you about the model's performance.


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics


# X, y = make_classification(n_samples=100, n_features=2, n_informative=2,            ## 100 data points with 2 features (independent variables), each data point will have one cluster, either 0 or 1
#                            n_redundant=0, n_clusters_per_class=1, random_state=42)  ## No redundant features   X is (100,2) and y = (100)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)

# y_pred = log_reg.predict(X_test)       ## Prediction of classes
# y_pred_prob = log_reg.predict_proba(X_test)   ## Predict probabilities

# ## Metrics

# accuracy = metrics.accuracy_score(y_test, y_pred)
# precision = metrics.precision_score(y_test, y_pred) #(What proportion of positive identifications was actually correct?)
# recall = metrics.recall_score(y_test, y_pred) #(What proportion of actual positives was identified correctly?)
# conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print("Confusion Matrix:")
# print(conf_matrix)





######## PROBLEM 7D ######

# Plot Decision Boundary:

# Concept: Visualize the line (or hyperplane in higher dimensions) that the logistic regression model uses to separate the classes. This is more complex to code.
# Task (If time permits): Search for "plot decision boundary logistic regression python" online. You'll typically find examples that involve:
# Creating a mesh grid of points covering the feature space.
# Using the trained model to predict the class for each point on the mesh grid.
# Plotting the predictions as a contour plot (plt.contourf) underneath the scatter plot of the actual data points. This visually shows the separation learned by the model.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Generate Binary Classification Data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# Step 2: Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Plot Data Points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary of Logistic Regression")

# Step 5: Create Mesh Grid for Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),     #np.meashgrid creates a grid for both xx and yy of 100x100 Therefore when they are combined, the 2d space has 100x100 points defined by (xx,yy)
                     np.linspace(y_min, y_max, 100))

# Step 6: Predict for Each Grid Point & Plot Contour
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])      ## xx.ravel converts 2d into 1D therefore (10000, 1 ) array. When xx and yy are combined it would become (10000x2) and used to predict th class
Z = Z.reshape(xx.shape)                               ## Before this step, z is a 1d array with 10000,1, with this it is converted into (100x100) 2d matrix
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

plt.show()