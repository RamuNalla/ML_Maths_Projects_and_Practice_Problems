# class Person:
#     def __init__(s, N, A):
#         s.name = N
#         s.age = A

#     def myfunc(self):
#         print("Hello my name is " + self.name)

#     def __str__(self):
#         return f"{self.name}({self.age})"
    
#     def printname(self):
#         print(self.name, self.age)


# class Student(Person):
#     def __init__(self, Na, Ag, year):
#         super().__init__(Na, Ag)
#         self.graduationyear = year

#     def welcome(self):
#         print("Welcome", self.name, "to the class of")

    
# p1 = Student("Ramu", 28)

# p1.welcome()


# def square_nums(nums):
#     result = []
#     for i in nums:
#         result.append(i*i)
#     return result

# my_nums = square_nums([1, 2, 3, 4, 5])

# print(my_nums)


# import numpy as np

# class DataIterator:
#     def __init__(self, num_samples, batch_size):
#         self.num_samples = num_samples  # Total number of data points
#         self.batch_size = batch_size  # Number of samples per batch
#         self.data = np.random.rand(num_samples, 10)  # Simulating dataset with 10 features
#         self.labels = np.random.randint(0, 2, num_samples)  # Simulating binary labels (0 or 1)
#         self.index = 0  # Keeps track of the current batch position

#     def __iter__(self):
#         return self  # Returns the iterator object itself

#     def __next__(self):
#         if self.index >= self.num_samples:  # Stop iteration if all samples are processed
#             raise StopIteration  

#         start, end = self.index, self.index + self.batch_size  # Define batch start and end indices
#         batch_data = self.data[start:end]  # Extract batch features
#         batch_labels = self.labels[start:end]  # Extract batch labels
#         self.index += self.batch_size  # Move index to next batch
#         return batch_data, batch_labels  # Return batch

# # Using the iterator
# batch_size = 2
# num_samples = 10
# data_iter = DataIterator(num_samples, batch_size)

# batch, labels = next(data_iter)
# print (batch, labels)  # Output: (32, 10) (32,)

# # for batch, labels in data_iter:
# #     print(batch, labels)  # Output: (32, 10) (32,)


# class count:
#     def __init__(self, max_value):
#         self.max_value = max_value
#         self.current = 0
#         self.a, self.b = 0, 1

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.current > self.max_value:
#             raise StopIteration
#         value = self.a
#         self.a, self.b = self.b, self.a + self.b
#         self.current += 1
#         return value
    
# num_iterator = count(10)

# for num in num_iterator:
#     print(num)


# import numpy as np

# arr = np.array([1,2,3,4,5])
# x = arr.copy()

# x[0] = 42

# y = arr.view()

# # y[0] = 42

# print(arr.shape)
# print(x)
# print(y)


# # print(arr[::3])

# # print(arr.ndim)

##################################################################################################

# DOT PRODUCT, CROSS PRODUCT, ELEMENT WISE MULTIPLICATION

# import numpy as np 

# vector_1 = [1,2,3]
# vector_2 = [4,5,6]

# dot_product = np.dot(vector_1, vector_2)
# cross_product = np.cross(vector_1, vector_2)
# element_wise = np.multiply(vector_1, vector_2)

# print(dot_product)
# print(cross_product)    
# print(element_wise)

#**********************************************************************************************
# ARRAY RESHAPE

# arr = np.arange(1, 10)  # [1 2 3 4 5 6 7 8 9]
# reshaped = arr.reshape(3, 3)  # Converts to 3x3 matrix
# mat = np.array([[1, 2], [3, 4]])  # 2D array
# print(mat[:,1])

# mat = np.array([1,2,3,4,5,6])
# print(mat)
# mat = mat.reshape(2,3)
# print(mat)

#***********************************************************************************************
# MATRIX MUlTIPLICATION, ADDITION, TRANSPOSE, IDENTITY

# import numpy as np

# mat1 = np.array([[1,2],[3,4]])
# mat2 = np.array([[5,6], [7,8]])

# mat3 = mat1 + mat2  # Addition
# mat4 = mat1 * mat2  #Element wise multiplication

# mat5 = mat1 @ mat2   # Matrix multiplication
# mat6 = np.dot(mat1, mat2) # Using numpy
  
# mat7 = mat1.T   #Transpose

# mat8 = np.eye(3)

# print("Matrix addition:\n", mat3)
# print("Element wise multiplication: \n", mat4)
# print("Matrix multiplication using @\n", mat5)
# print("Matrix multplication using np.dot\n", mat6)
# print("Matrix Traspose:\n", mat7)
# print("Identity matrix: \n", mat8)

#***********************************************************************************************
# MATRIX VISUALIZATION USING MATPLOTLIB

# import numpy as np
# import matplotlib.pyplot as plt

# mat = np.array([[1,2,3,4,5], [6,7,8,9,0], [2,4,6,8,0]])
# mat = np.eye(10)
# plt.imshow(mat, cmap='gray', interpolation='nearest')
# plt.colorbar()  # Add color legend
# plt.title("Matrix Visualization")
# plt.show()

# fig, ax = plt.subplots()
# cax = ax.matshow(mat, cmap='coolwarm')

## Add values inside the cells

# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         ax.text(j, i, f"{mat[i, j]}", ha='center', va='center', color='black')

# plt.colorbar(cax)
# plt.title("Matrix with Annotations")
# plt.show()


#################################################################################################################
#################################################################################################################
# ************    KHAN ACADEMY **********************************

#***** 1. VECTORS

# 1A: Scalar Multiplication, Dot Product, Cross Product, Element wise multiplication, Magnitide, Normalization
#     Euclidean Distance

# import numpy as np 

# vector_1 = [1,2,3]
# vector_2 = [4,5,6]

# matrix_1 = [[1, 0, 0], [0,2,4]]
# matrix_2 = [[2, 0, 1], [1, 0, 1]]
# matrix_3 = [[1, 2, 3], [4, 5, 6]]
# matrix_4 = [[7, 8], [9, 10], [11, 12]]

# matrix_5 = [[1, 4, 3], [2, 2, 3], [1,3,7]]
# target_vector = [2,2,2]

# def vector_addition(vector_1, vector_2):
#     return np.array(vector_1) + np.array(vector_2)

# #vector_add = np.array(vector_1) + np.array(vector_2)  # Vector Addition
# scalar_multiply = 3*np.array(vector_1)  # Scalar multiplication
# dot_product = np.dot(vector_1, vector_2)
# cross_product = np.cross(vector_1, vector_2)
# element_wise = np.multiply(vector_1, vector_2)
# magnitude = np.linalg.norm(vector_1)
# normalized_vector1 = vector_1/magnitude
# euclidean_distance = np.linalg.norm(np.array(vector_1) - np.array(vector_2))

# dim_matrix_1 = np.array(matrix_1).shape 
# matrix_add = np.array(matrix_1) + np.array(matrix_2)

# def matrix_multiply(mat_1, mat_2):                  ########### Function to calculate two matrices multiplication
#     mat1 = np.array(mat_1)
#     mat2 = np.array(mat_2)
#     if mat1.shape[1] != mat2.shape[0]:
#         raise ValueError("Matrix multiplication not possible")
#     return mat1 @ mat2    # we can also use np.dot(mat1, mat2)

# matrix_1_Transpose = np.transpose(np.array(matrix_1))  ## or np.array(matrix_1).T

# def closest_vector(matrix_5, target_vector):           ######## Function to compute closest vector to the target vector
#     matrix_5 = np.array(matrix_5)
#     target_vector = np.array(target_vector)
#     lowest_distance = float('inf')
#     id = -1

#     distances = np.linalg.norm(matrix_5 - target_vector, axis = "1")   # axis = "1" means row wise operations, target vector is subtracted from each row
#     id = np.argmin(distances)   # Returns the index

#     # for i in range(matrix_5.shape[0]):
#     #     distance = np.linalg.norm(matrix_5[i] - target_vector)
#     #     if distance < lowest_distance:
#     #         lowest_distance = distance
#     #         id = i
#     return matrix_5[id]

# closest_vector1 = closest_vector(matrix_5, target_vector)


# print("Vector Addition: ", vector_addition(vector_1, vector_2))
# print("Scalar Multiplication: ", scalar_multiply)
# print("Dot Product: ", dot_product)
# print("Cross Product: ",cross_product)    
# print("Element wise multiplication: ",element_wise)
# print("Vector magnitude: ", magnitude)
# print("Normalizing vector1: ",normalized_vector1)
# print("Euclidean distance between vectors: ", euclidean_distance)

# print("Matrix_1 Dimension: ", dim_matrix_1)
# print("Matrix Addition: ", matrix_add)
# print("Matrix Multiplication: ", matrix_multiply(matrix_3, matrix_4))
# print("Matrix 1 Transpose: ", matrix_1_Transpose)

# print("Closest Vector Computation: ", closest_vector1)






# 1B: Visualizing Vectors using Matplotlib (QUIVER FUNCTION)

# import numpy as np
# import matplotlib.pyplot as plt

# A = np.array([2,4])
# B = np.array([4,6])

# fig, ax =  plt.subplots()
# ax.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label="Vector A")
# ax.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', label="Vector B")

# ax.set_xlim(-1, 8)
# ax.set_ylim(-1, 8)

# # Add grid and labels
# plt.grid()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('2D Vector Visualization')
# plt.legend()
# plt.show()


# 1C: Representing an image as a matrix

# import numpy as np
# import matplotlib.pyplot as plt

# image = np.array([
#     [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Row 1 (Red, Green, Blue)
#     [[255, 255, 0], [0, 255, 255], [255, 0, 255]],  # Row 2 (Yellow, Cyan, Magenta)
#     [[0, 0, 0], [127, 127, 127], [255, 255, 255]]  # Row 3 (Black, Gray, White)
# ], dtype=np.uint8)

# pixel_vector = image[0,0]

# plt.imshow(image)
# plt.axis("off")
# plt.show()

# # image_float = image.astype(np.float32)  # Convert to float
# # plt.imshow(image_float)  # Some libraries may misinterpret values
# # plt.show()

# print("RGB vector of first pixel ", pixel_vector)






#***** 2. MATRICES*****************************************************************************************************************************

# 2A: ROTATION MATRIX

# import numpy as np
# import matplotlib.pyplot as plt

# deg_of_rotation = 60
# rotation_matrix = np.array([
#     [np.cos(np.deg2rad(deg_of_rotation)), -np.sin(np.deg2rad(deg_of_rotation))], 
#     [np.sin(np.deg2rad(deg_of_rotation)), np.cos(np.deg2rad(deg_of_rotation))]
# ])

# input_vector = np.array([1,1])
# output_vector = np.dot(rotation_matrix, input_vector)

# print("Output vector: ", output_vector)

# fig, ax =  plt.subplots()
# ax.quiver(0, 0, input_vector[0], input_vector[1], angles='xy', scale_units='xy', scale=1, color='b', label="Input Vector")
# ax.quiver(0, 0, output_vector[0], output_vector[1], angles='xy', scale_units='xy', scale=1, color='r', label="Output Vector")
# max_val = max(abs(output_vector).max(), abs(input_vector).max()) + 0.5

# ax.set_xlim(-max_val, max_val)
# ax.set_ylim(-max_val, max_val)

# # Add grid and labels
# plt.grid()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('2D Vector Rotation')
# # **Text annotations for clarity**
# ax.text(input_vector[0], input_vector[1], ' Input Vector', fontsize=12, color='blue', verticalalignment='bottom', horizontalalignment='right')
# ax.text(output_vector[0], output_vector[1], ' Output Vector', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='right')

# plt.legend()
# plt.show()





# 2B: SCALING and SHEARING

# import numpy as np
# import matplotlib.pyplot as plt

# scale_x = 2
# scale_y = 0.5
# scaling_matrix = np.array([
#     [scale_x, 0], 
#     [0, scale_y]
# ])

# shearing_matrix = np.array([
#     [2,1],
#     [0.5,1]
# ])

# square = np.array([
#     [0, 0],  # Bottom-left
#     [1, 0],  # Bottom-right
#     [1, 1],  # Top-right
#     [0, 1],  # Top-left
#     [0, 0]   # Closing the shape
# ])

# scaled_square = np.dot(square, scaling_matrix.T)    #### Be very careful here, do a simple calculation
# sheared_square = np.dot(square, shearing_matrix.T)

# print("Output Scaled vector: ", scaled_square)
# print("Output Sheared vector: ", sheared_square)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# # Original and Scaled
# ax[0].plot(square[:, 0], square[:, 1], 'bo-', label='Original Square')  # All x values and y values
# ax[0].plot(scaled_square[:, 0], scaled_square[:, 1], 'ro-', label='Scaled Square')
# ax[0].set_title('Scaling Transformation')
# ax[0].legend()
# ax[0].grid()
# ax[0].set_xlim(-1, 3)
# ax[0].set_ylim(-1, 2)

# # Original and Sheared
# ax[1].plot(square[:, 0], square[:, 1], 'bo-', label='Original Square')
# ax[1].plot(sheared_square[:, 0], sheared_square[:, 1], 'go-', label='Sheared Square')
# ax[1].set_title('Shearing Transformation')
# ax[1].legend()
# ax[1].grid()
# ax[1].set_xlim(-1, 3)
# ax[1].set_ylim(-1, 2)

# plt.show()




# 2C: SOLVE A SYSTEM OF LINEAR EQUATIONS DIRECTLY and GAUSSIAN ELIMINATION

# import numpy as np

# A = np.array([
#     [2,3,-1],
#     [4,1,2],
#     [-2,5,-3]
# ])

# B = np.array([5,6,-4])

# def gaussian_elimination(A, B):
#     A = A.astype(float)  # Ensure floating-point division  ####### This is important 
#     B = B.astype(float)  # Convert to float
    
#     n = len(B)

#     for i in range(n):
#         max_row = np.argmax(np.abs(A[i:,i])) + i # find the row with maximum value in ith column, Since the search is from 
#         if A[max_row, i] == 0:
#             raise ValueError("Matrix is singular or nearly singular!")
        
#         A[[i,max_row]] = A[[max_row, i]]  # Swapping the rows of A. NumPy allows fancy indexing, meaning A[[i, max_row]] = A[[max_row, i]] swaps rows without needing a temporary variable.
#         B[[i,max_row]] = B[[max_row, i]]  

#         for j in range(i+1,n):
#             factor = A[j,i]/A[i,i]
#             A[j, i:] -= factor * A[i,i:]   # applying the factor for each element of jth row. Therefore i:
#             B[j] -= factor * B[i]
    
#     s = np.zeros(n)    
#     for i in range(n-1, -1, -1):         # Back substitution to solve Ux = b, U is the upper triangle matrix
#         s[i] = (B[i] - np.dot(A[i, i+1:], s[i+1:]))/A[i, i]

#     return s


# A_inverse = np.linalg.inv(A)
# solution = np.linalg.solve(A,B)
# solution_gaussian = gaussian_elimination(A,B)

# print("Solution: ", solution_gaussian)
# print("Inverse of A: ", A_inverse)




## 2D: SOLVING SYSTEM OF EQUATIONS WITH LEAST SQUARED METHOD  (INFINITELY MANY SOLUTIONS)
# If infinitely many solutions exist, the least squares method returns the minimum norm solution, which is 
# the smallest possible solution in terms of magnitude.

# import numpy as np

# A = np.array([[2, 4], [1, 2]])  # Singular matrix, det(A) = 0
# B = np.array([6, 3])

# x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

# print("Solution:", x)             ## The solution present is one case if (x,y) that fits into the equation, it has infinite solutions
# print("Residuals:", residuals)   ## Residuals empty meaning infinitely many solutions  
# print("Rank of A:", rank)
# print("Singular values of A:", s)



## 2E: SOLVING SYSTEM OF EQUATIONS WITH LEAST SQUARED METHOD  (NO SOLUTIONS)

# import numpy as np

# A = np.array([[1, 1], [2, 2], [3, 3]])  # Linearly dependent rows
# B = np.array([1, 2, 10])  # Inconsistent last equation

# x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

# print("Solution:", x)             ## The solution present is one case if (x,y) that fits into the equation, it has infinite solutions
# print("Residuals:", residuals)   ## Residuals nonzero means no exact solution 
# print("Rank of A:", rank)
# print("Singular values of A:", s)



## 2F: SOLVING SYSTEM OF EQUATIONS WITH LEAST SQUARED METHOD  (UNIQUE SOLUTION)

# import numpy as np

# A = np.array([
#     [2,3,-1],
#     [4,1,2],
#     [-2,5,-3]
# ])

# B = np.array([5,6,-4])

# x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

# print("Solution:", x)             ## The solution present is one case if (x,y) that fits into the equation, it has infinite solutions
# print("Residuals:", residuals)   ## Residuals will be empty here also, means system is solvable with no error
# print("Rank of A:", rank)
# print("Singular values of A:", s)


## 2G: SOLUTION CHECK FOR UNIQUE SOLUTION, INFINITELY MANY SOLUTIONS, NO SOLUTION


# import numpy as np

# A = np.array([
#     [2,3,-1],
#     [4,1,2],
#     [-2,5,-3]
# ])

# B = np.array([5,6,-4])

# determinant = np.linalg.det(A)
# rank_A = np.linalg.matrix_rank(A)
# rank_augmented = np.linalg.matrix_rank(np.column_stack((A, B)))

# if rank_A == A.shape[1] and determinant != 0:
#     print("The system of equations has a unique solution")
# elif rank_A == rank_augmented and determinant == 0:
#     print("The system of equations has infintely many solutions")
# elif rank_A < rank_augmented:
#     print("The system has no solution")

# augmented_matrix = np.column_stack((A, B))   ## Correct usage of column_stack
# print(augmented_matrix)




## 2H: FLIPPING AN IMAGE HORIZONTALLY and SLICING

# import numpy as np
# import matplotlib.pyplot as plt

# image = np.array([
#     [  0,  50, 100, 150, 200], 
#     [ 25,  75, 125, 175, 225], 
#     [ 50, 100, 150, 200, 250], 
#     [ 25,  75, 125, 175, 225], 
#     [  0,  50, 100, 150, 200]
# ], dtype=np.uint8)

# plt.subplot(1,3,1)
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.title("Original Image")
# plt.axis("off")

# flipped_image = image[:, ::-1]    # flips all the columns
# sliced_image = image[:, 0:3]

# plt.subplot(1,3,2)
# plt.imshow(flipped_image, cmap='gray', vmin=0, vmax=255)
# plt.title("Flipped Image")
# plt.axis("off")

# plt.subplot(1,3,3)
# plt.imshow(sliced_image, cmap='gray', vmin=0, vmax=255)
# plt.title("Sliced Image")
# plt.axis("off")

# plt.show()






#***** 3. EIGEN VALUES, EIGEN VECTORS*****************************************************************************************************************************


## 3A: Calculating Eigen values and Eigen Vectors and verifying AV = lambda*V

# import numpy as np

# A = np.array([
#     [2,3,-1],
#     [4,1,2],
#     [-2,5,-3]
# ])

# eig_values, eig_vectors = np.linalg.eig(A)

# print("Eigen Values: ", eig_values)
# print("Eigen Vectors:\n", eig_vectors)

# for i in range(A.shape[0]):
#     left_side = A @ eig_vectors[:, i]
#     right_side = eig_values[i]*eig_vectors[:, i]

#     print(f"A*V {i}: \n", left_side)
#     print(f"lambda*V {i}: \n", right_side)




## 3B: Visualizing Eigen Vectors in 2D 

# import numpy as np
# import matplotlib.pyplot as plt

# A = np.array([[3, 1],
#               [1, 3]])

# eig_values, eig_vectors = np.linalg.eig(A)

# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)

# origin = np.zeros((2, 2))

# plt.quiver(origin[0], origin[1], eig_vectors[0], eig_vectors[1], angles='xy', scale_units='xy', scale=1, color=['r', 'b'], label="Eigenvectors")

# transformed_vectors = A @ eig_vectors
# plt.quiver(origin[0], origin[1], transformed_vectors[0], transformed_vectors[1], angles='xy', scale_units='xy', scale=1, color=['r', 'b'], alpha=0.5, label="Transformed vectors")

# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.grid()
# plt.legend()
# plt.title("Eigenvectors and Their Transformations")
# plt.show()


## 3C: Generate a Random Correlated data and plot it, Calculating the covariance matrix

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)  # To ensure same random numbers are created when rerun the code

# X = np.random.randn(100,2)

# correlation_matrix = np.array([[2,1],[1,1]])

# correlated_data = X @ correlation_matrix

# plt.figure(figsize=(6, 6))     ## 6 inches width an 6 inches height
# plt.scatter(correlated_data[:, 0], correlated_data[:, 1], alpha=0.5, color="blue")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.title("Generated Correlated 2D Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()





## 3D: Calculating the covariance matrix manually and verifying with numpy calculation, Plotting the eigen vectors of covariance matrix

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)  # To ensure same random numbers are created when rerun the code

# X = np.random.randn(100,2)   ## 100 rows and 2 columns ==> 100 samples and 2 features/variables

# correlation_matrix = np.array([[2,1],[1,1]])
# correlated_data = X @ correlation_matrix

# m = correlated_data.shape[0]             ## m = number of samples, rows. Covariance matrix uses division by number of samples

# correlated_data_mean = np.mean(correlated_data, axis = 0)    # axis = 0 means column wise mean ==> it generates an array with each column means
# correlated_data_centered = correlated_data - correlated_data_mean   # Python broadcasting
# covariance_matrix = (1/(m-1))*np.dot(correlated_data_centered.T, correlated_data_centered)

# print("Covariance matrix using manual computation:\n", covariance_matrix)

# covariance_matrix_python = np.cov(correlated_data, rowvar = False)   #rowvar is useful ==> rowvar = False means Rows are samples and columns are features, if #rowvar is 
# print("Covariance matrix using Python:\n", covariance_matrix_python)

# #Interpreting the covariance matrix
# #  Var(feature1) is greater than Var(feature2): this is expected due to the correlation matrix we selected. Feature 1 and Feature 2 are positively correlated

# eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

# plt.figure(figsize = (7,7))
# plt.scatter(correlated_data[:,0], correlated_data[:,1], alpha = 0.5, color = "blue", label="Correlated data" )

# origin = correlated_data_mean

# for i in range(len(eig_values)):
#     plt.quiver(*origin, *eig_vectors[:, i]*np.sqrt(eig_values[i]), angles='xy', scale_units='xy', scale=1.5, color=['red', 'green'][i],
#                 label = f"Eigevector {i+1} (λ = {eig_values[i]:.2f})")

#                 # Scaling by sqrt(eigen_value) represents the actual spread of data along each eigen vector

# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.title("Correlated 2D Data with Covariance Eigenvectors")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.show()






## 3D: Calculating PCA using standard library and verifying the manual calculation

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# np.random.seed(42)  # To ensure same random numbers are created when rerun the code

# X = np.random.randn(100,2)   ## 100 rows and 2 columns ==> 100 samples and 2 features/variables

# correlation_matrix = np.array([[2,1],[1,1]])
# correlated_data = X @ correlation_matrix

# m = correlated_data.shape[0]             ## m = number of samples, rows. Covariance matrix uses division by number of samples

# correlated_data_mean = np.mean(correlated_data, axis = 0)    # axis = 0 means column wise mean ==> it generates an array with each column means
# correlated_data_centered = correlated_data - correlated_data_mean   # Python broadcasting
# covariance_matrix = (1/(m-1))*np.dot(correlated_data_centered.T, correlated_data_centered)

# covariance_matrix_python = np.cov(correlated_data, rowvar = False)   #rowvar is useful ==> rowvar = False means Rows are samples and columns are features, if #rowvar is 

# eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
# pc1_manual = correlated_data_centered @ eig_vectors[:,0]  # #Manually calculating the PC1 using eigen vectors of covariance matrix


# # Calculating PCs from sklearn library
# pca_library = PCA(n_components = 2)
# pca_library.fit(correlated_data_centered)

# pc1_sklearn = pca_library.transform(correlated_data_centered)[:,0]  #first column making it the Principal component 1

# print("Manual PC1 projection (first 5 values): \n", pc1_manual[:5])
# print("Sklearn PC1 projection (first 5 columns): \n", pc1_sklearn[:5])

# fig, ax = plt.subplots(figsize=(7, 7))

# plt.scatter(correlated_data_centered[:,0], correlated_data_centered[:,1], alpha = 0.5, color = "blue", label="Correlated data" )  ## Plotting the correlated data

# origin = correlated_data_mean
# for i in range(len(eig_values)):
#     plt.quiver(*origin, *eig_vectors[:, i]*np.sqrt(eig_values[i]), angles='xy', scale_units='xy', scale=1.5, color=['red', 'green'][i],  ## Plotting the Eigen vectors (PC1 direction)
#                 label = f"Eigevector {i+1} (λ = {eig_values[i]:.2f})")


# projected_points = np.outer(pc1_manual, eig_vectors[:, 0])  ## Project the 1D PC1 into 2D plan ==> All projected points lie along the eigen vector of PC1

# ax.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label="Projection onto PC1", alpha=0.7)  ## Plotting the projected points into PC1

# for i in range(len(correlated_data_centered)):     
#     ax.plot([correlated_data_centered[i, 0], projected_points[i, 0]],            # Connecting the actual point to the projected point
#             [correlated_data_centered[i, 1], projected_points[i, 1]], 
#             'k--', linewidth=0.5)

# ax.axhline(0, color='black', linewidth=0.5)
# ax.axvline(0, color='black', linewidth=0.5)
# ax.grid(True, linestyle="--", linewidth=0.5)
# ax.set_title("PCA Projection onto First Principal Component")
# ax.set_xlabel("Feature 1 (Centered)")
# ax.set_ylabel("Feature 2 (Centered)")
# ax.legend()
# plt.show()




## 3E: Apply PCA to a toy image dataset for dimensionality reduction

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_digits

# digits = load_digits()  

# X = digits.data  #each data is a 64 dimenesional features (variables) with 1797 samples count
# y = digits.target

# n_comp = 20  # reducing 64 to 20 principal components
# pca = PCA(n_components = n_comp)
# X_pca = pca.fit_transform(X)       #This gives the projected data of original data along 20PCs  (Size: 1797 x 20)

# X_reconstructed = pca.inverse_transform(X_pca)     # Reconstruct back into original dataset (Size: 1797 x 64)

# fig, axes = plt.subplots(2, 10, figsize=(10, 3))

# for i in range(10):
#     axes[0,i].imshow(X[i].reshape(8,8), cmap = 'gray')    #reshape makes the 1x64 vector into 8x8 matrix
#     axes[0,1].axis('off')

#     axes[1,i].imshow(X_reconstructed[i].reshape(8,8), cmap = 'gray')
#     axes[1,i].axis('off')

# axes[0, 0].set_title("Original Images")
# axes[1, 0].set_title(f"Reconstructed with {n_comp} PCs")
# plt.show()

