
## DESCRIPTIVE STATISTICS AND DATA EXPLORATION



## PROBLEM 4A

# NumPy Statistics:

# Task: Create a NumPy array with ~20 numbers (e.g., exam scores).

# Calculate (using NumPy functions): np.mean, np.median, np.std, np.var, np.min, np.max, np.percentile (e.g., 25th, 50th, 75th percentiles - which correspond to quartiles). Print and interpret the results.

# import numpy as np

# exam_scores = np.array([78, 92, 85, 60, 95, 70, 88, 75, 80, 90,
#                          65, 82, 79, 93, 86, 72, 89, 68, 91, 84])

# mean_score = np.mean(exam_scores)
# median_score = np.median(exam_scores)
# std_dev = np.std(exam_scores)
# variance = np.var(exam_scores)
# min_score = np.min(exam_scores)
# max_score = np.max(exam_scores)
# percentile_25 = np.percentile(exam_scores, 25)
# percentile_50 = np.percentile(exam_scores, 50)  # Same as median
# percentile_75 = np.percentile(exam_scores, 75)

# # Print results
# print(f"Mean Score: {mean_score:.2f}")
# print(f"Median Score: {median_score:.2f}")
# print(f"Standard Deviation: {std_dev:.2f}")
# print(f"Variance: {variance:.2f}")
# print(f"Minimum Score: {min_score}")
# print(f"Maximum Score: {max_score}")
# print(f"25th Percentile (Q1): {percentile_25:.2f}")
# print(f"50th Percentile (Median/Q2): {percentile_50:.2f}")
# print(f"75th Percentile (Q3): {percentile_75:.2f}")



## PROBLEM 4B & 4C
# Introduction to Pandas:

# Task: Create a Pandas DataFrame from a dictionary. E.g., {'Name': ['Alice', 'Bob', 'Charlie'], 'Score': [85, 92, 78], 'Hours_Studied': [10, 15, 8]}.

# Explore: Print the DataFrame. Use .head(), .info(), .describe() methods. .describe() is very useful as it calculates many descriptive stats at once.
# Calculate: Select the 'Score' column (a Pandas Series) and calculate its mean and standard deviation using Pandas methods (.mean(), .std()).

# Library: matplotlib.pyplot as plt

# Task: Use the NumPy array or the 'Score' column from the Pandas DataFrame.

# Plot:
# Create a histogram (plt.hist()) to visualize the distribution of scores. Experiment with the bins parameter. Add labels (plt.xlabel, plt.ylabel) and a title (plt.title). Use plt.show().
# Create a box plot (plt.boxplot()) to visualize the median, quartiles, and potential outliers.


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Score': [85, 92, 78], 'Hours_Studied': [10, 15, 8]})

# print("df.head():\n", df.head())
# print("df.info():\n", df.info())
# print("df.describe():\n", df.describe())

# score = df['Score']

# print('Mean of the score:', score.mean())
# print('STD of the score:', score.std())

# fig, axes = plt.subplots(2,1,figsize=(6,8))

# axes[0].hist(df['Score'], bins=3, edgecolor='black', color='red')
# axes[0].set_xlabel("Score")
# axes[0].set_ylabel("Frequency")
# axes[0].set_title("Histogram of the Scores")

# # Boxplot Plot
# axes[1].boxplot(df['Score'])
# axes[1].set_xlabel("Score")
# axes[1].set_ylabel("Value") # Corrected ylabel
# axes[1].set_title("Boxplot of the Scores")

# plt.show()