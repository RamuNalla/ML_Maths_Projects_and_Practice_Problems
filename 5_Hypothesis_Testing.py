
############ HYPOTHESIS TESTING  #################

### 5A: GENERATE SAMPLE DATA:

# Task: Create two NumPy arrays representing samples from two different groups (e.g., test scores from Group A and Group B). Make them potentially come from distributions with slightly different means but similar variances.

# Example: group_a = scipy.stats.norm.rvs(loc=75, scale=10, size=30)
# Example: group_b = scipy.stats.norm.rvs(loc=80, scale=10, size=30)

# import numpy as np
# import scipy.stats as sci

# group_a = sci.norm.rvs(loc = 75, scale = 10, size = 30)
# group_b = sci.norm.rvs(loc = 80, scale = 10, size = 30)

# # Print the generated data (optional)
# print("Group A:")
# print(group_a)
# print("\nGroup B:")
# print(group_b)

# # Calculate and print means and standard deviations to verify
# print("\nGroup A Statistics:")
# print(f"  Mean: {np.mean(group_a):.2f}")
# print(f"  Standard Deviation: {np.std(group_a):.2f}")

# print("\nGroup B Statistics:")
# print(f"  Mean: {np.mean(group_b):.2f}")
# print(f"  Standard Deviation: {np.std(group_b):.2f}")





## PROBLEM 5B ########

# Perform Independent t-test:

# Library: scipy.stats.ttest_ind
# Task: Use scipy.stats.ttest_ind(group_a, group_b) to perform an independent two-sample t-test. This test assumes the two samples are independent and checks if their means are significantly different.
# Output: The function returns a test statistic and a p-value. Store these. statistic, p_value = scipy.stats.ttest_ind(group_a, group_b)


# import numpy as np
# import scipy.stats as stats

# group_a = stats.norm.rvs(loc = 75, scale = 10, size = 30)
# group_b = stats.norm.rvs(loc = 80, scale = 10, size = 30)

# statistic, p_value = stats.ttest_ind(group_a, group_b)

# print(f"T-statistic: {statistic:.4f}")  # calculted t-statistic
# print(f"P-value: {p_value:.4f}")

# alpha = 0.05  # significance level

# if p_value < alpha:
#     print("The p-value is less than the significance level (0.05).")
#     print("Therefore, we reject the null hypothesis.")
#     print("There is statistically significant evidence that the means of Group A and Group B are different.")
# else:
#     print("The p-value is greater than or equal to the significance level (0.05).")
#     print("Therefore, we fail to reject the null hypothesis.")
#     print("There is not enough statistically significant evidence that the means of Group A and Group B are different.")





## PROBLEM 5C ########

# Experiment with Data:

# Task: Rerun the process with different scenarios:

# No Difference: Generate group_a and group_b from the same distribution (e.g., both loc=75). What happens to the p-value (it should generally be > 0.05)?
# Large Difference: Generate groups with a much larger difference in means (e.g., loc=75 vs loc=95). What happens to the p-value (it should become very small)?
# Smaller Sample Size: Reduce the size parameter (e.g., size=10 for both groups). How does this affect the p-value, even if the means are different? (Smaller samples make it harder to detect differences).
# Different Variances: Generate groups with different standard deviations (scale). Note: The standard ttest_ind assumes equal variances. You can add the argument equal_var=False to perform Welch's t-test, which doesn't assume equal variances. Compare results.


# import numpy as np
# import scipy.stats as stats

# def run_ttest(group_a, group_b, equal_var=True):
#     statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
#     print(f"T-statistic: {statistic:.4f}")
#     print(f"P-value: {p_value:.4f}")

#     alpha = 0.05  # Significance level

#     print("Interpretation:")
#     if p_value < alpha:
#         print("Reject null hypothesis: means are different.")
#     else:
#         print("Fail to reject null hypothesis: means are not significantly different.")
#     print("-" * 20)

# # 1. No Difference
# print("Scenario 1: No Difference")
# group_a = stats.norm.rvs(loc=75, scale=10, size=30)
# group_b = stats.norm.rvs(loc=75, scale=10, size=30)
# run_ttest(group_a, group_b)

# # 2. Large Difference
# print("Scenario 2: Large Difference")
# group_a = stats.norm.rvs(loc=75, scale=10, size=30)
# group_b = stats.norm.rvs(loc=95, scale=10, size=30)
# run_ttest(group_a, group_b)

# # 3. Smaller Sample Size
# print("Scenario 3: Smaller Sample Size")
# group_a = stats.norm.rvs(loc=75, scale=10, size=10)
# group_b = stats.norm.rvs(loc=80, scale=10, size=10)
# run_ttest(group_a, group_b)

# # 4. Different Variances
# print("Scenario 4: Different Variances (Equal Variance Assumed)")
# group_a = stats.norm.rvs(loc=75, scale=10, size=30)
# group_b = stats.norm.rvs(loc=80, scale=15, size=30) # Different scale/variance
# run_ttest(group_a, group_b)

# print("Scenario 4: Different Variances (Welch's t-test)")
# run_ttest(group_a, group_b, equal_var=False)






## PROBLEM 5D ########

# Permutation Test Implementation:

# Concept: A non-parametric way to test the difference between means without assuming normality. The idea is to shuffle the labels (which group each data point belongs to) many times and see how often the observed difference in means (or something more extreme) occurs just by chance in the shuffled data.

# Task:

# Combine group_a and group_b into one array.
# Calculate the observed difference in means between the original group_a and group_b.
# Repeat many times (e.g., 1000 or 10000):
# Shuffle the combined array (np.random.shuffle).
# Split the shuffled array back into two pseudo-groups of the original sizes.
# Calculate the difference in means between these shuffled pseudo-groups.
# Store this difference.
# Calculate the p-value: Count how many times the absolute difference from the shuffled data was greater than or equal to the absolute observed difference. Divide this count by the total number of repetitions.

# Compare: Compare the p-value from the permutation test to the p-value from the t-test.

import numpy as np
from scipy.stats import norm, ttest_ind
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility

# Generate random data from normal distributions
group_a = norm.rvs(loc=75, scale=10, size=30)  # Mean = 75, SD = 10
group_b = norm.rvs(loc=80, scale=10, size=30)  # Mean = 80, SD = 10

obs_diff = np.abs(np.mean(group_a) - np.mean(group_b))
print(f"Observed Difference in Means: {obs_diff:.4f}")

#Permutation test

num_permutations = 10000
permuted_diffs = []

combined = np.concatenate([group_a, group_b])  # Merging both groups into a large array

for i in range(num_permutations):
    np.random.shuffle(combined)     # shuffle the combined array

    new_a = combined[:len(group_a)]    
    new_b = combined[len(group_b):]

    permuted_diff = np.abs(np.mean(new_a) - np.mean(new_b))     ## Absolute value will ensure it is a two-tailed test
    permuted_diffs.append(permuted_diff)


permuted_diffs = np.array(permuted_diffs)
p_value = np.mean(permuted_diffs >= obs_diff)    ## p-value is the proportion of permuted differences greater than the observed difference

print(f"P-value from Permutation Test: {p_value:.4f}")

t_stat, p_value_ttest = ttest_ind(group_a, group_b, equal_var=False)  ## computing the p-vallue from welch test (non equal variances)
print(f"P-value from Welch's t-test: {p_value_ttest:.4f}")         ## Wlch test is two-tailed test by default

# Plot the distribution of permuted differences
plt.figure(figsize=(8, 5))
plt.hist(permuted_diffs, bins=60, alpha=0.7, color='red', edgecolor='black', density=True)
plt.axvline(obs_diff, color='blue', linestyle='dashed', linewidth=2, label=f'Observed Diff: {obs_diff:.4f}')

plt.xlabel('Difference in Means')
plt.ylabel('Density')
plt.title('Permutation Test: Distribution of Permuted Mean Differences')
plt.legend()
plt.show()



