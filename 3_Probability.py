
######## 3: PROBABILITY BASICS ####################################W

## PROBLEM 3A

# Dice Roll Simulation:

# Task: Simulate rolling two standard six-sided dice 10,000 times using numpy.random.randint.
# Calculate: Estimate the probability of:
#   The sum being exactly 7.
#   The sum being greater than 8.
#   Rolling doubles.
# Compare: Compare your simulated probabilities to the exact theoretical probabilities.


# import numpy as np

# num_of_rolls = 10000

# count_sum_7 = 0
# count_sum_greater_8 = 0
# count_doubles = 0

# for i in range(num_of_rolls):
#     x1 = np.random.randint(1, 7)
#     x2 = np.random.randint(1, 7)
    
#     if x1+x2 ==7:
#         count_sum_7 += 1
#     if x1+x2 > 8:
#         count_sum_greater_8 += 1
#     if x1==x2:
#         count_doubles += 1
    
    
# th_prob_sum_7 = (6/36)             ## Theoretical probability is not influenced by number of trials
# th_prob_greater_8 = 10/36   
# th_prob_doubles = 6/36       

# prob_sum_7 = count_sum_7/num_of_rolls   ## Experimental prob changes with the number of trials but converges to the theoretical prob as trials increase
# prob_sum_greater_8 = count_sum_greater_8/num_of_rolls
# prob_doubles = count_doubles/num_of_rolls

# print(f"Experimental Probability of sum 7: {prob_sum_7:.4f} | Theoretical: {th_prob_sum_7:.4f}")
# print(f"Experimental Probability of sum >8: {prob_sum_greater_8:.4f} | Theoretical: {th_prob_greater_8:.4f}")
# print(f"Experimental Probability of doubles: {prob_doubles:.4f} | Theoretical: {th_prob_doubles:.4f}")




### Problem 3B

# Card Drawing Simulation (Without Replacement):

# Task: Simulate drawing two cards from a standard 52-card deck without replacement. Represent the deck (e.g., list of strings like '2H', 'KD', 'AC'). Use numpy.random.choice with replace=False. Repeat the simulation many times (e.g., 5000).
# Estimate: The probability of drawing two Aces.
# Calculate: The exact theoretical probability ((4/52) * (3/51)) and compare.

# import numpy as np

# suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
# ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# num_of_sim = 10000
# ace_count = 0
# deck = []

# for i in range(len(suits)):
#     for j in range(len(ranks)):
#         card = ranks[j] + suits[i]  # Concatenate two strings  ## IT SHOULD BE RANKS + SUITS AND NOT SUITS + RANK. IF OTHERWISE SOME CASES LIKE 10C, THE SECOND ELEMENT IS A DIGIT
#         deck.append(card)

# #OR
# # deck = [rank + suit for suit in suits for rank in ranks]

# for i in range(num_of_sim):
#     drawn_cards = np.random.choice(deck, 2, replace=False) ## select two cards from deck
#     if drawn_cards[0][0] == 'A' and drawn_cards[1][0] == 'A':
#         ace_count += 1


# prob_ace = ace_count/num_of_sim   
# th_prob_ace = ((4/52)*(3/51))

# print(f"Estimated probability of drawing two Aces in {num_of_sim} simulations: {prob_ace:.4f}")
# print(f"Theoretical probability of drawing two Aces: {th_prob_ace:.4f}")





### Problem 3C

# CONDITIONAL PROBABILITY FUNCTION

# Task: Write a Python function conditional_probability(prob_A_and_B, prob_B) that takes P(A and B) and P(B) as input and returns P(A|B). Include checks to ensure prob_B is not zero.
# Test: Use it with simple examples (e.g., from dice rolls or card draws).


# import numpy as np

# def cond_prob(prob_A_and_B, prob_B):
    
#     if prob_B == 0:
#         raise ValueError("P(B) cannot be zero")
    
#     return prob_A_and_B/prob_B

# p_A_and_B = 1/6
# p_B = 3/6    

# # - Even numbers: {2, 4, 6} → P(B) = 3/6 = 0.5
# # - Rolling a 6 is in this subset → P(A and B) = 1/6

# result = cond_prob(p_A_and_B, p_B)
# print(f"P(A | B) = {result:.4f}")




#### PROBLEM 3D


# Bayes' Theorem Implementation:

# Task: Write a Python function bayes_theorem(prob_B_given_A, prob_A, prob_B) that implements Bayes' Theorem P(A|B) = [P(B|A) * P(A)] / P(B).
# Solve Classic Problem: Use your function to solve a standard Bayes' Theorem word problem. A common one is the "Disease Test" problem:
# A disease affects 1% of the population (P(Disease) = 0.01).
# A test for the disease is 98% accurate for people with the disease (P(Positive Test | Disease) = 0.98).
# The test has a 5% false positive rate (meaning it incorrectly shows positive for 5% of people without the disease, so P(Positive Test | No Disease) = 0.05).

# Question: If a randomly selected person tests positive, what is the actual probability they have the disease? i.e., find P(Disease | Positive Test).

# Hint: You'll need to calculate the overall probability of testing positive, P(Positive Test), using the law of total probability: P(Positive Test) = P(Positive Test | Disease) * P(Disease) + P(Positive Test | No Disease) * P(No Disease).

# import numpy as np

# def bayes_theorem(prob_B_given_A, prob_A, prob_B):
    
#     if prob_B == 0:
#         raise ValueError("P(B) cannot be zero")
    
#     return (prob_B_given_A*prob_A/prob_B)

# #Solving disease test problem

# p_disease = 0.01                           # Prevalence of the disease across general population 
# p_positive_given_disease = 0.98            # Accuracy of the medical test given that the patient has the disease
# p_positive_given_no_disease = 0.05         # False positives of the medical test, incorrectly shows as positive without the disease


# p_positive = (p_positive_given_disease*p_disease) + (p_positive_given_no_disease*(1-p_disease))  # computing p_+ve irrespective of disease or no disease

# p_disease_given_positive = bayes_theorem(p_positive_given_disease, p_disease, p_positive)

# print(f"Prob. of having the actually having disease when tested positive = {p_disease_given_positive:.4f}")





#### PROBLEM 3E

# Monty Hall Problem Simulation:

# Task: Simulate the Monty Hall game show problem many times (e.g., 10,000 trials for each strategy).
# Setup: 3 doors, 1 car, 2 goats. Contestant picks a door. Host (who knows where the car is) opens one of the other doors revealing a goat. Contestant can switch or stay.
# Simulate 'Stay' Strategy: Record the win percentage.
# Simulate 'Switch' Strategy: Record the win percentage.
# Verify: Confirm that the 'Switch' strategy results in approximately a 2/3 win probability, while the 'Stay' strategy results in approximately a 1/3 win probability. This demonstrates counter-intuitive conditional probabilities.

# import numpy as np

# num_of_sims = 10000

# items_behind_door = ['car', 'goat', 'goat']
# count_stay_car = 0

# ## STAY STRATEGY COUNT
# for i in range(num_of_sims):
#     pick = np.random.randint(0, 3)
#     if items_behind_door[pick] == 'car':
#         count_stay_car += 1


# prob_stay = count_stay_car/num_of_sims

# print(f"Prob. of selecting a car with STAY strategy = {prob_stay:.4f}")

# # SWITCH STRATEGY COUNT  ## Can combine both strategies into single loop

# count_switch_car = 0

# for j in range(num_of_sims):
#     pick = np.random.randint(0,3)   # first pick

#     remaining_doors = [i for i in range(3) if i!=pick and items_behind_door[i]!='car']    # Remaining doors is the one 'not pick' and 'not with car'. If the contestant selects 1 as pick and 0 is the car, remainin doors only contains 2. Alos, if the contestant picks 0 as pick and 0 as the car, then remaining doors will be [1,2]
#     host_opens = np.random.choice(remaining_doors)      

#     switch_pick = [i for i in range(3) if i !=pick and i!= host_opens]   #Switch pick will be the option which is not pick and not host opening one

#     if items_behind_door[switch_pick[0]] == 'car':
#         count_switch_car +=1
    
# prob_switch = count_switch_car/num_of_sims

# print(f"Prob. of selecting a car with SWITCH strategy = {prob_switch:.4f}")

    




######## 3: PROBABILITY DISTRIBUTIONS ####################################W


#### PROBLEM 3F

# Bernoulli Trials:

# Library: scipy.stats.bernoulli

# Task: Simulate a biased coin flip (e.g., p(Heads) = 0.7).
# Use:
#    bernoulli.rvs(p=0.7, size=100) to generate 100 random samples.
#    bernoulli.pmf(k, p=0.7) to find the probability of getting outcome k (0 or 1).


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import bernoulli

# p = 0.7
# num_trials = 100

# samples = bernoulli.rvs(p, size=num_trials)      # Creates a random value between 0 and 1, if that value is <p, assign 0, >=p, assign 1.
# # count = 0
# # for i in range(len(samples)):
# #     if samples[i] == 1:
# #         count += 1

# #OR

# count = np.sum(samples)


# p_heads = bernoulli.pmf(1, p)     ## calculates the PMF of a bernoulli distribution of a given outcome '1'
# p_tails = bernoulli.pmf(0, p) 
# print(f"P(Heads) = {p_heads}")
# print(f"P(Tails) = {p_tails}")

# unique, counts = np.unique(samples, return_counts=True)     # Find the unique values and their count in an array

# plt.bar(['Tails(0)', 'Heads(1)'], counts, color = ['red', 'blue'])
# plt.xlabel("Outcome")
# plt.ylabel("Frequency")
# plt.title("Simulated Biased Coin Flips (p=0.7 for Heads)")
# plt.show()





##### PROBLEM 3G

# Binomial Distribution:

# Library: scipy.stats.binom

# Scenario: You flip the biased coin (p=0.7) 10 times (n=10).

# Use:
# binom.pmf(k, n=10, p=0.7) to find the probability of getting exactly k heads (e.g., k=0, 1, ..., 10). Plot the PMF using matplotlib.
# binom.cdf(k, n=10, p=0.7) to find the probability of getting at most k heads (P(X <= k)). Calculate P(X <= 5).
# binom.rvs(n=10, p=0.7, size=1000) to generate 1000 samples (each sample is the number of heads in 10 flips). Plot a histogram of these samples and compare its shape to the PMF plot.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom

# p = 0.7
# n = 10

# k_values = np.arange(0, n+1)     # creates an array [0,1,2,3,4...10] representing all possible values of heads in 10 flips
# pmf_values = binom.pmf(k_values, n, p)   ## calculaties p(x=k) for each value of k in binomial distribution

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)

# plt.bar(k_values, pmf_values, color = 'blue', alpha=0.7)
# plt.xlabel("Number of Heads (k)")
# plt.ylabel("P(x = k)")
# plt.title("Binomial PMF (n=10, p=0.7)")
# plt.xticks(k_values)

# k_cdf = 5 # to find P(x<=5)
# prob_at_most_5 = binom.cdf(k_cdf, n, p)
# print(f"P(X <= {k_cdf}) = {prob_at_most_5:.4f}")

# samples = binom.rvs(n, p, size = 1000)     ## 1000 times repeating the experiment. each experiment consists of slipping the coin 10 times. samples contain number of heads 

# plt.subplot(1,2,2)
# plt.hist(samples, bins=np.arange(-0.5, n+1.5, 1), density=True, alpha = 0.7, color='red', edgecolor = 'black')
# plt.xlabel("Number of Heads in 10 Flips")
# plt.ylabel("Frequency")
# plt.title("Histogram of Simulated Samples")

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()





##### PROBLEM 3H

# Normal Distribution:

# Library: scipy.stats.norm

# Scenario: Assume heights are normally distributed with mean=170cm and std dev=10cm (loc=170, scale=10).

# Use:
# norm.pdf(x, loc=170, scale=10) to get the value of the probability density function at height x. Plot the PDF over a range of heights (e.g., 140 to 200).
# norm.cdf(x, loc=170, scale=10) to find P(Height <= x). Calculate the probability of a person being shorter than 160cm (P(X <= 160)).
# Calculate the probability of a person being between 165cm and 180cm (P(165 <= X <= 180) = P(X <= 180) - P(X <= 165)).
# norm.ppf(q, loc=170, scale=10) (Percent Point Function - inverse of CDF) to find the height below which q proportion of the population lies. 
# Find the height corresponding to the 90th percentile (q=0.9).
# norm.rvs(loc=170, scale=10, size=1000) to generate 1000 random height samples. Plot a histogram.


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# mean_height = 170
# std_dev = 10

# x_values = np.linspace(140,200, 1000)

# pdf_values = norm.pdf(x_values, loc = mean_height, scale = std_dev)   # Compute the pdf values for x_values

# # plt.plot(x_values, pdf_values, color="blue", label="Normal Distribution (Mean=170, Std=10)")
# # plt.xlabel("Height (cm)")
# # plt.ylabel("Probability Density")
# # plt.title("Probability Density Function (PDF) of Heights")
# # plt.legend()
# # plt.show()

# prob_less_than_160 = norm.cdf(160, loc=mean_height, scale = std_dev)
# print(f"Probability of a person being shorter than 160cm: {prob_less_than_160:.4f}")

# prob_between_165_180 = norm.cdf(180, loc=mean_height, scale=std_dev) - norm.cdf(165, loc=mean_height, scale=10)
# print(f"Probability of a person being between 165cm and 180cm: {prob_between_165_180:.4f}")

# height_90th_percentile = norm.ppf(0.9, loc=mean_height, scale=std_dev)
# print(f"Height corresponding to the 90th percentile: {height_90th_percentile:.2f} cm")



# random_samples = norm.rvs(loc = mean_height, scale=std_dev, size = 1000)    # generates 1000 random samples following the normal distribution
# plt.hist(random_samples, bins=30, density=True, alpha=0.6, color='red', edgecolor='black')
# plt.plot(x_values, pdf_values, color="blue", label="Theoretical PDF")

# plt.xlabel("Height (cm)")
# plt.ylabel("Density")
# plt.title("Histogram of Simulated Heights vs Theoretical PDF")
# plt.legend()
# plt.show()






#### PROBLEM 3I

# Fit Normal Distribution to Data:

# Task: Generate 500 samples from a normal distribution with a known mean and standard deviation (e.g., mean=50, std=5) using norm.rvs.

# Use: scipy.stats.norm.fit(data) to estimate the mean (loc) and standard deviation (scale) from the generated data.
# Compare: Compare the estimated parameters to the true parameters (50 and 5) you used to generate the data.


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# true_mean = 50
# true_std_dev = 5

# random_samples = norm.rvs(loc = true_mean, scale=true_std_dev, size = 500)

# estimated_mean, estimated_std = norm.fit(random_samples)    #norm.fit fits the data into normal distribution and gives the mean and std as output

# print(f"True Mean: {true_mean}, Estimated Mean: {estimated_mean:.4f}")
# print(f"True Std Dev: {true_std_dev}, Estimated Std Dev: {estimated_std:.4f}")



# x_values = np.linspace(30, 70, 1000)

# # True and estimated PDFs
# true_pdf = norm.pdf(x_values, loc=true_mean, scale=true_std_dev)
# estimated_pdf = norm.pdf(x_values, loc=estimated_mean, scale=estimated_std)

# # Plot histogram of samples
# plt.hist(random_samples, bins=30, density=True, alpha=0.6, color='gray', edgecolor='black', label="Histogram of Samples")

# # Plot true PDF
# plt.plot(x_values, true_pdf, color="blue", label="True Normal Distribution (μ=50, σ=5)")

# # Plot estimated PDF
# plt.plot(x_values, estimated_pdf, 'r--', label=f"Estimated Distribution (μ={estimated_mean:.2f}, σ={estimated_std:.2f})")

# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.title("Fitting a Normal Distribution to Data")
# plt.legend()
# plt.show()




#### PROBLEM 3J

# Binomial Approaching Normal:

# Task: Use scipy.stats.binom and scipy.stats.norm.

# Steps:
# Choose a probability p (e.g., p=0.5).
# Plot the Binomial PMF for increasing values of n (e.g., n=10, n=50, n=200).
# On the same plots, overlay the PDF of a Normal distribution with mean μ = n*p and standard deviation σ = sqrt(n*p*(1-p)).
# Observe: As n gets larger, the shape of the Binomial PMF increasingly resembles the Normal PDF (illustrating the Central Limit Theorem).



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binom, norm

# # Probability of success
# p = 0.5

# # Different values of n
# n_values = [10, 50, 200]

# # Creating subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for i, n in enumerate(n_values):
#     # Compute binomial PMF values
#     k_values = np.arange(0, n + 1)
#     binom_pmf = binom.pmf(k_values, n, p)

#     # Compute normal approximation
#     mu = n * p  # Mean
#     sigma = np.sqrt(n * p * (1 - p))  # Standard deviation
#     x_values = np.linspace(0, n, 1000)
#     normal_pdf = norm.pdf(x_values, mu, sigma)

#     # Plot Binomial PMF
#     axes[i].bar(k_values, binom_pmf, color='blue', alpha=0.6, label=f'Binomial PMF (n={n})', edgecolor='black')

#     # Plot Normal Approximation
#     axes[i].plot(x_values, normal_pdf, 'r-', label=f'Normal PDF (μ={mu}, σ={sigma:.2f})')

#     # Labels and legend
#     axes[i].set_title(f'Binomial vs Normal Approximation (n={n})')
#     axes[i].set_xlabel('k (Number of Successes)')
#     axes[i].set_ylabel('Probability')
#     axes[i].legend()

# plt.tight_layout()
# plt.show()