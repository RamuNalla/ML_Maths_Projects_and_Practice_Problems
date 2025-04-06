# WEEK 2 PROJECT: Spam Classifier (Naive Bayes) & Review

# Weekend Project Focus: Applying probability concepts (especially conditional probability and Bayes' Theorem) to build a simple text classifier from scratch.


# Project Steps (To be done over the weekend / Day 14):

# Dataset:

# Find: Search online for "SMS Spam Collection Dataset". A common one is available from the UCI Machine Learning Repository (often found as SMSSpamCollection file). It's a tab-separated file with "ham" (not spam) or "spam" labels.
# Load: Use Pandas pd.read_csv to load the data. You might need options like sep='\t' and header=None, then name the columns (e.g., 'label', 'message').

# Data Preprocessing (Simple):

# Lowercase: Convert all messages to lowercase (.str.lower()).
# Remove Punctuation: Remove punctuation (e.g., using regular expressions str.replace(r'[^\w\s]', '') or string manipulation).
# Split: Split messages into lists of individual words (.str.split()).
# (Optional): Consider removing common "stop words" (like "the", "a", "is") using a predefined list (e.g., from nltk library if installed, or a simple manual list).

# Train/Test Split:

# Library: sklearn.model_selection.train_test_split
# Task: Split your processed data (messages as lists of words, and labels) into training and testing sets.

# Build Vocabulary:

# Task: Create a set of all unique words that appear in the training messages. This is your vocabulary.

# Calculate Probabilities (Training):

# Goal: Calculate P(word | Spam) and P(word | Ham) for each word in the vocabulary, and the overall prior probabilities P(Spam) and P(Ham).
# Steps:
# Separate training messages into spam and ham messages.
# Calculate P(Spam) = (Number of spam messages in train) / (Total number of messages in train).
# Calculate P(Ham) = 1 - P(Spam).
# Count word occurrences: For each word in the vocabulary, count how many times it appears in all spam messages and how many times in all ham messages.
# Laplace Smoothing: To avoid zero probabilities for words not seen in a category, add 1 to each word count (numerator) and add the total number of unique words in the vocabulary (the size of your vocabulary set) to the total count of words in that category (denominator).
# Calculate P(word | Spam) = (Count(word in spam) + 1) / (Total words in spam + Vocabulary size).
# Calculate P(word | Ham) = (Count(word in ham) + 1) / (Total words in ham + Vocabulary size).
# Store: Store these probabilities (e.g., in dictionaries).

# Classifier Function (Prediction):

# Task: Create a function classify(message) that takes a new message (as a list of words).
# Steps:
# Calculate the score (log-probability is better to avoid underflow) for the message being Spam: log(P(Spam)) + Σ log(P(wordᵢ | Spam)) for each word wordᵢ in the message that is also in your vocabulary.
# Calculate the score for the message being Ham: log(P(Ham)) + Σ log(P(wordᵢ | Ham)).
# Compare: Return "spam" if the spam score is higher, otherwise return "ham". Ignore words in the message that are not in your training vocabulary.

# Evaluate on Test Set:

# Task: Run your classify function on each message in the test set.
# Compare: Compare your predicted labels with the true labels (y_test).
# Metrics: Calculate accuracy, precision, recall, and the confusion matrix using sklearn.metrics.
# Interpret: Analyze the results. Where does the classifier make mistakes?


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

data = pd.read_csv("C:\\Users\\mkongathi\\OneDrive - Microsoft\\Desktop\\Ramu\\Ppn\\sms+spam+collection\\SMSSpamCollection.csv", sep="\t", header = None, names = ["Label", "Message"])    ## Since, their is no header, read_csv consider first row as the header. Therefore header = "none" and assigned labels with names array
#print(data.head())

data["Message"] = data["Message"].str.lower()
data["Message"] = data["Message"].str.replace(r'[^\w\s]', '', regex=True)   ## replace all character which are not w-word, s-white space with ''-nothing
data["Message"] = data["Message"].str.split()                               ## Splits text into words  ##Tokenizes text into individual words

stop_words = {"the", "a", "is", "to", "and", "in", "for", "of", "on"}
data["Message"] = data["Message"].apply(lambda words: [word for word in words if word not in stop_words])   ## lambda function to remove the stop_words     

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

print(data.head())


## Building a vocabulary on the entire words present in data_train by removing duplicates

vocabulary = set(word for message in data_train["Message"] for word in message)    ## Loops throigh each message in data_train and loops through each word in each message
                                                                                    ## Set removes all the duplicates, Vocabulary now contains all unique words



#Calculating probabilities of Ham and Spam

num_spam = (data_train["Label"] == "spam").sum()
num_ham = (data_train["Label"] == "ham").sum()

total_messages = len(data_train)

P_spam = num_spam/total_messages
P_ham = num_ham/total_messages



# Counting occurrences fo each word seperately in spam and ham messages 

## Intialize Dictionaries to store word counts. Dictionary where keys are words and values are number of times each word appears
spam_word_counts = {}
ham_word_counts = {}  

spam_messages = data_train[data_train["Label"] == "spam"]["Message"]    #Filter rows where the label is "spam" and keepts its messages
ham_messages = data_train[data_train["Label"] == "ham"]["Message"]

for message in spam_messages:
    for word in message:
        spam_word_counts[word] = spam_word_counts.get(word, 0) + 1   ## get function gets the value for a particular key, if word is not present it gives 0, Therefore adding 1 when it gives 0

for message in ham_messages:
    for word in message:
        ham_word_counts[word] = ham_word_counts.get(word, 0) + 1 

#Apply laplace smoothing and computing probabilities

total_words_in_spam = sum(spam_word_counts.values())  ## Calculate total word in spam and ham messages
total_words_in_ham = sum(ham_word_counts.values())

vocab_size = len(vocabulary)   ## Vocab contains all words without duplication

P_word_given_spam = {}     ## Probabiltiies are stored in dictionaries
P_word_given_ham = {}

for word in vocabulary:
    P_word_given_spam[word] = (spam_word_counts.get(word, 0) + 1) / (total_words_in_spam + vocab_size)     # +1 is added to each word count to avoid zero probabilities.
    P_word_given_ham[word] = (ham_word_counts.get(word, 0) + 1) / (total_words_in_ham + vocab_size)


##Building a Naive Bayes Classifier 

def classify(message):

    spam_score = np.log(P_spam)
    ham_score = np.log(P_ham)

    words = [word.lower() for word in message]
    words = [word for word in words if word.isalnum()]          ## Preprocess the message by removing punctuation and keeping only alphanumeriv words

    for word in words:
        if word in vocabulary:
            spam_score += np.log(P_word_given_spam.get(word, 1/(total_words_in_spam+vocab_size)))
            ham_score += np.log(P_word_given_ham.get(word, 1/(total_words_in_ham+vocab_size)))

    return "spam" if spam_score > ham_score else "ham"

# Apply classifier to test set
y_pred = data_test["Message"].apply(classify)


# Evaluate performance
accuracy = accuracy_score(data_test["Label"], y_pred)
precision = precision_score(data_test["Label"], y_pred, pos_label="spam")     # We need to label positive class when calculing precision and accuracy
recall = recall_score(data_test["Label"], y_pred, pos_label="spam")
conf_matrix = confusion_matrix(data_test["Label"], y_pred)

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)
