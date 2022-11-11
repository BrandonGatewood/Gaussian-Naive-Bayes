# Brandon Gatewood
# CS 445 Program 2
#
# This program uses Gaussian Naive Bayes classifier to classify the Spambase data from the UCI ML repository.

import csv
import numpy as np
from sklearn.metrics import confusion_matrix

np.seterr(divide='ignore')

# --- Step 1 ---
# Load data
f = open('spambase.data', 'r')
read = csv.reader(f)
data = np.array(list(read))

# Split data into spam and not spam
spam = np.array(data[:1813])
not_spam = np.array(data[1813:])

# Create training data and testing data 40% spam and 60% not spam
train_data = np.concatenate((spam[:906], not_spam[:1394]), axis=0)
test_data = np.concatenate((spam[906:], not_spam[1394:]), axis=0)

# Save target from test data
test_shape = test_data.shape
target = np.zeros(test_shape[0])
for i in range(test_shape[0]):
    target[i] = test_data[i][test_shape[1] - 1]

# --- Step 2 ---
# Compute prior probability for each class, 1 (spam), 0 (not spam) in the training data
num_of_spam = 906
num_of_not_spam = 2300 - num_of_spam
p_one = 906 / 2300
p_zero = num_of_not_spam / 2300

# Calculate the imperial estimates of all the parameters, mean and standard deviation, of all the Gaussian class
# conditional distribution
# In the training data, first 906 elements are positive and the rest are negative.
pos_transpose = train_data[:906].T.astype(float)
neg_transpose = train_data[906:].T.astype(float)

pos_rows, pos_cols = pos_transpose.shape
neg_rows, neg_cols = neg_transpose.shape
pos_rows = pos_rows - 1
neg_rows = neg_rows - 1

pos_mean = np.zeros(pos_rows)
pos_std = np.zeros(pos_rows)
neg_mean = np.zeros(neg_rows)
neg_std = np.zeros(neg_rows)

# Count number of distinct values each feature can have
pos_count = np.zeros(pos_rows)
neg_count = np.zeros(neg_rows)

for i in range(pos_rows):
    pos_count[i] = len(np.unique(pos_transpose))
for i in range(neg_rows):
    neg_count[i] = len(np.unique(neg_transpose))

# Calculate the positive and negative mean and standard deviation of the training set
for i in range(pos_rows):
    pos_mean[i] = np.mean(pos_transpose[i])
    pos_std[i] = np.std(pos_transpose[i])

    # Apply laplace smoothing
    if pos_std[i] == 0:
        pos_std[i] = (pos_std[i] + 1) / (num_of_spam + pos_count[i])

for i in range(neg_rows):
    neg_mean[i] = np.mean(neg_transpose[i])
    neg_std[i] = np.std(neg_transpose[i])

    # Apply laplace smoothing
    if neg_std[i] == 0:
        neg_std[i] = (neg_std[i] + 1) / (num_of_not_spam + neg_count[i])

# --- Step 3 ---
# Run Gaussian Naive Bayes on testing data
predicted = np.zeros(test_shape[0])

prob_pos = np.zeros(57)
prob_neg = np.zeros(57)

for i in range(test_shape[0]):
    for j in range(test_shape[1] - 1):
        x_j_pos = 1 / (np.sqrt(2 * np.pi) * pos_std[j])
        x_j_pos = x_j_pos * pow(np.e,
                                -(pow((test_data[i][j].astype(float) - pos_mean[j]), 2) / (2 * (pow(pos_std[j], 2)))))
        x_j_pos = np.log(x_j_pos)

        x_j_neg = 1 / (np.sqrt(2 * np.pi) * neg_std[j])
        x_j_neg = x_j_neg * pow(np.e,
                                -(pow((test_data[i][j].astype(float) - neg_mean[j]), 2) / (2 * (pow(neg_std[j], 2)))))
        x_j_neg = np.log(x_j_neg)

        # Probability of x_j given positive and negative
        prob_pos[j] = x_j_pos
        prob_neg[j] = x_j_neg

    positive = np.log(p_one) + np.sum(prob_pos)
    negative = np.log(p_zero) + np.sum(prob_neg)

    # Find the max probability of positive and negative class
    if positive > negative:
        predicted[i] = 1

# Compute Confusion matrix, accuracy, precision and recall
cfm = confusion_matrix(target, predicted)
tp = cfm[0][0]
fp = cfm[0][1]
fn = cfm[1][0]

accuracy = (target == predicted).sum() / len(target) * 100
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall) + "\n")
print(cfm)
