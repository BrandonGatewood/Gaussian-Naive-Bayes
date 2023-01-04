# A simple Gaussian-Naive-Bayes to classify the Spambase data from the UCI ML repository, which can be found here:  

https://archive.ics.uci.edu/ml/datasets/spambase 

The program splits the data into two sets, training and testing data. Each set contains 40% spam and 60% not spam to reflect the statistics of the full data set. Then the program will create a probalistic model with the training data, computing the prior probability for each class spam (1) and not spam (0). It will also compute the mean and standard deviation for each of the 57 features. If the standard deviation is 0, it will use laplace smoothing to avoid a divide by zero error. Once all that has been completed, it will run the Gaussian Naive Bayes algorithm to classify the instances in the testing data. 
