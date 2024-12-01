
import numpy as np
import math
class MovReviewClassifier:
    def __init__(self):
        self.log_class_conditional_likelihoods = 0 
        self.log_class_priors = 0
        
    def estimate_log_class_priors(self, y_data):
        
        total = len(y_data) #total number of reviews
        #initialize counts for positive and negative reviews
        total0s = 0 
        total1s = 0
        for label in y_data: #iterate through labels to count the number of positive and negative reviews
            if label == 0:
                total0s += 1
            else:
                total1s +=1
        log_class_priors = np.array([math.log((total0s/total),10),math.log((total1s/total),10)]) #calculate the log priors for each class
        return log_class_priors

    def estimate_log_class_conditional_likelihoods(self, X, y, alpha=1.0):
        ### YOUR CODE HERE...
        numReviews, numFeatures = X.shape #get the number of reviews and features
        theta = np.zeros((2, numFeatures)) #initialize theta array to zeros
        for featurekey in range(numFeatures): #loop through features
            countpos = 0
            countneg = 0
            countkeypos = 0
            countkeyneg =0 
            for reviewkey in range(numReviews): #loop through reviews
                if y[reviewkey]== 1: #if the review is positive
                    countpos += 1 #increment count of pos reviews
                    countkeypos += X[reviewkey,featurekey] #Add current TFIDF score to countkeypos
                else:
                    countneg += 1 #if 0 increment count of neg reviews
                    countkeyneg += X[reviewkey,featurekey] #Add TFIDF score to countkeyneg

            theta[0][featurekey] = math.log((countkeyneg +alpha) / (countneg + alpha * numFeatures) ,10) #calculate the log likelihood for negative reviews
            theta[1][featurekey] = math.log((countkeypos +alpha) / (countpos + alpha * numFeatures ),10) #calculate the log likelihood for positive reviews


        return theta

    def train(self,X,y):
        self.log_class_priors = self.estimate_log_class_priors(y) 
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(X,y)
        
    def predict(self, X): #method to predict the class of a review (pos/neg)
        
        numReviews, numFeatures = X.shape #get the number of reviews and features
        class_predictions = np.zeros(numReviews) #initialize an array to store the predictions
        for reviewKey in range(numReviews): #loop through the reviews
            prob0= self.log_class_priors[0] #retrieve the log priors for negative reviews
            prob1 = self.log_class_priors[1] #retrieve the log priors for positive reviews
            for featureKey in range(numFeatures): #loop through the features
                prob0 += X[reviewKey,featureKey] * self.log_class_conditional_likelihoods[0][featureKey] #calculate the probability of the review being negative
                prob1 += X[reviewKey,featureKey] * self.log_class_conditional_likelihoods[1][featureKey] #calculate the probability of the review being positive
                if prob1 > prob0: #if the probability of the review being positive is greater than the probability of the review being negative
                    class_predictions[reviewKey] = 1 #set the review as positive
                else:
                    class_predictions[reviewKey] = 0 #set the review as negative


        return class_predictions #return the predictions
    

def create_classifier(X,y):
    classifier = MovReviewClassifier() #create a new instance of the classifier
    classifier.train(X,y) #train the classifier
    return classifier
