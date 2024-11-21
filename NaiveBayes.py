
import numpy as np
import math
class MovReviewClassifier:
    def __init__(self):
        self.log_class_conditional_likelihoods = 0
        self.log_class_priors = 0
        
    def estimate_log_class_priors(self, y_data):
        ### YOUR CODE HERE...
        total = len(y_data)
        total0s = 0
        total1s = 0
        for label in y_data:
            if label == 0:
                total0s += 1
            else:
                total1s +=1
        log_class_priors = np.array([math.log((total0s/total),10),math.log((total1s/total),10)])

        return log_class_priors

    def estimate_log_class_conditional_likelihoods(self, X, y, alpha=1.0):
        ### YOUR CODE HERE...
        numReviews, numFeatures = X.shape
        theta = np.zeros((2, numFeatures))
        for featurekey in range(numFeatures):
            countpos = 0
            countneg = 0
            countkeypos = 0
            countkeyneg =0
            for reviewkey in range(numReviews):
                if y[reviewkey]== 1:
                    countpos += 1 #increment count of pos reviews
                    countkeypos += X[reviewkey,featurekey] #Add current TFIDF score to countkeypos
                else:
                    countneg += 1
                    countkeyneg += X[reviewkey,featurekey]

            theta[0][featurekey] = math.log((countkeyneg +alpha) / (countneg + alpha * numFeatures) ,10)
            theta[1][featurekey] = math.log((countkeypos +alpha) / (countpos + alpha * numFeatures ),10)


        return theta

    def train(self,X,y):
        self.log_class_priors = self.estimate_log_class_priors(y)
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(X,y)
        
    def predict(self, X):
        
        numReviews, numFeatures = X.shape
        class_predictions = np.zeros(numReviews)
        for reviewKey in range(numReviews):
            prob0= self.log_class_priors[0]
            prob1 = self.log_class_priors[1]
            for featureKey in range(numFeatures):
                prob0 += X[reviewKey,featureKey] * self.log_class_conditional_likelihoods[0][featureKey]
                prob1 += X[reviewKey,featureKey] * self.log_class_conditional_likelihoods[1][featureKey]
                if prob1 > prob0:
                    class_predictions[reviewKey] = 1
                else:
                    class_predictions[reviewKey] = 0


        return class_predictions
    

def create_classifier(X,y):
    classifier = MovReviewClassifier()
    classifier.train(X,y)
    return classifier
