__author__ = 'kolosnjaji'


import numpy as np

from scipy.stats import mode

import clustering
from sklearn.linear_model import LogisticRegression# return weights of m (k?) classifiers

class LowBudgetCrowdsourcing:

    def __init__(self, top_k=10):
        self.top_k = top_k


    def train(self, X, Y):
        num_clients = len(X)
        self.train_X = X
        self.train_Y = Y
        self.num_clients = num_clients

        chosen_clients = self.sub_clustering(X,Y, self.top_k)
        self.chosen_clients = chosen_clients


# return predicted Y
    def predict(self, X):
        n,m = X.shape
        predictions = []
#        predictions = self.linreg_pred(self, X, self.chosen_clients)
        predictions = self.logreg_pred(X, self.chosen_clients)
        predictions_all = np.dstack(predictions)

        predictions_majvote = mode(predictions_all, axis=2)

        return predictions_majvote

    def predict_for_all(self, X):
        (n,m) = X.shape
        predictions = []
#        predictions = self.linreg_pred(X, range(self.num_clients))
        predictions = self.logreg_pred(X, range(self.num_clients))
        predictions_all = np.dstack(predictions)

        predictions_majvote = mode(predictions_all, axis=2)

        return predictions_majvote

# linear regression
    def linreg_train(self, X,Y):
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

    def linreg_pred(self, X, chosen_clients):
        predictions = []
        for client in chosen_clients:
            predictions.append(np.dot(X, self.linreg_params[client,:] ))
        return predictions

    def logreg_train(self, X,Y):
        logreg = LogisticRegression()
        return (logreg.fit(X,Y).coef_, logreg)


    def logreg_pred(self, X, chosen_clients):

        predictions = []
        for client in chosen_clients:
            predictions.append(self.models[client].predict(X))
        return predictions

# return medians, also detect outliers
    def kmedoids(self, params, top_k): # left to implement...
        return clustering.kMedoids(params, top_k)

# subroutine to cluster functionals
    def sub_clustering(self, X,Y,  top_k, classifier='linear'):
        num_clients = len(X)
        fitted_params = np.zeros((num_clients, np.size(X[0],1)))
        print "Regression for {0} clients...".format(num_clients)
        self.models = []
        for i in range(num_clients):

            params, model = self.logreg_train(X[i], Y[i])
            fitted_params[i,:] = np.squeeze(params)
            self.models.append(model)
        self.linreg_params = fitted_params
        print "Computing medoids..."
        medians = self.kmedoids(fitted_params, top_k)
        print "Medians computed."

        chosen_indices  = medians[0]

        return chosen_indices

