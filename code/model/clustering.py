__author__ = 'kolosnjaji'

# code from: Bauckhage, Christian. Numpy/scipy Recipes for Data Science: k-Medoids Clustering. Technical Report, University of Bonn, 2015.

import numpy as np
from numpy.linalg import norm

def distance_matrix(X):
    n,m = X.shape
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            D[i,j] = norm(X[i,:]-X[j,:])
            D[j,i] = norm(X[i,:]-X[j,:])
    return D

def kMedoids(X, k, tmax=1000):
    # determine dimensions of distance matrix D
    print "Computing distances..."
    D = distance_matrix(X)
    print "Mean distance: {0} {1}".format(np.mean(D), np.median(D))
    print "Mean X: {0} {1}".format(np.mean(X, 0), np.std(X,0))

    m, n = D.shape

    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k, replace=False))

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}

    for t in xrange(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
           # print kappa, J
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)

     # check for convergence
        #if np.array_equal(M, Mnew):
        #    break

        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

     # return results
    print M
    print C
    return M, C