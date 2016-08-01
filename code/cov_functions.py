__author__ = 'kolosnjaji'

import numpy as np

def sqexp(x,y, params):
    my_lambda = np.diag((1.0/(params[1:]))/params[1:])
    my_diff = x-y
#    print np.count_nonzero(my_diff)
    exponential = params[0]*params[0]*np.exp( -0.5 * np.dot(np.dot(my_diff.T,my_lambda),my_diff)) + np.random.random(1)/10
    #print "x: {0} y: {1} Diff: {2}, exp: {3}, lambda: {4}, params: {5}, exp:{6}".format(x,y, my_diff, exponential, my_lambda, params[0],-0.5 * np.dot(np.dot(my_diff.T,my_lambda),my_diff))
    return exponential
