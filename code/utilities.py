__author__ = 'kolosnjaji'

import numpy as np

import cov_functions as cf

def kernel(x,y, cov_fn=cf.sqexp, params=None):
    if (params==None):
        params =np.ones(np.size(x,1)+1)

    cov_mat = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(i, len(y)):
            cov_mat[i,j] = cov_fn(x[i,:], y[j,:], params)
            #print "Cov mat: {0}".format(cov_mat[i,j])
            cov_mat[j,i] = cov_mat[i,j]

    return cov_mat

def active_set_sigma(gp, x_set, num_points):
    (gp_mean, gp_cov) = gp.fit(x_set)
    #x_highest = np.argmax(np.diagonal(gp_cov))
    diag_cov = np.diagonal(gp_cov)
    sorted_indices = np.reverse(np.argsort(diag_cov))
    return diag_cov[sorted_indices[1:num_points]]





