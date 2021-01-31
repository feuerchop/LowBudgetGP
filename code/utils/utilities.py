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

def getBoxbyX(X, grid=50, padding=True):
    '''
    Get the meshgrid X,Y given data set X
    :param X: dataset
    :param grid: meshgrid step size
    :param padding: if add extra padding around
    :return: X,Y
    '''
    if X.shape[1] > 2:
        print 'We can only get the grid in 2-d dimension!'
        return None
    else:
        minx = min(X[:,0])
        maxx = max(X[:,0])
        miny = min(X[:,1])
        maxy = max(X[:,1])
        padding_x = 0.05*(maxx-minx)
        padding_y = 0.05*(maxy-miny)
        if padding:
            X,Y = np.meshgrid(np.linspace(minx-padding_x, maxx+padding_x, grid),
                              np.linspace(miny-padding_y, maxy+padding_y, grid))
        else:
            X,Y = np.meshgrid(np.linspace(minx, maxx, grid),
                              np.linspace(miny, maxy, grid))
    return (X, Y)

def setAxSquare(ax):
    xlim0, xlim1 = ax.get_xlim()
    ylim0, ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1-xlim0)/(ylim1-ylim0))
