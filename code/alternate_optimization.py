__author__ = 'kolosnjaji'

###################################################################################################################
#
# Alternate optimization of a Gaussian Process for communication efficiency
# Parameter groups:
#   q: weight of each sample (n parameters)
#   theta: GP parameters: signal variance + characteristic length scale for each dimension
###################################################################################################################

import numpy as np
from numpy.linalg import inv, det
import utilities as ut
import math
import matplotlib.pyplot as plt
from scipy.optimize import check_grad

def gradient_kern_q(x, x_0, theta):
    n = np.size(x,0)
    my_lambda = np.diag(theta[1:])
    my_kern = ut.kernel(x,x, params=theta)# get the whole matrix
    gradients = []
    for i in range(n): # which q?
        gradient = np.zeros((n,n))
        for j in range(i,n):
            grad_part = np.dot(inv(my_lambda),(x[i,:] - x[j,:]))*my_kern[i,j]
            gradient[i,j] = -grad_part*x_0[i,:] + grad_part*x_0[j,:] # differentiating a multivariate function
            gradient[j,i] = gradient[i,j]
#        print gradient
#        raw_input()
        gradients.append(gradient) # n gradient matrices
    return gradients

def gradient_kern_theta(x, theta): # gradient by GP parameters
    n = np.size(x,0)
    my_kernel = ut.kernel(x, x, params=theta)
    my_lambda = np.diag(theta[1:])
    m = np.size(x,1)
    gradients = []
    gradient_f = np.zeros((n,n))
    for i in range(n): # grad_sigma
        for j in range(i,n):
            gradient_f[i,j]= 2*theta[0]*np.exp((-1/2)*np.dot(np.dot((x[i,:]-x[j,:]),inv(my_lambda)),(x[i,:]-x[j,:])))
            gradient_f[j,i]= gradient_f[i,j]
    gradients.append(gradient_f)

    for i in range(m): # for all params of lambda
        grad_lambda = np.zeros((n,n))
        for j in range(n):
            for k in range(j,n):
                my_diff = x[j,:]-x[k,:]
                grad_lambda[j,k] = -2*my_kernel[j,k]*my_diff[i]*my_diff[i]/(theta[i+1]*theta[i+1]*theta[i+1])
                grad_lambda[k,j] = grad_lambda[j,k]
        gradients.append(grad_lambda)

    return gradients

def gradient_q(theta, args): # gradient by q

    x = args[0]
    x_0 = args[1]
    y = args[2]


    print "computing gradient of q..."
    n = np.size(x)
    m=1
    grad = np.zeros((n,1)) # gradient for all q_i

    grads_kern = gradient_kern_q(x,x_0, theta)
    my_kernel = ut.kernel(x,x,params=theta)

    for i in range(n):
        grad[i] = (-1/2)*np.trace(np.dot(inv(my_kernel),grads_kern[i]))+(1/2)*np.dot(np.dot(np.dot(np.dot(y.T,inv(my_kernel)),grads_kern[i]),inv(my_kernel)),y)

    return grad

def gradient_theta(theta, args):
    print "computing gradient of theta..."
    x = args[0]
    y = args[1]

    n = np.size(x)
    m=1
    grads_kern = gradient_kern_theta(x,theta)
    grad = np.zeros(m+1) # gradient for all q_i

    my_kernel = ut.kernel(x,x,params=theta)
    for i in range(m+1):
        grad[i] = (-1/2)*np.trace(np.dot(inv(my_kernel),grads_kern[i]))+(1/2)*np.dot(np.dot(np.dot(np.dot(y.T,inv(my_kernel)),grads_kern[i]),inv(my_kernel)),y)

    return grad

def logprob(theta, args):

    x = args[0]
    y = args[1]
    n = np.size(x,0)

    my_kernel = ut.kernel(x,x, params=theta)

    logp_y = (-1/2)*np.log(det(my_kernel))-1/2*np.dot(np.dot(y.T,inv(my_kernel)),y) -(n/2) * np.log(2*math.pi)

    return logp_y


def main():

    mat = np.loadtxt(open("../data/bike_sharing/Bike-Sharing-Dataset/day.csv","rb"),delimiter=",",skiprows=1, usecols=(0,14))

    x_0 = np.expand_dims(mat[:,0],1)
    y = np.expand_dims(mat[:,1],1)

    #plt.plot(x_0,y)
    #print x_0
    #plt.show()
    #raw_input()

    q_init = np.ones((np.size(y,0),1))
    q_init-=0.5

    x = x_0*q_init


    # initialized q and x, now first optimize the gp hyperparams

    num_iterations_theta=1
    num_iterations_q=1
    num_iterations_all = 1000
    grad_theta_speed = 1e-9
    grad_q_speed = 1e-3
    t_barrier = 10

    n = np.size(x)
    m=1
    theta= np.ones(m+1) # m+1 parameters, signal variance + characteristic length scale for each dimension

    for i in range(1,m+1):
        theta[i]+=1
    theta[0] = 5

    q = q_init

    for i in range(num_iterations_all):

        for j in range(num_iterations_theta):
            my_kernel = ut.kernel(x,x, params=theta)
            np.savetxt("foo.csv", my_kernel, fmt='%.3f',)
            print "Init kernel: {0}".format(my_kernel)
            print "Rank: {0}, full rank: {1}".format(np.linalg.matrix_rank(my_kernel), np.size(x,0))
            print np.corrcoef(my_kernel)
            #print gradient_theta(x, y, theta)
            print "Grad check: {0}".format(check_grad(logprob, gradient_theta, theta, [x, y]))
            theta-=grad_theta_speed*gradient_theta(theta, [x, y])
            print theta
            print inv(my_kernel)
            print det(my_kernel)
            my_kernel = ut.kernel(x,x, params=theta)
            print "New kernel: {0}".format(my_kernel)
            print inv(my_kernel)
            print det(my_kernel)
            logp_y = logprob(theta, [x, y])
            print "Params: {0}, log_prob: {1}, prob {2} det_kernel {3}".format(theta,logp_y, np.exp(logp_y), det(my_kernel))

        # now optimize the parameter q
        #
        # for j in range(num_iterations_q):
        #     print np.shape(grad_q_speed)
        #     print np.shape(q)
        #     new_grad = gradient_q(x, x_0, y, theta) + (1/t_barrier)*np.sum(1/(q*np.log(10))) # logarithmic barrier method
        #     print q.T
        #     q-=(grad_q_speed*new_grad)
        #
        #     x = x_0*q

if __name__ == "__main__":

    main()





