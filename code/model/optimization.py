__author__ = 'kolosnjaji'
import numpy as np

class GenGradDescModel:

    def __init__(self, x,y_target, w_init, v_init):
        self.x = x
        self.y_target = y_target
        self.w = w_init
        self.v = v_init

    def Slambda(self,x, param_lambda):
        res_x = np.zeros(x.shape[0])
        res_size = x.shape[0]
        for i in range(res_size):
            if (res_x[i]>param_lambda):
                res_x[i]=x[i]-param_lambda
            elif (res_x[i]<(-param_lambda)):
                res_x[i]=x[i]+param_lambda
            else:
                pass

    def grad_update(self,x, my_grad, TIMESTEP=0.001, PARAM_LAMBDA=0.01): # for each vector
        return x+self.Slambda(x-TIMESTEP*my_grad,PARAM_LAMBDA)

    def grad_compute(self):

        pass


    def cross_entropy_all(self,x,y,y_target,v,w,param_lambda ):
        M,N = y.shape # m-clients, n-data points;
        N,D = x.shape # n-data points, d-dimensionality of d
        ce_n = 0
        for n in range(N):
            sum_cl = np.sum(v*y[n,:])
            ce_n += y_target[n]*np.log(sum_cl) + (1-y_target[n])*np.log(1-sum_cl)
        ce_n = -ce_n/N
        return ce_n + param_lambda * np.linalg.norm(v, 1)

    def log_reg(self,x,w):
        num_data = x.shape[0]
        y = np.zeros(num_data)
        for i in range(num_data):
            y[i] = 1/(1+np.exp(-np.dot(w,x[i,:])))
        return y

    def grad_log_reg(self,x,w):
        num_data = x.shape[0]
        grad_y = np.zeros(num_data)

        for i in range(num_data):
            grad_y[i] = x[i,:] * x[i,:]*np.exp(np.dot(w,x))/np.pow((np.exp(np.dot(w,x))+1),2)

        return grad_y

    def grad_ce_w(self,x, y_target, w,v,):
        y_gradients = self.grad_log_reg(x,w)











