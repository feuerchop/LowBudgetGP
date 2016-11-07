__author__ = 'kolosnjaji'
import numpy as np

class GenGradDescModelNoAnnotations:

    def __init__(self, w_init, v_init):
        self.w = w_init
        self.v = v_init
        self.num_clients = self.w.shape[0]
        self.PARAM_LAMBDA = 0.0001

    def Slambda(self,x, param_lambda):
        res_x = np.zeros(x.shape[0])
        res_size = x.shape[0]
        for i in range(res_size):
            if (x[i]>param_lambda):
                res_x[i]=x[i]-param_lambda
            elif (x[i]<(-param_lambda)):
                res_x[i]=x[i]+param_lambda
            else:
                res_x[i]=0
        return res_x

    def grad_update(self,x, my_grad, TIMESTEP=0.0000001): # for each vector
        return self.Slambda(x-TIMESTEP*my_grad,self.PARAM_LAMBDA)

    def optimization(self,x,y_target, NUM_IT=1000, NUM_IT_P=10, PARAM_LAMBDA_W = 0.0001):
        y = self.log_reg(x, self.w)
        loss = self.cross_entropy_all(x, y, y_target)
        print "Initial Loss: {0}".format(loss)
        print "Optimization..."
        for i in range(NUM_IT):
            #print "Optimizing w..."
            for j in range(NUM_IT_P):
                # print j
                grad_w = self.grad_ce_w(x,y_target,self.w,self.v)
                self.w = self.w - grad_w*PARAM_LAMBDA_W;
            #print "Optimizing v..."
            for j in range(NUM_IT_P):
                # print j
                grad_v = self.grad_ce_v(x,y_target,self.w,self.v)
                self.v = self.grad_update(self.v,grad_v)
                self.v = self.v/np.sum(self.v) # keep sum to 1
            y = self.log_reg(x,self.w)
            loss = self.cross_entropy_all(x,y,y_target)
            errors = self.errors_count(y,y_target)
            print "Iteration {0} Loss: {1} Errors:{2}".format(i, loss, errors)

    def cross_entropy_all(self,x,y,y_target):
        M,N = y.shape # m-clients, n-data points;
        N,D = x.shape # n-data points, d-dimensionality of d
        ce_n = 0
        print self.v
        for n in range(N):
            sum_cl = np.dot(self.v,y[:,n])
            ce_n += y_target[n]*np.log(sum_cl) + (1-y_target[n])*np.log(1-sum_cl)
        ce_n = -ce_n/N
        return ce_n + self.PARAM_LAMBDA * np.linalg.norm(self.v, 1)

    def log_reg(self,x,w):
        num_data = x.shape[0]
        num_clients = w.shape[0]
        y = np.zeros((num_clients, num_data))
        for c in range(num_clients):
            for i in range(num_data):
                y[c,i] = 1/(1+np.exp(-np.dot(w[c,:],x[i,:])))
        return y

    def grad_log_reg(self, x, w): # gradient of logistic regression
        grad_list = [] # gradients for every client
        num_data = x.shape[0]
        num_params = x.shape[1]
        num_clients = 21
        for c in range(num_clients):
            grad_matrix = np.zeros((num_data, num_params))
            for i in range(num_data):
                grad_matrix[i,:] = np.dot(x[i, :],np.exp(np.dot(w[c,:], x[i,:])))/np.power((np.exp(np.dot(w[c,:],x[i,:]))+1),2)
            grad_list.append(grad_matrix)
        return grad_list # return list of gradients for every client

    def grad_ce_w(self,x, y_target, w,v,): # return vector size(w)
        y_gradients = self.grad_log_reg(x,w) # return list of matrices size(n,w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        num_params = w.shape[1]
        sum_grad = np.zeros((num_clients, num_params))
        y = self.log_reg(x, w)
        for c in range(num_clients):
            for n in range(num_data):
                sum_cl = np.dot(v,y[:,n])
            sum_grad[c,:] += y_target[n]*(1/sum_cl)*v[c]*y_gradients[c][n,:] + (1-y_target[n])*(1/(1-sum_cl))*(-(v[c]* y_gradients[c][n,:]))
        return -sum_grad/num_data

    def grad_ce_v(self,x,y_target,w,v): # return vector, size(v) = number of clients

        y = self.log_reg(x,w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        sum_grad = np.zeros(num_clients) #
       # print self.v
        for n in range(num_data):
            sum_cl = np.dot(v, y[:, n])
            sum_grad += y_target[n]*(1/sum_cl)*np.log(y[:,n]) + (1-y_target[n]) * (1/(1-sum_cl))*np.log(1-y[:,n])

        return sum_grad

    def errors_count(self,y,y_target):
        N = y_target.shape[0]
        y_labels = np.zeros((N,1))
        for n in range(N):
            label_float = np.dot(self.v,y[:,n])
            if label_float>0.5:
                y_labels[n] = 1
        return np.sum(y_labels!=y_target)

    def test(self, Y):
        pass





