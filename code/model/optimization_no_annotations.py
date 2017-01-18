__author__ = 'kolosnjaji'
import numpy as np
import warnings

class GenGradDescModelNoAnnotations:

    def __init__(self, w_init, v_init):
        self.w = w_init
        self.v = v_init
        self.num_clients = self.w.shape[0]
        #self.PARAM_LAMBDA = 0.0001

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

    def grad_update(self,x, my_grad, TIMESTEP): # for each vector # optimization of v
        return self.Slambda(x-TIMESTEP*my_grad,self.PARAM_LAMBDA)

    def optimization(self,x,y_target, y_client_annotation_indices, y_client_annotations, NUM_IT, NUM_IT_P, PARAM_LAMBDA_W, PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA, TIMESTEP):
        self.PARAM_LAMBDA = PARAM_LAMBDA
        y = self.log_reg(x, self.w)
        loss = self.cross_entropy_all(x, y, y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        print "Initial Loss: {0}".format(loss)
        print "Optimization..."
        for i in range(NUM_IT):
            #print "Optimizing w..."
            for j in range(NUM_IT_P):
                # print j
                grad_w = self.grad_ce_w(x,y_target, y_client_annotations, y_client_annotation_indices, self.w,self.v, PARAM_LAMBDA_ANNOTATIONS)
                #print "Grad W: pos: {0} neg: {1} zero: {2}".format(np.sum(grad_w>0), np.sum(grad_w<0), np.sum(grad_w==0))
                self.w = self.w - grad_w*PARAM_LAMBDA_W;
                #print "W:\n" + str(self.w)
            #print "Optimizing v..."
            for j in range(NUM_IT_P):
                #print "V:\n{0}".format(self.v * 100)
                # print j
                grad_v = self.grad_ce_v(x,y_target,self.w,self.v)
                self.v = self.grad_update(self.v,grad_v, TIMESTEP=TIMESTEP)
                self.v[self.v<0] = 0
                self.v = self.v/np.sum(self.v) # keep sum to 1
            y = self.log_reg(x,self.w)

            #print "W: {0}".format(self.w)
            #print np.where(self.v>0)
            print "V not 0:{0}".format(len(np.where(self.v>0)[0]))
           # print "Y:\n{0}".format(y)
            loss = self.cross_entropy_all(x,y,y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
            errors = self.errors_count(y,y_target)
            print "Iteration {0} Loss: {1} Error Percentage:{2}%".format(i, loss, errors)

    def cross_entropy_all(self,x,y,y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS):
        M,N = y.shape # m-clients, n-data points;
        N,D = x.shape # n-data points, d-dimensionality of d
        ce_n = 0
        sum_cl = np.dot(y.T, self.v)



        for n in range(N):
            if (sum_cl[n]>=1 or sum_cl[n]<=0):
                print "Warning!"
            ce_n += y_target[n]*np.log(sum_cl[n]) + (1-y_target[n])*np.log(1-sum_cl[n])
        ce_n = -ce_n/N

        #print "Y: {0}".format(sum_cl.T)
        #### we have annotation errors in additon
        sum_annot_error = 0
        num_points = 0
        for c in range(len(y_client_annotations)):
            for l in range(len(y_client_annotations[c])):
                sum_annot_error += np.power(y_client_annotations[c][l] - y[c,y_client_annotation_indices[c][l]],2)
                num_points+=1



        return ce_n + self.PARAM_LAMBDA * np.linalg.norm(self.v, 1) + sum_annot_error*PARAM_LAMBDA_ANNOTATIONS/float(num_points)

    def log_reg(self,x,w):
        num_data = x.shape[0]
        num_clients = w.shape[0]
        y = np.zeros((num_clients, num_data))

        for c in range(num_clients):
            for i in range(num_data):
                y[c,i] = 1/(1+np.exp(-np.dot(w[c,:],x[i,:])-0.5))
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

    def grad_ce_w(self,x, y_target, y_client_annotations, y_client_annotation_indices, w,v,PARAM_LAMBDA_ANNOTATIONS): # return vector size(w)
        y_gradients = self.grad_log_reg(x,w) # return list of matrices size(n,w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        num_params = w.shape[1]
        sum_grad = np.zeros((num_clients, num_params))
        sum_grad_annotations = np.zeros((num_clients, num_params))
        num_points = 0
        y = self.log_reg(x, w)
        for c in range(num_clients):
            for n in range(num_data):
                sum_cl = np.dot(v,y[:,n])
            sum_grad[c,:] += y_target[n]*(1/sum_cl)*v[c]*y_gradients[c][n,:] + (1-y_target[n])*(1/(1-sum_cl))*(-(v[c]* y_gradients[c][n,:]))
            cl_indices_list = y_client_annotation_indices[c]
            for i,ind in enumerate(cl_indices_list):
                sum_grad_annotations[c,:] += 2*(y[c,ind]-y_client_annotations[c][i]) * y_gradients[c][ind,:] # additional gradient
                num_points+=1
        sum_grad = -(sum_grad/num_data) + sum_grad_annotations*PARAM_LAMBDA_ANNOTATIONS/float(num_points)
        return sum_grad

    def grad_ce_v(self,x,y_target,w,v): # return vector, size(v) = number of clients

        y = self.log_reg(x,w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        sum_grad = np.zeros(num_clients) #
        for n in range(num_data):
            sum_cl = np.dot(v, y[:, n])
            #sum_grad += y_target[n]*(1/sum_cl)*np.log(y[:,n]) + (1-y_target[n]) * (1/(1-sum_cl))*np.log(1-y[:,n])
            sum_grad += y_target[n]*(1/sum_cl)*y[:,n] + (1-y_target[n]) * (1/(1-sum_cl))*(-y[:,n])
            #print "V:" + str(v)
            #print "Y:" + str(y)

        return -sum_grad

    def errors_count(self,y,y_target):
        N = y_target.shape[0]
        y_labels = np.zeros((N,1))
        for n in range(N):
            label_float = np.dot(self.v,y[:,n])

            if label_float>0.5:
                y_labels[n] = 1

        return np.sum(y_labels!=y_target)*100/float(len(y_labels))

    def test(self, test_X, test_Y, test_annotation_indices, test_annotations, PARAM_LAMBDA_ANNOTATIONS):
        data_num, data_dim = test_X.shape

        client_response = self.log_reg(test_X, self.w) # clients x num_data
        error_count = self.errors_count(client_response, test_Y)
        print "TESTING ACCURACY: {0}%".format((data_num-error_count)*100/data_num)
        loss = self.cross_entropy_all(test_X, client_response, test_Y, test_annotations, test_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        return loss







