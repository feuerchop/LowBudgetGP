__author__ = 'kolosnjaji'
import numpy as np
import warnings
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

class GenGradDescModelNoAnnotations:

    def __init__(self, w_init, v_init, multiclass=False, num_classes=10):
        self.w = w_init
        self.v = v_init
        self.num_clients = self.w.shape[0]
        self.multiclass = multiclass
        self.num_classes = num_classes
        #self.PARAM_LAMBDA = 0.0001

    @staticmethod
    def sigmoid_array(x):

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def slambda(x, param_lambda):
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

    def pretrain(self, x, y_target, num_iterations=10):
        num_dimensions = x.shape[1]
        # self.mlp_pretrained = []
        # num_clients = len(y_client_annotations)
        # for client in range(num_clients):
        #     mlp_clf = MLPClassifier(hidden_layer_sizes=(num_dimensions,), activation='sigmoid', solver='adam')
        #     x_client = x[y_client_annotation_indices, :]
        #     y_client = y_client_annotations
        #     mlp_clf.fit(x_client,y_client)
        #     self.mlp_pretrained.append(mlp_clf)
        self.mlp_pretrained = MLPClassifier(hidden_layer_sizes=(num_dimensions), activation='logistic', solver='adam', max_iter=num_iterations)
        self.mlp_pretrained.fit(x,np.ravel(y_target))
        print self.mlp_pretrained.coefs_[0].shape # 4096x4096
        print self.mlp_pretrained.coefs_[1].shape # 4096x1 - logistic regression
        print len(self.mlp_pretrained.coefs_)
        print num_dimensions



    def grad_update(self,x, my_grad, TIMESTEP): # for each vector # optimization of v
        return self.slambda(x-TIMESTEP*my_grad,self.PARAM_LAMBDA)

    def optimization(self,x,y_target, y_client_annotation_indices, y_client_annotations, NUM_IT, NUM_IT_P, PARAM_LAMBDA_W, PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA, TIMESTEP, method='LOGREG'):

        train_loss = []
        v_nonzero = []
        error_percentage = []

        self.PARAM_LAMBDA = PARAM_LAMBDA
        num_clients = len(y_client_annotation_indices)
        if (method=="MLP"):
            self.pretrain(x,y_target)
            self.x_intermediate = self.sigmoid_array(np.dot(x, self.mlp_pretrained.coefs_[0]))
            self.w = np.tile(self.mlp_pretrained.coefs_[1].T,(num_clients,1))
            y = self.log_reg(self.x_intermediate, self.w)


        elif (method=="LOGREG"):
            y = self.log_reg(x, self.w)

        else:
            print "Method not recognized!"
            exit(-1)

        loss = self.cross_entropy_all(x, y, y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        print "Initial Loss: {0}".format(loss)
        print "Optimization..."
        for i in range(NUM_IT):
            #print "Optimizing w..."
            for j in range(NUM_IT_P):
                # print j
                if (method=="MLP"):
                    grad_w = self.grad_ce_w(self.x_intermediate, y_target, y_client_annotations, y_client_annotation_indices, self.w, self.v, PARAM_LAMBDA_ANNOTATIONS)
                else:
                    grad_w = self.grad_ce_w(x,y_target, y_client_annotations, y_client_annotation_indices, self.w,self.v, PARAM_LAMBDA_ANNOTATIONS)
                #print "Grad W: pos: {0} neg: {1} zero: {2}".format(np.sum(grad_w>0), np.sum(grad_w<0), np.sum(grad_w==0))
                self.w = self.w - grad_w*PARAM_LAMBDA_W;
                #print "W:\n" + str(self.w)
            #print "Optimizing v..."
            for j in range(NUM_IT_P):
                #print "V:\n{0}".format(self.v * 100)
                # print j
                if (method=='MLP'):
                    grad_v = self.grad_ce_v(self.x_intermediate, y_target, self.w, self.v)
                else:
                    grad_v = self.grad_ce_v(x,y_target,self.w,self.v)
                #print self.v
                self.v = self.grad_update(self.v,grad_v, TIMESTEP=TIMESTEP)
                #self.v = self.v - grad_v*TIMESTEP

                #print self.v
                self.v[self.v<0] = 0
                self.v = self.v/np.sum(self.v) # keep sum to 1

            if (method=="MLP"):
                y = self.log_reg(self.x_intermediate,self.w)
            else:
                y = self.log_reg(x, self.w)

            #print "W: {0}".format(self.w)
            #print np.where(self.v>0)
            #print self.v
            print "V not 0:{0}".format(len(np.where(self.v>0)[0]))
           # print "Y:\n{0}".format(y)
            loss = self.cross_entropy_all(x,y,y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
            errors = self.errors_count(y,y_target)
            print "Iteration {0} Loss: {1} Error Percentage:{2}%".format(i, loss, errors)
            train_loss.append(loss)
            error_percentage.append(errors)
            v_nonzero.append(len(np.where(self.v>0)[0]))
        return (train_loss, v_nonzero, error_percentage)

    def cross_entropy_all(self,x,y,y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS):

        N,D = x.shape # n-data points, d-dimensionality of d
        ce_n = 0



        if (self.multiclass):
            M, N, L = y.shape  # m-clients, n-data points, labels;
            #print "{0} {1} {2}".format(M,N,L)
            #print np.transpose(y,[0,2,1]).shape
            sum_cl = np.dot(np.transpose(y, [1,2,0]), self.v)

            for n in range(N):
                if (np.any(sum_cl[n]<0)):
                    print "Warning, sum<=0!"
                ce_n +=np.dot(y_target[n,:],np.log(sum_cl[n]))


        else:
            sum_cl = np.dot(y.T, self.v)
            M, N = y.shape  # m-clients, n-data points
            for n in range(N):
                if (sum_cl[n]>=1 or sum_cl[n]<=0):
                    print "Warning about the sum!"
                ce_n += y_target[n]*np.log(sum_cl[n]) + (1-y_target[n])*np.log(1-sum_cl[n])
        ce_n = -ce_n/N

        #print "Y: {0}".format(sum_cl.T)
        #### we have annotation errors in additon
        sum_annot_error = 0
        num_points = 0
        for c in range(len(y_client_annotations)):
            for l in range(len(y_client_annotations[c])):

                sum_annot_error += np.linalg.norm(y_client_annotations[c][l] - y[c,y_client_annotation_indices[c][l]],2)
                num_points+=1



        return ce_n + self.PARAM_LAMBDA * np.linalg.norm(self.v, 1) + sum_annot_error*PARAM_LAMBDA_ANNOTATIONS/float(num_points)

    def log_reg(self,x,w):
        num_data = x.shape[0]
        num_clients = w.shape[0]
        if (self.multiclass):
            y = np.zeros((num_clients, num_data, self.num_classes))

            for c in range(num_clients):
                for i in range(num_data):
                    sum_c = 0
                    for k in range(self.num_classes):
                        #print "{0}".format(np.dot(x[i,:], w[c,:,k]))
                        y[c,i,k] = np.exp(np.dot(x[i,:],w[c,:,k]))
                        sum_c +=y[c,i,k]
                    y[c,i,:] = y[c,i,:]/sum_c
                    
        else:

            y = np.zeros((num_clients, num_data))

            for c in range(num_clients):
                for i in range(num_data):
                    y[c,i] = 1/(1+np.exp(-np.dot(w[c,:],x[i,:]))) # sigmoid
        return y

    def grad_log_reg(self, x, w): # gradient of logistic regression
        grad_list = [] # gradients for every client
        num_data = x.shape[0]
        num_params = x.shape[1]
        num_clients = w.shape[0]
        
        if (self.multiclass):
            for c in range(num_clients):
                grad_list_classes = []
                grad_matrix = np.zeros((num_data,num_params))
                sum_exp=0
                exp_list = [] # np.zeros((num_data,num_params))
                for k in range(self.num_classes):
#                    print "{0} {1}".format(self.w[c,:,x].shape, x.shape)
                    exp_1 = np.exp(x*self.w[c,:,k])
                    exp_list.append(exp_1)
                    #exp_1 = np.expand_dims(exp_1,1)
                    sum_exp+=exp_1

                for k in range(self.num_classes):
                    grad_matrix = x*exp_list[k]/sum_exp-x*exp_list[k]*exp_list[k]/(sum_exp*sum_exp)
                    grad_list_classes.append(grad_matrix)

                grad_list.append(grad_list_classes)
        else:
            
            for c in range(num_clients):
                grad_matrix = np.zeros((num_data, num_params))
                for i in range(num_data):
                    grad_matrix[i,:] = np.dot(x[i, :],np.exp(np.dot(w[c,:], x[i,:])))/np.power((np.exp(np.dot(w[c,:],x[i,:]))+1),2)
                grad_list.append(grad_matrix)
        return grad_list # return list of gradients for every client

    def grad_ce_w(self,x, y_target, y_client_annotations, y_client_annotation_indices, w,v,PARAM_LAMBDA_ANNOTATIONS): # return vector size(w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        num_params = w.shape[1]


        if (self.multiclass):
            y_gradients = self.grad_log_reg(x,w) # list of matrices size (n,num_params,num_classes)
            sum_grad = np.zeros((num_clients, num_params,self.num_classes))
            sum_grad_annotations = np.zeros((num_clients,num_params, self.num_classes))
            num_points=0
            y = self.log_reg(x,w) # num_clients x num_data x num_classes
            for c in range(num_clients):
                for n in range(num_data):
                    sum_cl = np.dot(v,y[:,n,:])
                    for k in range(self.num_classes):
                        sum_grad[c,:,k] += y_target[n,k]*(1/sum_cl)*v[c]*y_gradients[c][n,:,k]
                cl_indices_list = y_client_annotation_indices[c]
                for i, ind in enumerate(cl_indices_list):
                    for k in range(self.num_classes):
                        sum_grad_annotations[c, :,k] += 2 * (y[c, ind,k] - y_client_annotations[c][i,k]) * y_gradients[c][k][ind,:]  # additional gradient
                    num_points += 1

            sum_grad = -(sum_grad / num_data) + sum_grad_annotations * PARAM_LAMBDA_ANNOTATIONS / float(num_points)

        else:
            y_gradients = self.grad_log_reg(x,w) # return list of matrices size(n,w)

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
        if (self.multiclass):
            sum_grad = np.zeros(num_clients)
            for n in range(num_data):
                for k in range(self.num_classes):
                    sum_cl = np.dot(v, y[:, n,k])
                #print y_target.shape
                #print y.shape
                    sum_grad += y_target[n,k] * (1 / sum_cl) * y[:, n,k]


        else:
            sum_grad = np.zeros(num_clients)
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
        if (self.multiclass):
            y_target_decimal = np.zeros((N,1))
            for n in range(N):
                label = np.argmax(np.dot(self.v,y[:,n]))
                y_labels[n] = label
                y_target_decimal[n] = np.argmax(y_target[n,:])
#                print y_labels[n]
#                print y_target_decimal[n]
            return np.sum(y_labels!=y_target_decimal)*100/float(len(y_labels))
            
            
        else:
            for n in range(N):
                label_float = np.dot(self.v,y[:,n])

                if label_float>0.5:
                    y_labels[n] = 1
                    

            return np.sum(y_labels!=y_target)*100/float(len(y_labels))

    def test(self, test_X, test_Y, test_annotation_indices, test_annotations, PARAM_LAMBDA_ANNOTATIONS, method='MLP'):
        data_num, data_dim = test_X.shape
        if (method == 'MLP'):
            test_X_intermediate = self.sigmoid_array(np.dot(test_X, self.mlp_pretrained.coefs_[0]))
            client_response = self.log_reg(test_X_intermediate, self.w)  # clients x num_data
        elif (method=='LOGREG'):
            client_response = self.log_reg(test_X, self.w) # clients x num_data
        else:
            pass
        error_count = self.errors_count(client_response, test_Y)
        print "TESTING ACCURACY: {0}%".format((data_num-error_count)*100/data_num)

        if (method== 'MLP'):
            loss = self.cross_entropy_all(test_X_intermediate, client_response, test_Y, test_annotations, test_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        elif (method=='LOGREG'):
            loss = self.cross_entropy_all(test_X, client_response, test_Y, test_annotations, test_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        else:
            pass
        return (data_num-error_count)*100/data_num







