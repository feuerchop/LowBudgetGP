__author__ = 'kolosnjaji'
import numpy as np
import warnings
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from pathos.multiprocessing import ProcessPool

class GenGradDescModelNoAnnotations:

    def __init__(self, w_init, v_init, w_init_h = None, multiclass=False, num_classes=10, stochastic=False):
        self.w = w_init
        if np.any(w_init_h):
            self.w_h = w_init_h
            self.num_h = self.w_h.shape[1]
        self.v = v_init
        self.num_clients = self.w.shape[0]
        self.multiclass = multiclass
        self.num_classes = num_classes
        self.stochastic = stochastic
        self.avg_grad_h =0
        self.avg_grad_o =0
        self.epsilon=1e-15
        self.p = ProcessPool(nodes=10)
        #self.PARAM_LAMBDA = 0.0001

    @staticmethod
    def sigmoid_array(x):

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax_array(a):
        [n,k] = a.shape
        softmax_mat = np.zeros((n,k))
        for i in range(n):
            softmax_mat[i,:] = np.exp(a[i,:])/float(np.sum(np.exp(a[i,:])+self.epsilon))
        return softmax_mat
            

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

    def optimization(self,x,y_target, y_client_annotation_indices, y_client_annotations, NUM_IT, NUM_IT_W, NUM_IT_V, PARAM_LAMBDA_W, PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA, TIMESTEP, method='LOGREG', stochastic_size=0):
        self.method = method
        self.h_num =  x.shape[0] # number of hidden layers 
        train_loss = []
        v_nonzero = []
        error_percentage = []
 
        self.PARAM_LAMBDA = PARAM_LAMBDA
        num_clients = len(y_client_annotation_indices)
        if (method=="MLP"):
        #    self.pretrain(x,y_target)
        #    self.x_intermediate = self.sigmoid_array(np.dot(x, self.mlp_pretrained.coefs_[0]))
        #    self.w = np.tile(self.mlp_pretrained.coefs_[1].T,(num_clients,1))
            y = self.mlp(x, self.w)


        elif (method=="LOGREG"):
            y = self.log_reg(x, self.w)

        else:
            print "Method not recognized!"
            exit(-1)
        print "Computing the loss"
        loss = self.cross_entropy_all(x, y, y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        print "Initial Loss: {0}".format(loss)
        print "Optimization..."
        for i in range(NUM_IT):
            print "Optimizing w..."
            for j in range(NUM_IT_W):
                print j
                if (self.stochastic):
                    num_data = np.size(x,0)
                    stochastic_subset = np.random.randint(0, num_data, int(stochastic_size))
                else:
                    stochastic_subset=[]
                print "Computing the gradient..."
                if (method=="MLP"): # two sets of weights (3-layer perceptron)
                    w_old_h = self.w_h
                    w_old_o = self.w
                    (grad_w_h, grad_w_o) = self.grad_ce_w_backprop(x, y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS, stochastic_subset)
                    new_avg, grad_update_h = self.rmsprop(self.avg_grad_h, grad_w_h, 0.1, PARAM_LAMBDA_W)
                    self.avg_grad_h = new_avg
                    new_avg, grad_update_o = self.rmsprop(self.avg_grad_o, grad_w_o, 0.1, PARAM_LAMBDA_W)
                    self.avg_grad_o = new_avg
                    self.w = self.w-grad_update_o
                    self.w_h = self.w_h - grad_update_h
                else:
                    w_old = self.w
                    grad_w = self.grad_ce_w(x,y_target, y_client_annotations, y_client_annotation_indices, self.w,self.v, PARAM_LAMBDA_ANNOTATIONS, stochastic_subset)
                    print "Grad W: pos: {0} neg: {1} zero: {2}".format(np.sum(grad_w>0), np.sum(grad_w<0), np.sum(grad_w==0))
                    new_avg, grad_update = self.rmsprop(self.avg_grad, grad_w, 0, PARAM_LAMBDA_W)
                    self.w = self.w-grad_update
                    self.avg_grad = new_avg

                #self.w = self.w - grad_w*PARAM_LAMBDA_W;

                #print "W:\n" + str(self.w)
            print "Optimizing v..."
            v_old = self.v
#            NUM_IT_P_V = NUM_IT_P
#            NUM_IT_P_V=0

            for j in range(NUM_IT_V):
                print "V:\n{0}".format(self.v * 100)
                print j

                if (self.stochastic):
                    stochastic_subset = np.random.randint(0, num_data, int(stochastic_size))
                else:
                    stochastic_subset = []
                

                if (method=='MLP'):
                    grad_v = self.grad_ce_v(x, y_target, self.w, self.v,  stochastic_subset)
                else:
                    grad_v = self.grad_ce_v(x,y_target,self.w,self.v, stochastic_subset)
                #print self.v                
                self.v = self.grad_update(self.v,grad_v, TIMESTEP=TIMESTEP)
                #self.v = self.v - grad_v*TIMESTEP

                #print self.v
                self.v[self.v<0] = 0
                self.v = self.v/np.sum(self.v) # keep sum to 1

            if (method=="MLP"):
                y = self.mlp(x,self.w)
            else:
                y = self.log_reg(x, self.w)

            #print "W: {0}".format(self.w)
            #print np.where(self.v>0)
            #print self.v
            print "V not 0:{0}".format(len(np.where(self.v>0)[0]))
            if len(np.where(self.v>0)[0])==0:
                print "V underflow!"
                self.v = v_old
                if (method == "MLP"):
                    self.w = w_old_o
                    self.w_h = w_old_h
                else:
                    self.w = w_old
                return (train_loss, v_nonzero, error_percentage)
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
        for c in range(len(y_client_annotations)): # for all clients
            for l in range(len(y_client_annotations[c])): # for all annotations
                sum_annot_error += np.linalg.norm(y_client_annotations[c][l] - y[c,y_client_annotation_indices[c][l]])
                num_points+=1



        return ce_n + self.PARAM_LAMBDA * np.linalg.norm(self.v, 1) + sum_annot_error*PARAM_LAMBDA_ANNOTATIONS/float(num_points) 
        

    def y_o_parallel(self,c):
        a_h = np.dot(self.x, self.w_h[c,:,:])
        y_h = self.softmax_array(a_h)
        a_o = np.dot(y_h, self.w[c,:,:])
        y_o = self.softmax_array(a_o)
        return (a_h,y_h,a_o,y_o)

#    def y_h_parallel(self,c):
#        return self.softmax_array(self.a_h[c])

#    def a_o_parallel(self,c):
#        return np.dot(self.y_h[c], self.w[c,:,:])

#    def y_o_parallel(self,c):
#        return self.softmax_array(self.a_o[c])

#        return self.softmax_array(np.dot(self.softmax_array(np.dot(self.x,self.w_h[c,:,:])), self.w[c,:,:]))

    def mlp(self,x,w): # compute forward pass of MLP, not sure why we send w
        print "Computing mlp..."
        num_data = x.shape[0]
        num_clients = w.shape[0]
        self.a_h = [0 for i in range(num_clients)]
        self.y_h = [0 for i in range(num_clients)]
        self.a_o = [0 for i in range(num_clients)]
        self.y_o = [0 for i in range(num_clients)]
           
        y_o = np.zeros((num_clients, num_data, self.num_classes), dtype=np.float32)
        self.x = x
        result_list = self.p.map(self.y_o_parallel, range(num_clients))        
        print "parallel computed"
        for c in range(num_clients):
            self.a_h[c] = result_list[c][0]
            self.y_h[c] = result_list[c][1]
            self.a_o[c] = result_list[c][2]
            self.y_o[c] = result_list[c][3]
            y_o[c,:,:] = self.y_o[c]
        return y_o

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

    def grad_softmax(self, a): # gradient of softmax (y=f(a)) w.r.t. the sum w*x, for n=1..N
        [n,k] = a.shape # k is number of classes if last layer
        grad_matrix = np.zeros((n,k))
        sum_exp = 0
        grad_softmax_val = self.softmax_array(a)               
            
        return grad_softmax_val*(1-grad_softmax_val)

    def grad_output_layer(self, x): # grad output layer: nxhxk
        grad_list = []
        num_data = x.shape[0]
        num_clients = len(self.a_o)
        for c in range(num_clients):
            n_grad = np.zeros((num_data,self.num_classes, self.num_h, self.num_classes))
            for n in range(num_data):
                grad_softmax = self.grad_softmax(self.a_o[c])
                y_tile = np.zeros((self.num_h, self.num_classes, self.num_classes))
                for i in range(self.num_classes):
                    y_tile[:,i,i] = self.y_h[c][n,:]
                           
                n_grad[n, :,:,:] = np.dot(np.diag(grad_softmax[n,:]), y_tile)
            #grad_list.append(np.dot(np.tile(self.y_h[c],(1,1,k)) ,self.grad_softmax(self.a_o[c])))
            grad_list.append(n_grad)
        return grad_list # list of gradients for output layer of all the clients
        

    def grad_hidden_parallel(self, c):
        num_params = self.x.shape[1]
        num_data = self.x.shape[0]
        grad_y_a_o = self.grad_softmax(self.a_o[c])
        grad_y_a_h = self.grad_softmax(self.a_h[c]) # nxh  
        grad_o = np.zeros((num_data, self.num_classes, num_params, self.num_h))
        for n in range(num_data):
            b = np.dot(np.dot(np.diag(grad_y_a_o[n,:,]), self.w[c,:,:].T), np.diag(grad_y_a_h[n,:]))
            for k in range(self.num_classes):
                for d in range(num_params):
                    for we in range(self.num_h):
                        grad_o[n,k,d,we] = b[k,we] * self.x[n,d]
        return grad_o
                
        

    def grad_hidden_layer(self,x): # grad hidden layer
        [num_data, num_params] = x.shape        
        num_clients = len(self.a_o)
        print "Number of clients:{0}".format(num_clients)
        grad_list = []
        self.x = x
        grad_list = self.p.map(self.grad_hidden_parallel, range(num_clients))
        return grad_list

                                                        
    def grad_mlp(self, x):
        num_data = x.shape[0]
        num_params = x.shape[1]
        num_clients = self.w.shape[0]        
        self.mlp(x, self.w) # new stochastic subset

        if (self.multiclass):
            grad_list_output = self.grad_output_layer(x)
            grad_list_hidden = self.grad_hidden_layer(x)
            return (grad_list_hidden, grad_list_output)

    def grad_ce_w_backprop(self, x,y_target, y_client_annotations, y_client_annotation_indices, PARAM_LAMBDA_ANNOTATIONS, stochastic_subset): # we assume there is a stochastic subset
        print "Computing grad of mlp"
        (grad_list_hidden, grad_list_output) = grad_lists =  self.grad_mlp(x[stochastic_subset,:])
        num_clients = self.w.shape[0]
        
        
         
        sum_grad_output = np.zeros((num_clients, self.num_classes, self.num_h,self.num_classes))
        sum_grad_annotations = np.zeros((num_clients,self.num_classes,self.num_h, self.num_classes))
        num_points=0
        print "Computing forward pass..."
        y = self.mlp(x[stochastic_subset,:],self.w) # num_clients x num_data x num_classes                                                                                                               

        num_clients = self.w.shape[0]
        num_data = len(stochastic_subset)
        num_params = x.shape[1]
           
        for c in range(num_clients):
            for n in range(num_data):
                for k in range(self.num_classes):
                    sum_cl = np.dot(self.v,y[:,n,k])
                    sum_grad_output[c,k,:,:] += y_target[stochastic_subset[n],k]*(1/sum_cl)*self.v[c]*grad_list_output[c][n, k, :,:]#y_gradients[c][k][n,:]
                    cl_indices_list = y_client_annotation_indices[c]
                for i, ind in enumerate(cl_indices_list):
                    for k in range(self.num_classes):
                        if ind in stochastic_subset:                                
                            sum_grad_annotations[c, k,:,:] += 2 * (y[c, n,k] - y_client_annotations[c][i,k]) * grad_list_output[c][stochastic_subset.tolist().index(ind),k,:,:]  # additional gradient                                                                                                                 
                            num_points += 1                           


        sum_grad_output = -(sum_grad_output / num_data) + sum_grad_annotations * PARAM_LAMBDA_ANNOTATIONS / float(num_points)

        sum_grad_hidden = np.zeros((num_clients, self.num_classes, num_params,self.num_h))
        sum_grad_annotations = np.zeros((num_clients, self.num_classes, num_params, self.num_h))       
       
        num_points=0
        #print "Computing forward pass..."
        #y = self.mlp(x[stochastic_subset,:],self.w) # num_clients x num_data x num_classes                                                                                                                 \
                                                                                                                                                                                                            
        for c in range(num_clients):
            for n in range(num_data):
                for k in range(self.num_classes):
                    sum_cl = np.dot(self.v,y[:,n,k])
                    sum_grad_hidden[c,k,:,:] += y_target[stochastic_subset[n],k]*(1/sum_cl)*self.v[c]*grad_list_hidden[c][n, k,:,:]#y_gradients[c][k][n,:]                               
                    cl_indices_list = y_client_annotation_indices[c]
                for i, ind in enumerate(cl_indices_list):
                    for k in range(self.num_classes):
                        if ind in stochastic_subset:
                            sum_grad_annotations[c, k, :,:] += 2 * (y[c, n,k] - y_client_annotations[c][i,k]) * grad_list_hidden[c][stochastic_subset.tolist().index(ind),k,:,:]  # additional gradient                                                                                                                                                           
                            num_points += 1

        sum_grad_hidden = -(sum_grad_hidden/num_data) + sum_grad_annotations* PARAM_LAMBDA_ANNOTATIONS/float(num_points)

       
        return (np.sum(sum_grad_hidden, axis=1), np.sum(sum_grad_output, axis=1))
       
          
        
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
                    if (np.isnan(np.max(w[c,:,k])) or np.isnan(np.min(w[c,:,k]))):
                        print "NaN!"
                        exit(-1)
                    exp_1 = np.exp(x*self.w[c,:,k])
                    exp_list.append(exp_1)
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

       

    def grad_ce_w(self,x, y_target, y_client_annotations, y_client_annotation_indices, w,v,PARAM_LAMBDA_ANNOTATIONS, stochastic_subset=0): # return vector size(w)
        num_data = x.shape[0]
        num_clients = w.shape[0]
        num_params = w.shape[1]

        if (self.stochastic):
            y_target = y_target[stochastic_subset,:]
            x = x[stochastic_subset,:]
            num_data = len(stochastic_subset)


        if (self.multiclass): # MULTICLASS CLASSIFICATION (e.g. MNIST)
            y_gradients = self.grad_log_reg(x,w) # list of matrices size (n,num_params,num_classes)
            sum_grad = np.zeros((num_clients, num_params,self.num_classes))
            sum_grad_annotations = np.zeros((num_clients,num_params, self.num_classes))
            num_points=0
            y = self.log_reg(x,w) # num_clients x num_data x num_classes
            for c in range(num_clients):
                for n in range(num_data):
                    for k in range(self.num_classes):
                        sum_cl = np.dot(v,y[:,n,k])
                        sum_grad[c,:,k] += y_target[n,k]*(1/sum_cl)*v[c]*y_gradients[c][k][n,:]
                cl_indices_list = y_client_annotation_indices[c]
                for i, ind in enumerate(cl_indices_list):
                    if (self.stochastic and (ind in stochastic_subset)):
                        for k in range(self.num_classes):
                            sum_grad_annotations[c, :,k] += 2 * (y[c, np.where(stochastic_subset==ind)[0][0],k] - y_client_annotations[c][i,k]) * y_gradients[c][k][np.where(stochastic_subset==ind)[0][0],:]  # additional gradient
                        num_points += 1
                    elif (not self.stochastic):
                        for k in range(self.num_classes):
                            sum_grad_annotations[c, :,k] += 2 * (y[c, ind,k] - y_client_annotations[c][i,k]) * y_gradients[c][k][ind,:]  # additional gradient                                                                     num_points += 1
                    else:
                        pass



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
                    if (self.stochastic and (ind in stochastic_subset)):
                        sum_grad_annotations[c,:] += 2*(y[c,np.where(stochastic_subset==ind)[0][0]]-y_client_annotations[c][i])*y_gradients[c][np.where(stochastic_subset==ind)[0][0],:]
                        num_points+=1
                    elif (not self.stochastic):
                        sum_grad_annotations[c,:] += 2*(y[c,ind]-y_client_annotations[c][i]) * y_gradients[c][ind,:] # additional gradient
                        num_points+=1
                    else:
                        pass
            sum_grad = -(sum_grad/num_data) + sum_grad_annotations*PARAM_LAMBDA_ANNOTATIONS/float(num_points)
        return sum_grad

    def grad_ce_v(self,x,y_target,w, v, stochastic_subset=0): # return vector, size(v) = number of clients

        if (self.method=="MLP"):
            y = self.mlp(x,w)
        else:
            y = self.logreg(x,w)
        num_data = x.shape[0]
        num_clients = w.shape[0]

        if (self.multiclass):                                                                                                                                                                            
            y_target = y_target[stochastic_subset,:]                                                                                                                                                              
            x = x[stochastic_subset,:]                                                                                                                                                                     
            num_data = len(stochastic_subset)
               

        if (self.multiclass):
            sum_grad = np.zeros(num_clients)
            for n in range(num_data):
                for k in range(self.num_classes):
                    sum_cl = np.dot(v, y[:, n,k])
                    sum_grad += y_target[n,k] * (1 / sum_cl) * y[:, n,k]


        else:
            sum_grad = np.zeros(num_clients)
            for n in range(num_data):
                sum_cl = np.dot(v, y[:, n])
                #sum_grad += y_target[n]*(1/sum_cl)*np.log(y[:,n]) + (1-y_target[n]) * (1/(1-sum_cl))*np.log(1-y[:,n])
                sum_grad += y_target[n]*(1/sum_cl)*y[:,n] + (1-y_target[n]) * (1/(1-sum_cl))*(-y[:,n])

        return -sum_grad

    def rmsprop(self, grad_avg, grad, gamma, learn_rate):
        grad_new = gamma*grad_avg + (1-gamma)*grad*grad
        update = learn_rate*grad/(np.sqrt(grad_new)+self.epsilon)
        return (grad_new, update)


    def errors_count(self,y,y_target):
        N = y_target.shape[0]
        y_labels = np.zeros((N,1))
        if (self.multiclass):
            y_target_decimal = np.zeros((N,1))
            for n in range(N):
                label = np.argmax(np.dot(self.v,y[:,n]))
                y_labels[n] = label
                y_target_decimal[n] = np.argmax(y_target[n,:])
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
            #test_X_intermediate = self.sigmoid_array(np.dot(test_X, self.mlp_pretrained.coefs_[0]))
            client_response = self.mlp(test_X, self.w)  # clients x num_data
        elif (method=='LOGREG'):
            client_response = self.log_reg(test_X, self.w) # clients x num_data
        else:
            pass
        error_count = self.errors_count(client_response, test_Y) # percentage
        print "TESTING ACCURACY: {0}%".format(100-error_count)

        if (method== 'MLP'):
            loss = self.cross_entropy_all(test_X, client_response, test_Y, test_annotations, test_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        elif (method=='LOGREG'):
            loss = self.cross_entropy_all(test_X, client_response, test_Y, test_annotations, test_annotation_indices, PARAM_LAMBDA_ANNOTATIONS)
        else:
            pass
        return 100-error_count







