__author__ = 'kolosnjaji'
import numpy as np
import warnings
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
import tensorflow as tf

class GenGradDescModelNoAnnotations:

    def __init__(self, w_init, v_init, w_h_init, multiclass=False, num_classes=10, stochastic=False):
        self.w = w_init
        self.w_h = w_h_init
        self.v = v_init
        self.num_clients = self.w.shape[0]
        self.num_hidden = self.w_h.shape[2]
        self.multiclass = multiclass
        self.num_classes = num_classes
        self.stochastic = stochastic
        self.avg_grad =0
        self.epsilon=1e-10
        #self.PARAM_LAMBDA = 0.0001
        self.y_ = tf.placeholder(tf.float32, [None, 10]) # ground truth tensor
        self.x_tensor = tf.placeholder(tf.float32, [None, 784])
        self.y_annotations_list = [tf.placeholder(tf.float32, [None, 10]) for i in range(self.num_clients)]
        self.y_annotations_indices = [tf.placeholder(tf.float32) for i in range(self.num_clients)]
        self.mlps = []
        self.PARAM_LAMBDA = tf.placeholder(tf.float32)
        self.PARAM_LAMBDA_ANNOTATIONS = tf.placeholder(tf.float32)
        for i in range(self.num_clients):
            [y,W, W_h] = self.create_mlp(self.x_tensor, i)
            self.mlps.append([y,W, W_h])
        self.v_tensor = tf.Variable(self.v, "v", dtype=tf.float32)
        y_all = tf.stack([mlp[0] for mlp in self.mlps])
        #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(tf.reduce_sum(tf.map_fn(self.mul_fn, tf.transpose(y_all, [2,0,1]))))))
        self.y_total = tf.squeeze(tf.map_fn(self.mul_fn, tf.transpose(y_all, [2,0,1])))
        self.log_y = tf.transpose(self.y_total+self.epsilon)
        self.mul_y = tf.multiply(self.y_, tf.log(self.log_y))
        self.ce_gt =  tf.reduce_sum(tf.multiply(self.y_,tf.log(tf.transpose(self.y_total + self.epsilon))))
        self.cross_entropy = self.ce_gt #tf.reduce_sum(tf.multiply(self.y_,tf.log(tf.transpose(self.y_total))))
        for c in range(self.num_clients): # annotations
            indices = tf.where(tf.not_equal(self.y_annotations_indices[c], tf.constant(0, dtype=tf.float32)))
            self.cross_entropy += 2* tf.reduce_sum(tf.square(tf.gather(self.y_annotations_list[c]-self.mlps[c][0], indices)))*self.PARAM_LAMBDA_ANNOTATIONS
        self.cross_entropy += self.PARAM_LAMBDA*tf.norm(self.v_tensor,1)

        

        

    def mul_fn(self, input_d):
        return tf.matmul(tf.expand_dims(self.v_tensor,0), input_d)

    def create_mlp(self, x_tensor, i):
        W = tf.get_variable('W_{0}'.format(i),  initializer=tf.constant(self.w[i,:,:], dtype=tf.float32))
        W_h = tf.get_variable('W_h_{0}'.format(i), initializer=tf.constant(self.w_h[i,:,:], dtype=tf.float32))
        y = tf.nn.softmax(tf.matmul(tf.matmul(self.x_tensor, W_h), W)) # tensor y
        print y.get_shape()
        return [y,W,W_h]

    def initialize_annotations(self, x, y_client_annotation_indices, y_client_annotations):
        num_clients = len(y_client_annotation_indices)
        num_data = x.shape[0]
        y_client_annotation_matrix = [np.zeros(( num_data, self.num_classes)) for i in range(num_clients)]
        y_client_annotation_indices_matrix = [np.zeros(num_data) for i in range(num_clients)]
        for c in range(num_clients):
            for i,n in enumerate(y_client_annotation_indices[c]):
                y_client_annotation_matrix[c][n,:] = y_client_annotations[c][i,:]
                y_client_annotation_indices_matrix[c][n] = 1
        return y_client_annotation_matrix, y_client_annotation_indices_matrix
        

    def optimization(self,x,y_target, y_client_annotation_indices, y_client_annotations, NUM_IT, NUM_IT_W, NUM_IT_V, PARAM_LAMBDA_W, PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA, TIMESTEP, method='LOGREG', stochastic_size=0):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        train_loss = []
        v_nonzero = []
        error_percentage = []
        y_client_annotation_list, y_client_annotation_indices_list = self.initialize_annotations(x, y_client_annotation_indices, y_client_annotations)
        
        num_clients = len(y_client_annotation_indices)

        y = self.mlp(x,self.w)

        feed_dict = {self.x_tensor:x, self.y_: y_target, self.PARAM_LAMBDA:PARAM_LAMBDA, self.PARAM_LAMBDA_ANNOTATIONS:PARAM_LAMBDA_ANNOTATIONS}
        for c in range(self.num_clients):
            feed_dict[self.y_annotations_list[c]] = y_client_annotation_list[c]
            feed_dict[self.y_annotations_indices[c]] = y_client_annotation_indices_list[c]
                                 
        print "Optimization"
        for iter in range(NUM_IT):
            # computing the loss
            loss = self.cross_entropy.eval(feed_dict=feed_dict, session = self.session)
            # computing the gradients
            gradients_w = [tf.gradients(self.cross_entropy, self.mlps[c][1])[0] for c in range(self.num_clients)]
            gradients_w_h = [tf.gradients(self.cross_entropy, self.mlps[c][2])[0] for c in range(self.num_clients)]
            gradients_v = tf.gradients(self.cross_entropy, self.v_tensor)[0]
#            print "Loss:{:.10f}".format(loss)
#            print "logy:{0}".format(np.log(self.log_y.eval(feed_dict=feed_dict, session= self.session)))
            # updates
            print self.v_tensor.eval(session=self.session)
            for c in range(self.num_clients):
                w_update = self.mlps[c][1].eval(session=self.session) - gradients_w[c].eval(session=self.session, feed_dict=feed_dict)*PARAM_LAMBDA_W
                w_h_update = self.mlps[c][2].eval(session=self.session) - gradients_w_h[c].eval(session=self.session, feed_dict=feed_dict)*PARAM_LAMBDA_W                
                v_tensor_update = self.v_tensor.eval(session=self.session) - gradients_v.eval(session=self.session, feed_dict=feed_dict)*TIMESTEP
#                print "Updating w..."
                self.session.run(self.mlps[c][1].assign(w_update))
                loss = self.cross_entropy.eval(feed_dict=feed_dict, session = self.session)
#                print "Loss: {:.10f}".format(loss)
#                print "Updating w_h..."
                self.session.run(self.mlps[c][2].assign(w_h_update))
                loss = self.cross_entropy.eval(feed_dict=feed_dict, session = self.session)
#                print "Loss: {:.10f}".format(loss)
#                reduction=1
#                while (v_tensor_update<0).any(): # something like line search
#                    reduction*=2

#                print "Updating v..."
                v_tensor_update = self.v_tensor.eval(session=self.session) - gradients_v.eval(session=self.session, feed_dict=feed_dict)*TIMESTEP
                v_tensor_update[v_tensor_update<0] = 0
                self.session.run(self.v_tensor.assign(v_tensor_update/np.sum(v_tensor_update)))
                loss = self.cross_entropy.eval(feed_dict=feed_dict, session = self.session)
#                print "Loss: {:.10f}".format(loss)
#            print "Gradients w: {0}".format([gradients_w[i].eval(session=self.session, feed_dict=feed_dict) for i in range (len(gradients_w))])
#            print "Update w: {0}".format(w_update)
#            print "Gradients w_h: {0}".format([gradients_w_h[i].eval(session=self.session, feed_dict=feed_dict) for i in range(len(gradients_w_h))])
#            print "Update w_h {0}".format(w_h_update)
#            print "Gradients w_h: {0}".format(w_h_update)
#            print "Gradients: {0}".format(gradients_v.eval(session=self.session, feed_dict=feed_dict))
#            print "Update: {0}".format(v_tensor_update)
#            print "V:{0}".format(self.v_tensor.eval(session=self.session))
            
            
            errors = self.errors_count(x,y_target)
            print "Iteration {0} Loss: {1} Error Percentage:{2}%".format(iter, loss, errors)
            train_loss.append(loss)
            error_percentage.append(errors)
            self.v = self.v_tensor.eval(session=self.session)
            v_nonzero.append(len(np.where(self.v>0)[0]))

        return (train_loss, v_nonzero, error_percentage)

    def mlp(self,x,w):
        num_data = x.shape[0]
        num_clients = w.shape[0]
        y = np.zeros((num_clients, num_data, self.num_classes))
        print "MLP computations"
        for c in range(num_clients):
            print c
            print x.shape
            y[c,:,:] = self.mlps[c][0].eval(feed_dict={self.x_tensor:x}, session = self.session)
        
        return y
        



    def errors_count(self,x,y_target):
        y_results = self.y_total.eval(feed_dict={self.x_tensor:x, self.y_:y_target}, session = self.session).T
#        print "Result shape :{0}".format(y_results.shape)
        N = x.shape[0]
        y_labels = np.zeros((N,1))
        y_target_int = np.zeros((N,1))
        for n in range(N):
            label = np.argmax(y_results[n,:])
            y_labels[n] = label
            y_target_int[n] = np.argmax(y_target[n,:])
        return np.sum(y_labels!=y_target_int)*100/float(len(y_labels))
        

    def test(self, test_X, test_Y, test_annotation_indices, test_annotations, PARAM_LAMBDA_ANNOTATIONS, method='MLP'):
        error_count = self.errors_count(test_X, test_Y) # percentage
        print "TESTING ACCURACY: {0}%".format(100-error_count)
        return 100-error_count







