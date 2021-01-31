__author__ = 'kolosnjaji'

import numpy as np
import h5py
from training import LowBudgetCrowdsourcing
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from optimization_no_annotations import GenGradDescModelNoAnnotations
import sys
import scipy.io as sio
import ConfigParser
import pickle


def load_data_uai(data_i):

    labels_list = []
    data_list = []
    groundtruth_list = []
    # /home/bojan/research/datasets/UAI14_data/class_data_{0}.mat

    print data_i
    config = ConfigParser.ConfigParser()
    config.readfp(open('params.conf'))
    data_path = config.get('General', 'DATA_PATH')
    f = h5py.File(data_path + '/class_data_{0}.mat'.format(data_i), 'r')
    labels_list.append(f['L'][:].T)
    data_list.append(f['x'][:].T)
    groundtruth_list.append(f['Y'][:].T)

    #print "Stacking..."
    my_labels = np.hstack(labels_list).T
    my_data = np.vstack(data_list)
    my_gt = np.vstack(groundtruth_list)

    return my_data, my_labels, my_gt

def load_data_synthetic():
    data_in = sio.loadmat('../../data/data_in.mat')
    #print "Max: {0}".format(np.max(data_in))
    num_clients = np.size(data_in['Y_sigmoid'],2)
    all_data = []
    all_data.append(data_in['Xt']) # x
    all_data.append(data_in['Z_sigmoid']) # ground truth
    all_data.append([]) # annotation indices
    all_data.append([])  # annotations
    for i in range(num_clients):
        all_data[2].append(range(0,len(data_in['Xt'])))
        all_data[3].append(data_in['Y_sigmoid'][:,:,i])

    return all_data

def load_data_mnist(num_experts):
    print "trying to load data..."
    data_in = sio.loadmat('../../data/mnist_dict_annotations_{0}.mat'.format(num_experts))
    print "Max: {0}".format(np.max(data_in['X']))
    print "mat file loaded"
    num_clients = np.size(data_in['y_annotations'],0)
    print num_clients
    all_data = []
    all_data.append(data_in['X'])
    all_data.append(data_in['y'])
    all_data.append([])
    all_data.append([])
    for i in range(num_clients):
        all_data[2].append(range(0, len(data_in['X'])))
        all_data[3].append(data_in['y_annotations'][i,:])

    return all_data

def divide_indices_mnist(all_data, num_cv=2):
    num_data = np.size(all_data[0], 0)
    print "dividing indices..."
    divide_indices = divide_indices_uai(num_cv, num_data)
    return divide_indices

def divide_indices_synthetic(all_data, num_cv=2):
    num_data = np.size(all_data[0], 0)
    divide_indices = divide_indices_uai(num_cv, num_data)
    return divide_indices

def divide_data_synthetic(all_data, cv_indices, test_i):
    cv_division = cv_indices[test_i]
    training_indices = cv_division[0]
    test_indices = cv_division[1]

    X_training = all_data[0][training_indices,:] #my_data[training_indices, :]
    y_training = all_data[1][training_indices] #my_gt[training_indices]
    y_training_label_indices = []  # num_clientsx list of indices
    y_training_label_values = []  # num_clientsxlist of labels
    y_training_labels = np.squeeze(np.array(all_data[3]))[:, training_indices].T #my_labels[training_indices, :]

    num_clients = y_training_labels.shape[1]
    num_labels = y_training_labels.shape[0]

    for c in range(num_clients):
        y_list = []
        y_indices = []
        for l in range(num_labels):
            if (y_training_labels[l, c] != -100):  # label exists
                y_list.append(y_training_labels[l, c])
                y_indices.append(l)
        y_training_label_values.append(y_list)
        y_training_label_indices.append(y_indices)

    X_test = all_data[0][test_indices, :]
    y_test = all_data[1][test_indices]
    y_test_label_indices = []  # num_clientsx list of indices
    y_test_label_values = []  # num_clientsxlist of labels
    y_test_labels = np.squeeze(np.array(all_data[3]))[:, test_indices].T

    num_labels = y_test_labels.shape[0]

    for c in range(num_clients):
        y_list = []
        y_indices = []
        for l in range(num_labels):
                y_list.append(y_test_labels[l, c])
                y_indices.append(l)
        y_test_label_values.append(y_list)
        y_test_label_indices.append(y_indices)

    train_data = [X_training, y_training, y_training_label_indices, y_training_label_values]
    test_data = [X_test, y_test, y_test_label_indices, y_test_label_values]
    #return (train_data, test_data)

    return [train_data, test_data]



def divide_indices_uai(num_cv=2, num_data=3220):

    num_test = num_data/float(num_cv) # number of data for the test set
    random_indices = np.random.permutation(range(num_data))

    cv_indices = []
    for i in range(num_cv):
        if (i+1)*num_test<len(random_indices):
            max_num = (i+1)*num_test
        else:
            max_num = len(random_indices)

        test_indices = random_indices[int(i*num_test):int(max_num)]
        training_indices = []
        for j in range(len(random_indices)):
            if not random_indices[j] in test_indices:
                training_indices.append(random_indices[j])
        cv_indices.append((training_indices, test_indices))
    return cv_indices

def divide_data_uai_no_annotations(my_data, my_labels, my_gt, cv_indices,i):
    cv_division = cv_indices[i]
    training_indices = cv_division[0]
    test_indices = cv_division[1]
    X_training = my_data[training_indices,:]
    y_training = my_gt[training_indices]
    y_training_label_indices = [] # num_clientsx list of indices
    y_training_label_values = [] # num_clientsxlist of labels
    y_training_labels = my_labels[training_indices, :]

    num_clients = y_training_labels.shape[1]
    num_labels = y_training_labels.shape[0]

    for c in range(num_clients):
        y_list = []
        y_indices = []
        for l in range(num_labels):
            if (y_training_labels[l,c] != -100): # label exists
                y_list.append(y_training_labels[l,c])
                y_indices.append(l)
        y_training_label_values.append(y_list)
        y_training_label_indices.append(y_indices)


    X_test = my_data[test_indices,:]
    y_test = my_gt[test_indices]
    y_test_label_indices = []  # num_clientsx list of indices
    y_test_label_values = []  # num_clientsxlist of labels
    y_test_labels = my_labels[test_indices, :]

    num_labels = y_test_labels.shape[0]

    for c in range(num_clients):
        y_list = []
        y_indices = []
        for l in range(num_labels):
            if (y_test_labels[l,c] != -100): # label exists
                y_list.append(y_test_labels[l,c])
                y_indices.append(l)
        y_test_label_values.append(y_list)
        y_test_label_indices.append(y_indices)


    train_data = (X_training, y_training, y_training_label_indices, y_training_label_values)
    test_data = (X_test, y_test, y_test_label_indices, y_test_label_values)
    return (train_data, test_data)



def divide_data_uai(my_data, my_gt, my_labels, cv_indices, i): # 3220x2400, 21x2400, 2400,
    cv_division = cv_indices[i]
    training_indices = cv_division[0]
    test_indices = cv_division[1]

    num_clients = np.size(my_labels,1)

    X_training_all = []
    y_training_all = []
    gt_training_all = []

    for client in range(num_clients):
        X_training = []
        y_training = []
        gt_training = []

        X_test = []
        y_test = []
        gt_test = []

        labels_1 = len(np.nonzero(my_labels[training_indices,client]==1)[0])
        labels_0 = len(np.nonzero(my_labels[training_indices,client]==0)[0])
        labels_100 = len(np.nonzero(my_labels[training_indices,client]==-100)[0])

        labels_sum_1 = labels_1+labels_100
        labels_sum_0 = labels_0+labels_100

  #      print "Client {0}, 1: {1}, 2: {2}, all: {3}".format(client, labels_sum_1, labels_sum_0, np.size(my_labels,0))
  #      print my_labels[training_indices,client]
        if not (labels_sum_0==len(training_indices) or labels_sum_1==len(training_indices)): # is everything 0 or 1
  #          print "adding client"
            for i in training_indices:
                if my_labels[i, client]!=-100:
                    X_training.append(my_data[i,:])
                    y_training.append(my_labels[i, client])
                    gt_training.append(my_gt[i])

            X_training_all.append(np.vstack(X_training))
            y_training_all.append(y_training)
    #        print y_training
            gt_training_all.append(gt_training)


    for i in test_indices:
        X_test.append(my_data[i,:])
        gt_test.append(my_gt[i,:])
    X_test = np.vstack(X_test)
    gt_test = np.vstack(gt_test)

    training_data = [X_training_all, y_training_all, gt_training_all, gt_x]
    test_data = [X_test, gt_test]
    return [training_data, test_data]

def test_clustering():

    mean_errors_all = []

    np.random.seed(1337)

    for data_number in range(10): # which dataset

        cv_number = 3

        (X,L,Y) = load_data_uai(data_number+1)

        indices = divide_indices_uai(cv_number, np.size(X,0))

        errors_all = []
        min_num_clients = 21

        for i in range(cv_number):

            (training_data, test_data) = divide_data_uai(X,L,Y,indices,i)

            num_clients = len(training_data[0])

            if (num_clients<min_num_clients):
                min_num_clients = num_clients

            num_clients_min = 1
            num_clients_max = num_clients
            num_clients_step = 1

            errors = []

            for k in range(num_clients_min, num_clients_max+1, num_clients_step):

                my_model = LowBudgetCrowdsourcing(top_k = k)

                my_model.train(training_data[0],training_data[1])
                my_predictions = my_model.predict(test_data[0])

                my_error = log_loss(test_data[1], np.squeeze(my_predictions[0]))/len(test_data[1])
#                print my_error
                errors.append(my_error)
            errors_all.append(errors)

#    errors_all = np.vstack(errors_all)
        mean_errors_cv = []
        for cl in range(min_num_clients):
            mean_errors_cv.append(np.mean([errors_all[cv][cl] for cv in range(cv_number)]))

        mean_errors_all.append(np.mean(mean_errors_cv))
        plt.figure()
        plt.plot(range(num_clients_min, min_num_clients+1), mean_errors_cv, '.-')
        plt.savefig('plot_cv_{0}.png'.format(data_number))

def test_synthetic(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, method):
#    np.random.seed(42)
    all_data = load_data_synthetic()
    num_cv = 2
    cv_indices = divide_indices_synthetic(all_data, num_cv)

    loss_cv = []
    num_clients = len(all_data[2])
    for test_i in range(num_cv):
        print "Test {0}".format(test_i)
        training_data, test_data = divide_data_synthetic(all_data, cv_indices, test_i)

        training_data[1] = np.where(training_data[1]>0.5, 1, 0)

        for cl_i in range(num_clients):
            training_data[3][cl_i] = np.where(np.array(training_data[3][cl_i]) > 0.5, 1, 0)

        [n, m] = all_data[0].shape
        random_v = np.random.normal(1, 0.5, num_clients) / num_clients
        random_v = random_v / np.sum(random_v)
        random_w = np.random.normal(0, 0.5, (num_clients, m))
        train_model = GenGradDescModelNoAnnotations(random_w, random_v)
        print "******************TRAINING******************"
        train_model.optimization(training_data[0], training_data[1], training_data[2], training_data[3], NUM_IT=1200,
                                 NUM_IT_P=NUM_IT_P, PARAM_LAMBDA_W=PARAM_LAMBDA_W,
                                 PARAM_LAMBDA_ANNOTATIONS=PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA=PARAM_LAMBDA,
                                 TIMESTEP=TIMESTEP, method=method)
        print "******************TESTING******************"
        loss_cv.append(train_model.test(test_data[0], test_data[1], test_data[2], test_data[3], PARAM_LAMBDA_ANNOTATIONS, method))
        print loss_cv[test_i]
    print np.mean(loss_cv)

def test_synthetic_mnist(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, method, num_experts, arg_iterations, stochastic_size):
#    np.random.seed(1337)
    all_data = load_data_mnist(num_experts)
    print "data loaded"
    num_cv = 2
    cv_indices = divide_indices_mnist(all_data, num_cv)
    print "indices divided"
    loss_cv = []
    num_clients = len(all_data[2])
    tests = []
    for test_i in range(num_cv):
        print "Test {0}".format(test_i)
        training_data, test_data = divide_data_synthetic(all_data, cv_indices, test_i)

#        training_data[1] = np.where(training_data[1]>0.5, 1, 0)

#        for cl_i in range(num_clients):
#            training_data[3][cl_i] = np.where(np.array(training_data[3][cl_i]) > 0.5, 1, 0)

        [n, m] = all_data[0].shape
        random_v = np.random.normal(1, 0.5, num_clients) / num_clients
        random_v = random_v / np.sum(random_v)
        num_classes = 10
        multiclass = True

        if (multiclass):
            random_w = np.random.normal(0, 0.0000005, (num_clients, m, num_classes))
        else:
            random_w = np.random.normal(0, 0.5, (num_clients, m))
        train_model = GenGradDescModelNoAnnotations(random_w, random_v, multiclass=True, num_classes=10, stochastic=True)
        print "Converting inputs to binary..."
        y_gt_train = np.zeros((np.size(training_data[1],0), num_classes))
        y_annotations_train = []
        for i in range(np.size(training_data[1],0)):
            y_gt_train[i,int(training_data[1][i][0])] = 1
        for c in range(len(training_data[3])):
            len_dat = np.size(training_data[3][c])
            y_annot = np.zeros((len_dat,num_classes))
            for i in range(len_dat):
                y_annot[i,int(training_data[3][c][i])] = 1
            y_annotations_train.append(y_annot)

        y_gt_test = np.zeros((np.size(test_data[1],0), num_classes))
        y_annotations_test = []
        for i in range(np.size(test_data[1],0)):
            y_gt_test[i, int(test_data[1][i][0])] = 1
        for c in range(len(test_data[3])):
            len_dat = np.size(test_data[3][c])
            y_annot = np.zeros((len_dat,num_classes))
            for i in range(len_dat):
                y_annot[i, int(test_data[3][c][i])] = 1
            y_annotations_test.append(y_annot)

        
        print "******************TRAINING******************"
        [train_loss, v_nonzero, error_percentage]=train_model.optimization(training_data[0], y_gt_train, training_data[2], y_annotations_train, NUM_IT=arg_iterations,
                                 NUM_IT_P=NUM_IT_P, PARAM_LAMBDA_W=PARAM_LAMBDA_W,
                                 PARAM_LAMBDA_ANNOTATIONS=PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA=PARAM_LAMBDA,
                                                                           TIMESTEP=TIMESTEP, method=method, stochastic_size=stochastic_size)
        print "******************TESTING******************"
        loss_cv.append(train_model.test(test_data[0], y_gt_test, test_data[2], y_annotations_test, PARAM_LAMBDA_ANNOTATIONS, method))
        tests.append((train_loss, v_nonzero, error_percentage, loss_cv[test_i]))
        print loss_cv[test_i]
    print np.mean(loss_cv)
    return tests

def test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_V, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, method, task_id, NUM_IT):
#    np.random.seed(42)
    NUM_IT_W = 10
    loss_data = []
    start_task = 0
    end_task = 9
    if task_id!=10:
        start_task = task_id
        end_task = task_id
    task_results = []
    for data_number in range(start_task, end_task+1): # which dataset

        cv_number = 2

        (X,L,Y) = load_data_uai(data_number+1)

        indices = divide_indices_uai(cv_number, np.size(X,0))
        loss_cv = []

        print "TEST: {0}".format(data_number+1)
        tests = []
        for i in range(cv_number):

            (training_data, test_data) = divide_data_uai_no_annotations(X,L,Y,indices,i)
            [n,m] = X.shape
            random_v = np.random.normal(1,0.3, 21)/21
#            random_v = np.ones(21)
            random_v = random_v/np.sum(random_v)

            random_w = np.random.normal(0, 0.005, (21,m))
            train_model = GenGradDescModelNoAnnotations(random_w, random_v)
            print "******************TRAINING******************"
            [train_loss, v_nonzero, error_percentage] = train_model.optimization(training_data[0], training_data[1], training_data[2], training_data[3], NUM_IT=NUM_IT, NUM_IT_W=NUM_IT_W, NUM_IT_V=NUM_IT_V, PARAM_LAMBDA_W = PARAM_LAMBDA_W, PARAM_LAMBDA_ANNOTATIONS=PARAM_LAMBDA_ANNOTATIONS, PARAM_LAMBDA=PARAM_LAMBDA, TIMESTEP=TIMESTEP, method=method)
            print "******************TESTING******************"
            loss_cv.append(train_model.test(test_data[0], test_data[1], test_data[2], test_data[3], PARAM_LAMBDA_ANNOTATIONS, method))
            print loss_cv[i]
            #pickle.dump((train_loss, v_nonzero, error_percentage), open("iterations_out.bin", "wb+"))
            tests.append((train_loss, v_nonzero, error_percentage, loss_cv[i]))
        task_results.append(tests)
        loss_data.append(np.mean(loss_cv))
    return task_results

if __name__=="__main__":

    arg_data = sys.argv[1]
    arg_method = sys.argv[2]
    arg_3 = int(sys.argv[3])
    arg_iterations = int(sys.argv[4])
#    if (len(sys.argv)>5):
#        stochastic_size = float(sys.argv[5])

    test = arg_data
    method = arg_method

    np.random.seed(42)# for reproducibility


    if (test == 'real'):
        task_id = arg_3 # if task id ==10 >>> test all

 


        if (method == 'logreg_all'):
            print "Testing all parameter configurations, takes some days"

            PARAM_LAMBDA_ANNOTATIONS_A = [0.01, 0.1, 1 ]
            NUM_IT_P_A = [1, 5, 10]
            PARAM_LAMBDA_W_A = [0.000001, 0.00001, 0.0001, 0.001]
            TIMESTEP_A = [0.00001, 0.000001, 0.0000001, 0.00000001]
            PARAM_LAMBDA_A = [0.00001, 0.0001, 0.001, 0.01]

            for PARAM_LAMBDA_ANNOTATIONS in PARAM_LAMBDA_ANNOTATIONS_A:
                for NUM_IT_P in NUM_IT_P_A:
                    for PARAM_LAMBDA_W in PARAM_LAMBDA_W_A:
                        for TIMESTEP in TIMESTEP_A:
                            for PARAM_LAMBDA in PARAM_LAMBDA_A:
                                sys.stdout = open('{0}_{1}_{2}_{3}_{4}.txt'.format(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA, TIMESTEP, PARAM_LAMBDA), 'w')
                                test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA)

        elif method== 'logreg':
            print "Fast test  of logreg with manually assigned parameters"

            if (len(sys.argv)>5):
                PARAM_LAMBDA = float(sys.argv[5])

                if (len(sys.argv)>6):                                                                                                                                                                                  
                    NUM_IT_V = int(sys.argv[6])       
                else:
                    NUM_IT_V = 10                                
            else:
                PARAM_LAMBDA = 0.00001
                NUM_IT_V = 10            

            PARAM_LAMBDA_ANNOTATIONS = 1000
#            NUM_IT_V = 10
            PARAM_LAMBDA_W = 0.0001
            TIMESTEP = 0.0001
            PARAM_LAMBDA = 0.00001
#            if (len(sys.argv)>4):
#                PARAM_LAMBDA = arg_lambda
            #sys.stdout = open('output_real_{0}_{1}_{2}.txt'.format(method, task_id, arg_iterations),'w+')
            print "Test..."
            for i in range(1): # make multiple crossvalidation tests
                task_results = test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_V, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'LOGREG', task_id, arg_iterations)
                pickle.dump(task_results, open("results_logreg_{0}_{1}.bin".format(PARAM_LAMBDA, i), 'wb+'))
        elif method=='mlp':
            print "Fast test  of mlp with manually assigned parameters"

            if (len(sys.argv)>5):
                PARAM_LAMBDA = float(sys.argv[5])

                if (len(sys.argv)>6):                                                                                                                                                                      \

                    NUM_IT_V = int(sys.argv[6])
                else:
                    NUM_IT_V = 10
            else:
                PARAM_LAMBDA = 0.00001
                NUM_IT_V = 10

            PARAM_LAMBDA_ANNOTATIONS = 1000
#            NUM_IT_P = 10
            PARAM_LAMBDA_W = 0.0001
            TIMESTEP = 0.000001
            PARAM_LAMBDA = 0.00001
            #sys.stdout = open('output_real_{0}_{1}_{2}.txt'.format(method, task_id, arg_iterations), 'w+')
            for i in range(10):
                task_results = test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_V, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'MLP', task_id, arg_iterations)
                pickle.dump(task_results, open("results_mlp_{0}_{1}.bin".format(PARAM_LAMBDA,i), 'wb+'))


        else:
            print "Please pick logreg or mlp"

    elif (test == 'synthetic'):
        PARAM_LAMBDA_ANNOTATIONS = 100
        NUM_IT_P = 5
        PARAM_LAMBDA_W = 0.000001
        TIMESTEP = 0.00000001
        PARAM_LAMBDA = 0.000001
        # test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'LOGREG')
        sys.stdout = open(
            'output_synthetic_{0}_{1}_{2}.txt'.format(method, task_id, arg_iterations), 'w+')

        test_synthetic(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'LOGREG', task_id, arg_iterations)

    elif (test =='mnist'):

        if (len(sys.argv)>5):                                                                                                                                                                                  
            stochastic_size = float(sys.argv[5])       

        print "testing mnist..."
        PARAM_LAMBDA_ANNOTATIONS = 1000
        NUM_IT_P = 5
        PARAM_LAMBDA_W = 0.0000001
        TIMESTEP = 0.00000001
        PARAM_LAMBDA = 0.001
        # test_optimization_no_annotations(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'LOGREG')
        #sys.stdout = open(
        #    'output_mnist_{0}_{1}_{2}.txt'.format(method, task_id, arg_iterations), 'w+')

        num_experts = arg_3

        task_results = test_synthetic_mnist(PARAM_LAMBDA_ANNOTATIONS, NUM_IT_P, PARAM_LAMBDA_W, TIMESTEP, PARAM_LAMBDA, 'LOGREG', num_experts, arg_iterations, stochastic_size)
        pickle.dump(task_results, open("results_mnist_{0}.bin".format(num_experts), 'wb+'))



    else:
        print "Select a test!"