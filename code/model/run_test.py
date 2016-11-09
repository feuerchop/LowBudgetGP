__author__ = 'kolosnjaji'

import numpy as np
import h5py
from training import LowBudgetCrowdsourcing
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from optimization_no_annotations import GenGradDescModelNoAnnotations

def load_data_uai(data_i):

    labels_list = []
    data_list = []
    groundtruth_list = []

    print data_i
    f = h5py.File('../../data/UAI14_data/class_data_{0}.mat'.format(data_i), 'r')
    labels_list.append(f['L'][:].T)
    data_list.append(f['x'][:].T)
    groundtruth_list.append(f['Y'][:].T)

    #print "Stacking..."
    my_labels = np.hstack(labels_list).T
    my_data = np.vstack(data_list)
    my_gt = np.vstack(groundtruth_list)

    return my_data, my_labels, my_gt

def divide_indices_uai(num_cv=2, num_data=3220):

    num_test = num_data/float(num_cv) # number of data for the test set
    random_indices = np.random.permutation(range(num_data))

    cv_indices = []
    for i in range(num_cv):
        if (i+1)*num_test<len(random_indices):
            max_num = (i+1)*num_test
        else:
            max_num = len(random_indices)

        test_indices = random_indices[i*num_test:max_num]
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


def test_optimization_no_annotations():
    loss_data = []
    for data_number in range(10): # which dataset

        cv_number = 3

        (X,L,Y) = load_data_uai(data_number+1)

        indices = divide_indices_uai(cv_number, np.size(X,0))
        loss_cv = []
        for i in range(cv_number):

            (training_data, test_data) = divide_data_uai_no_annotations(X,L,Y,indices,i)
            [n,m] = X.shape
            random_v = np.random.normal(1,0.5, 21)/21
            random_v = random_v/np.sum(random_v)
            random_w = np.random.normal(0, 0.0005, (21,m))
            train_model = GenGradDescModelNoAnnotations(random_w, random_v)
            print "******************TRAINING******************"
            train_model.optimization(training_data[0], training_data[1], training_data[2], training_data[3], NUM_IT=1000, NUM_IT_P=1)
            print "******************TESTING******************"
            loss_cv.append(train_model.test(test_data[0], test_data[1], test_data[2], test_data[3]))
            print loss_cv[i]
        loss_data.append(np.mean(loss_cv))

if __name__=="__main__":
    test_optimization_no_annotations()