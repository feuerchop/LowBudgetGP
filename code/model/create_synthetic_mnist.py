import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import KFold
import pickle

mnist_data = sio.loadmat('../../data/mldata/mnist-original.mat')

X = mnist_data['data'].T
y = mnist_data['label'].T

kf = KFold(n_splits=10)

percent_training = 0.5
num_experts = 2
num_data = len(y)

y_annotations = np.zeros((num_experts, num_data))

for expert in range(num_experts):
    my_clf = LogReg(n_jobs=4)
    num_train_subset = int(num_data*percent_training/100.0)
    print num_train_subset
    train_indices = np.random.randint(0, num_data, num_train_subset, dtype=np.int)
    X_train = X[train_indices,:]
    y_train = y[train_indices,:]
    print "training..."
    my_clf.fit(X_train, np.ravel(y_train))
    print my_clf.score(X,y)
    y_annotations[expert,:] = my_clf.predict(X)

dict_mnist = {}
dict_mnist['X'] = X
dict_mnist['y'] = y
dict_mnist['y_annotations'] = y_annotations

#pickle.dump(dict_mnist, open('../../data/mnist_dict_annotations_{0}.bin'.format(num_experts), 'wb+'))
sio.savemat('../../data/mnist_dict_annotations_{0}.mat'.format(num_experts), dict_mnist)














