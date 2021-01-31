from sklearn.linear_model import LogisticRegression as LogReg
import scipy.io as sio
from sklearn.model_selection import KFold
import numpy as np

lr_clf = LogReg()

file_in = sio.loadmat('../../data/data_in.mat')

Xt = file_in['Xt']
Y_sigmoid = file_in['Y_sigmoid']
Z_sigmoid = file_in['Z_sigmoid']
Y_labels = file_in['Y_labels']
Z_labels = file_in['Z_labels']

skf = KFold(n_splits=5)
print skf.get_n_splits(Xt, Z_labels)

for train_index, test_index in skf.split(Xt, Z_labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = Z_labels[train_index], Z_labels[test_index]
    lr_clf.fit(X_train, y_train)
    y_gen = lr_clf.predict(X_test)
    print y_gen
    print y_test.T










