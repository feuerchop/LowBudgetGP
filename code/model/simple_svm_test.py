from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score

#X
#y
data_in = pickle.load(open('../../../VT_data/data_matrices.bin', 'rb'))

data_x = data_in[0]
data_z = np.asarray(data_in[2])

print data_z

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, data_x, data_z, cv=3)
print scores

