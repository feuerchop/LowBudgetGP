import numpy as np
import h5py
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

labels_list = []
print "Reading data..."
for i in range(5):
    print i
    f = h5py.File('/home/bojan/research/datasets/UAI14_data/class_data_{0}.mat'.format(i+1), 'r')
    labels_list.append(f['L'][:].T)
    print np.size(f['L'][:].T,1)
print "Stacking..."
labels = np.hstack(labels_list)
labels[labels==-100] = 0.5
print "{0} {1}".format(np.size(labels,0), np.size(labels,1))
print "T-SNE..."
tsne_labels = TSNE(n_components=2, random_state=0).fit_transform(labels)

tsne_clusters = DBSCAN(eps=0.3, min_samples=2).fit_predict(labels)

print tsne_clusters
print "Plotting..."
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(tsne_labels[:,0], tsne_labels[:,1], tsne_labels[:,2])
#plt.show()
plt.plot(tsne_labels[:,0], tsne_labels[:,1], '.')
plt.show()