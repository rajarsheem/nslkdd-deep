import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mp
from sklearn.manifold import TSNE

def tsne_plot(data,sample_size):
    data = data[10000:10000+sample_size]
    data_x=data[:, :-1]
    labels = ['norm', 'dos', 'probe', 'r2l', 'u2r']
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    colors = ['red','green','blue','purple', 'yellow']
    tdata = model.fit_transform(data_x)
    x = tdata[:, 0]
    y = tdata[:, 1]
    target = data[:,-1]
    class_set = set(target)
    n_class=len(class_set)
    for klass in class_set:
        con_indices = target == klass
        pp.plot(x[con_indices], y[con_indices], linestyle='none', marker='o', label=int(klass))
    pp.legend(numpoints=1)
    pp.show()

#original data
# data = np.genfromtxt('train.csv',delimiter=',',dtype=np.float64,skip_header=True)
# data with learned features
data=np.load('new_input.npz')['train']
tsne_plot(data,10000)
