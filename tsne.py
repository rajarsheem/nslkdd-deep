import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mp
from sklearn.manifold import TSNE
data = np.genfromtxt('kddtrain_2class_normalized.csv',delimiter=',',dtype=np.float64,skip_header=True)
data = data[:20000]
labels = ['norm', 'dos', 'probe', 'r2l', 'u2r']

#print(type(data))
# for row in data:
#     print(len(row))
model = TSNE(n_components=2, init='pca',random_state=0)
np.set_printoptions(suppress=True)
colors = ['red','green','blue','purple', 'yellow']
tdata = model.fit_transform(data)
x = tdata[:, 0]
y = tdata[:, 1]
target = data[:,-1]
class_set = set(target)
print(len(class_set))
for klass in class_set:
    con_indices = target == klass
    pp.plot(x[con_indices], y[con_indices], linestyle='none', marker='o', label=int(klass))
pp.legend(numpoints=1)
pp.show()
