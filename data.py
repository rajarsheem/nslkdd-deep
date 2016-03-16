import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelBinarizer


def one_hot(yy):
    lb = LabelBinarizer()
    yy = lb.fit(yy).transform(yy)
    return yy


def get_batch_indices(r, class_size):
    n, k = [], []
    for i in r:
        n.append(len(i))
    for j, i in enumerate(n):
        temp = np.random.choice(i, class_size, replace=False)
        k.extend(r[j][temp])
    return k


def train_batch_data(data, set_size):
    targets = data[:, -1]
    r = []
    for i in range(1, 6):
        r.append(np.arange(len(targets))[targets == i])
    sample = data[get_batch_indices(r, set_size)]
    return sample[:, :-1], one_hot(sample[:, -1])


def testdata():
    # data = np.genfromtxt('KDDTest+_normalized.csv', delimiter=',', skip_header=True)
    data = np.load('out/new_input.npz')['test']
    return data[:, :-1], one_hot(data[:, -1])

# data = np.genfromtxt('kddtrain_2class_normalized.csv',delimiter=',',skip_header=True)
# data = np.load('new_input.npz')['train']
# train_x, train_y = train_batch_data(data,40)
# print(train_x.shape, train_y.shape)
# testdata()
