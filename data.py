import numpy as np
from collections import Counter


def one_hot(yy, labels=None):
    if labels == None:
        fset = set((yy))
    else:
        fset = set(labels)
    onehotmap = {x: list(map(int, list(bin(2 ** y)[2:].zfill(len(fset))))) for y, x in enumerate(fset)}
    lst = [onehotmap[x] for x in yy]
    return lst


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
    return sample[:, :-1], one_hot(sample[:, -1], [1, 2, 3, 4, 5])


def testdata():
    data = np.genfromtxt('KDDTest+_normalized.csv', delimiter=',', skip_header=True)
    return data[:, :-1], one_hot(data[:, -1])

# data = np.genfromtxt('kddtrain_2class_normalized.csv',delimiter=',',skip_header=True)
# testdata()