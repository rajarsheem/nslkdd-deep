import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

train_dataset = new_input1 = np.load('out/new_input.npz')['train']


def one_hot(yy):
    lb = LabelBinarizer()
    yy = lb.fit(yy).transform(yy)
    return yy


def compute_class_weight(y):
    y = y.astype(np.int32)
    # print(np.bincount(y))
    # w = (len(y) / np.bincount(y))) + 1)
    # print(w)
    # w = MinMaxScaler(feature_range=(1, 2)).fit_transform(w.reshape(-1,1)).flatten()
    # print(w)
    w = np.array([1,1,1.1,1.2,1.5],dtype=np.float32)
    r = np.array([w[i] for i in y]) + 1
    return r

c_w = compute_class_weight(train_dataset[:, -1]-1)


def recompute():
    global c_w
    c_w = compute_class_weight(train_dataset[:, -1]-1)


def get_batch_indices(r, class_size):
    n, k = [], []
    for i in r:
        n.append(len(i))
    for j, i in enumerate(n):
        temp = np.random.choice(i, class_size, replace=False)
        k.extend(r[j][temp])
    return k


def train_batch_data(set_size):
    train_batch_data.targets = train_dataset[:, -1]
    train_batch_data.compute = True
    train_batch_data.r = []
    if train_batch_data.compute:
        for i in range(1, 6):
            train_batch_data.r.append(
                np.arange(len(train_batch_data.targets))
                [train_batch_data.targets == i]
            )
    indices = get_batch_indices(train_batch_data.r, set_size)
    sample = train_dataset[indices]
    return sample[:, :-1], one_hot(sample[:, -1]), c_w[indices]


def fulldata(x):
    data = np.load('out/new_input.npz')[x]
    return data[:, :-1], one_hot(data[:, -1])

# data = np.genfromtxt('kddtrain_2class_normalized.csv',delimiter=',',skip_header=True)
# data = np.load('new_input.npz')['train']
# train_x, train_y = train_batch_data(data,40)
# print(train_x.shape, train_y.shape)
# testdata()
